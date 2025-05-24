#!/usr/bin/env python
"""
6_train_ce.py – Pair-BERT 학습 + rank-0 전용 추론
(✓ wide/long 형식 test.csv 자동 인식)
"""
import argparse, json, os, random, pathlib, yaml, gc, inspect, re
from typing import Optional, Set, List, Dict, Tuple

import numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

# ──────────────── 환경 · rank ────────────────────────────
SEED = 42
WORLD_GPUS = torch.cuda.device_count()
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
is_main = LOCAL_RANK == 0

TRAIN_DEVICE = torch.device("cuda", LOCAL_RANK) if WORLD_GPUS else torch.device("cpu")
INFER_DEVICE = TRAIN_DEVICE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if WORLD_GPUS: torch.cuda.manual_seed_all(seed)

# ───── transformers 버전에 따른 argument key 픽 ─────
TA_SIG = set(inspect.signature(TrainingArguments.__init__).parameters)
KEY_EVAL = "evaluation_strategy" if "evaluation_strategy" in TA_SIG else "eval_strategy"
KEY_SAVE = "save_strategy"        if "save_strategy"        in TA_SIG else "save_steps"
KEY_LOG  = "logging_strategy"     if "logging_strategy"     in TA_SIG else "logging_steps"

# ──────────── 인코딩 · metric util ─────────────────────
def enc_factory(tok, max_len:int):
    def _enc(b):
        e = tok(b["text_a"], b["text_b"],
                truncation=True, max_length=max_len,
                return_token_type_ids=False)
        if "label" in b:                 # 학습·검증
            e["labels"] = b["label"]
        for k in ("doc_id", "candidate_idx"):
            if k in b: e[k] = b[k]
        return e
    return _enc

def metric_fn(pred):
    y, yhat = pred.label_ids, pred.predictions.argmax(-1)
    p,r,f1,_ = precision_recall_fscore_support(
        y, yhat, average="binary", zero_division=0)
    return {"accuracy":accuracy_score(y, yhat),
            "precision":p, "recall":r, "f1":f1}

class CETrainer(Trainer):
    _loss = torch.nn.CrossEntropyLoss()
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        lbl = inputs.pop("labels"); out = model(**inputs)
        loss = self._loss(out.logits, lbl)
        return (loss, out) if return_outputs else loss

# ──────────── 학습 1회 함수 ───────────────────────────
def train_once(cfg, tr_json, va_json, exp_dir,
               exclude_ids:Optional[Set[int]]=None)->float:
    set_seed()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    mdl = (AutoModelForSequenceClassification
           .from_pretrained(cfg["model_name"], num_labels=2)
           .to(TRAIN_DEVICE))

    raw_tr = load_dataset("json", data_files=tr_json)["train"]
    if exclude_ids:
        raw_tr = raw_tr.filter(lambda ex: ex["doc_id"] not in exclude_ids)

    ds_tr = (raw_tr.shuffle(seed=SEED)
                   .map(enc_factory(tok, cfg["max_len"]), batched=True,
                        remove_columns=["text_a","text_b","doc_id"]))
    ds_va = (load_dataset("json", data_files=va_json)["train"]
                   .map(enc_factory(tok, cfg["max_len"]), batched=True,
                        remove_columns=["text_a","text_b","doc_id"]))

    ta = TrainingArguments(
        output_dir=str(exp_dir),
        seed=SEED,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        fp16=bool(WORLD_GPUS),
        load_best_model_at_end=True, metric_for_best_model="f1",
        ddp_find_unused_parameters=False,
        report_to="none",
        **{KEY_EVAL:"epoch", KEY_SAVE:"epoch", KEY_LOG:"epoch"}
    )

    trainer = CETrainer(model=mdl, args=ta, tokenizer=tok,
                        data_collator=DataCollatorWithPadding(tok),
                        train_dataset=ds_tr, eval_dataset=ds_va,
                        compute_metrics=metric_fn)
    trainer.train()

    if is_main:
        best = exp_dir/"best"; best.mkdir(exist_ok=True)
        trainer.save_model(best); tok.save_pretrained(best)

    f1 = trainer.evaluate()["eval_f1"]

    del trainer, mdl, tok, ds_tr, ds_va, raw_tr
    if WORLD_GPUS: torch.cuda.empty_cache()
    gc.collect()
    return f1

# ─────────────── top-2 → pair 변환 ─────────────────────
def _extract_sent_dict(df: pd.DataFrame,
                       id_col_candidates: List[str]) -> Dict[str, List[str]]:
    """wide·long 모두 지원하여 {ID: [sent0, sent1, …]} 반환"""
    # 0) 컬럼 이름에서 BOM(﻿)·앞뒤 공백 제거
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # 1) ID 컬럼 찾기
    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    if id_col is None:
        raise ValueError(
            "test.csv 에 ID 컬럼이 없습니다.\n"
            f"  헤더: {list(df.columns)}\n"
            f"  후보: {id_col_candidates}"
        )

    # 2-A) wide 형식? → sentence_\d+ 열 찾기
    sent_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"sentence_\d+", c, re.I)],
        key=lambda x: int(x.split("_")[1]),
    )
    if sent_cols:
        return {
            row[id_col]: [row[c] for c in sent_cols if pd.notna(row[c])]
            for _, row in df.iterrows()
        }

    # 2-B) long 형식? sentence / text / sent 단일 열
    for c in ["sentence", "text", "sent"]:
        if c in df.columns:
            return df.groupby(id_col)[c].apply(list).to_dict()

    raise ValueError(
        "test.csv 에 문장 컬럼( sentence_*, sentence, text … )을 찾지 못했습니다."
    )

def build_pairs_from_top2(top2_path: str, csv_path: str,
                          out_path: str) -> Tuple[str, Dict[str, int]]:
    """top-2(rank) 파일 + test.csv → Pair-BERT 입력 JSONL"""
    # CSV 읽기 (UTF-8-SIG → BOM 자동 제거, 구분자 자동 감지)
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    sent_dict = _extract_sent_dict(df, ["ID", "id", "doc_id"])

    top2_ds = load_dataset("json", data_files=top2_path)["train"]
    id2num = {aid: n for n, aid in enumerate(sorted({r["ID"] for r in top2_ds}))}

    out_lines = []
    for ex in top2_ds:
        aid, (i, j) = ex["ID"], ex["rank"]
        sents = sent_dict[aid]
        # candidate_idx 0 → text_a 가 i번째 문장
        out_lines.append({"doc_id": aid, "text_a": sents[i], "text_b": sents[j],
                          "candidate_idx": 0, "first_idx": i})
        # candidate_idx 1 → text_a 가 j번째 문장
        out_lines.append({"doc_id": aid, "text_a": sents[j], "text_b": sents[i],
                          "candidate_idx": 1, "first_idx": j})

    tmp = pathlib.Path(out_path)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out_lines),
                   encoding="utf-8")
    return str(tmp), id2num

# ─────────────── 테스트 인코딩 · 추론 ─────────────────────
def encode_test(tok, path: str, max_len: int):
    raw = load_dataset("json", data_files=path)["train"]

    doc_ids     = raw["doc_id"]
    cand_idxs   = raw["candidate_idx"]
    first_idxs  = raw["first_idx"]          # ← 새 필드

    ds_enc = raw.map(enc_factory(tok, max_len), batched=True)
    ds_enc = ds_enc.remove_columns(
        ["text_a", "text_b", "doc_id", "candidate_idx", "first_idx"]
    )
    return ds_enc, doc_ids, cand_idxs, first_idxs

def load_model(bdir: pathlib.Path, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return (AutoModelForSequenceClassification
            .from_pretrained(bdir, torch_dtype=dtype)
            .to(device).eval())

def predict_ensemble(ck_root, prefix, tok, ds_tok, bs: int = 128) -> np.ndarray:
    bests = sorted((p / "best") for p in ck_root.glob(f"{prefix}_fold*/"))
    if not bests:
        bests = [ck_root / prefix / "best"]

    dl = torch.utils.data.DataLoader(
        ds_tok,
        batch_size=bs,
        collate_fn=DataCollatorWithPadding(tok),
        shuffle=False,          # ← 순서를 고정해야 인덱스가 맞습니다
        drop_last=False,
    )

    total = torch.zeros(len(ds_tok), 2, device=INFER_DEVICE, dtype=torch.float32)

    for bidx, bdir in enumerate(bests, 1):
        mdl = load_model(bdir, INFER_DEVICE)
        bar = tqdm(dl, desc=f"[{bidx}/{len(bests)}] {bdir.parent.name}", disable=not is_main)

        ptr = 0  # 누적 시작 인덱스
        with torch.no_grad():
            for bt in bar:
                bsz = bt["input_ids"].size(0)  # 현재 배치 크기
                bt = {k: v.to(INFER_DEVICE, non_blocking=True) for k, v in bt.items()}

                logits = mdl(**bt).logits          # [bsz, 2]
                total[ptr:ptr + bsz] += logits     # 해당 구간에만 더하기
                ptr += bsz                         # 다음 배치 위치로 이동

        del mdl
        gc.collect()
        if WORLD_GPUS:
            torch.cuda.empty_cache()

    return (total / len(bests)).cpu().numpy()

# ─────────────────────────── main ───────────────────────
def main(a):
    cfg = yaml.safe_load(open(a.cfg, encoding="utf-8"))
    ck_root = pathlib.Path("first_sentence_kobert/checkpoints")

    # 1) 학습
    if a.cv:
        folds = sorted(pathlib.Path(a.fold_dir).glob("pair_fold_*.jsonl"))
        if not folds: raise FileNotFoundError("pair_fold_*.jsonl not found")
        f1s=[]
        for k, fp in enumerate(folds):
            ids = set(pd.read_json(fp, lines=True)["doc_id"])
            f1 = train_once(cfg, a.train, str(fp),
                            ck_root/f"{a.exp}_fold{k}", exclude_ids=ids)
            if is_main: print(f"Fold{k} F1={f1:.4f}")
            f1s.append(f1)
        if is_main: print(f"▶ CV mean F1 = {np.mean(f1s):.4f}")
    else:
        if not a.valid: raise ValueError("--valid 필요")
        f1 = train_once(cfg, a.train, a.valid, ck_root/a.exp)
        if is_main: print(f"▶ Single valid F1 = {f1:.4f}")

    # 2) rank-0 추론
    if is_main and (a.test_jsonl or a.test_top2_jsonl):
        print("\n[ Predicting test … ]")
        seed_ck = (ck_root/f"{a.exp}_fold0/best") if a.cv else (ck_root/a.exp/"best")
        tok = AutoTokenizer.from_pretrained(seed_ck, use_fast=False)

        id2num = None
        if a.test_top2_jsonl:
            if not a.test_csv:
                raise ValueError("--test_csv 경로가 필요합니다.")
            pair_path, id2num = build_pairs_from_top2(
                a.test_top2_jsonl, a.test_csv, "tmp/test_pairs_top2.jsonl")
            test_path = pair_path
        else:
            test_path = a.test_jsonl

        ds_t, doc_ids, cand_idxs, first_idxs = encode_test(tok, test_path, cfg["max_len"])
        logits = predict_ensemble(ck_root, a.exp, tok, ds_t, bs=128)
        prob0 = torch.softmax(torch.from_numpy(logits), dim=-1)[:,1].numpy() #반드시 뒷번호 뽑기기

        best: Dict[str, Tuple[int,float]] = {}
        # cand_idx(0·1) 대신 실제 문장 번호(first_idx)를 저장
        for d, fi, p in zip(doc_ids, first_idxs, prob0):
            if (d not in best) or (p > best[d][1]):
                best[d] = (int(fi), float(p))

        out_lines = []
        if id2num:
            for aid in sorted(best.keys(), key=lambda x:id2num[x]):
                out_lines.append(json.dumps(
                    {"doc_id": id2num[aid], "idx": best[aid][0]}, ensure_ascii=False))
        else:
            for aid in sorted(best.keys(), key=int):
                out_lines.append(json.dumps(
                    {"doc_id": int(aid), "idx": best[aid][0]}, ensure_ascii=False))

        out_p = pathlib.Path(a.out_jsonl)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text("\n".join(out_lines), encoding="utf-8")
        print("✅ Saved:", out_p)

# ──────────── CLI ───────────────────────────────────────
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--train", required=True)
    P.add_argument("--valid")
    P.add_argument("--cfg",   required=True)
    P.add_argument("--exp",   required=True)
    P.add_argument("--cv",    action="store_true")
    P.add_argument("--fold_dir", default="data/fold_pair")

    # 테스트
    P.add_argument("--test_jsonl",      help="Pair 포맷 JSONL")
    P.add_argument("--test_top2_jsonl", help="top-2(rank) 포맷 JSONL")
    P.add_argument("--test_csv",        help="원문 문장 풀 CSV (wide 또는 long)")

    P.add_argument("--out_jsonl", default="data/proc/test_first_top1.jsonl")
    main(P.parse_args())