#!/usr/bin/env python
"""
6_train_ce.py
────────────────────────────────────────────────────────────
Pair-BERT(문장쌍 Cross-Encoder) 학습 + test 추론 스크립트

예시 ── 7-fold CV 학습 + 테스트 예측
python 6_train_ce.py \
  --train      data/proc/pair_train.jsonl \
  --cv --fold_dir data/fold_pair \
  --cfg        first_sentence_kobert/cfg/pair_roberta.yaml \
  --exp        pair_roberta_cv \
  --test_jsonl data/proc/test_pair.jsonl \
  --out_jsonl  data/proc/test_first_top1.jsonl
"""
import argparse, json, os, random, pathlib, yaml, gc
from typing import Optional, Set, List

import numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ───────────────────────── 설정 ─────────────────────────
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────── 인코딩/평가 util ─────────────────────
def enc_factory(tok, max_len: int):
    def _enc(batch):
        enc = tok(batch["text_a"], batch["text_b"],
                  truncation=True, max_length=max_len,
                  return_token_type_ids=False)
        enc["labels"] = batch["label"]
        return enc
    return _enc


def metric_fn(pred):
    y_true, y_pred = pred.label_ids, pred.predictions.argmax(-1)
    p,r,f1,_ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": p, "recall": r, "f1": f1}


class CETrainer(Trainer):
    _loss = torch.nn.CrossEntropyLoss()
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        lab = inputs.pop("labels")
        out = model(**inputs)
        loss = self._loss(out.logits, lab)
        return (loss, out) if return_outputs else loss


# ────────────────────── 학습 1회 ──────────────────────────
def train_once(cfg: dict,
               train_json: str,
               valid_json: str,
               exp_dir: pathlib.Path,
               exclude_ids: Optional[Set[int]] = None) -> float:
    """
    return: eval F1 (float) — trainer 등은 메모리 해제 후 폐기
    """
    set_seed()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(
              cfg["model_name"], num_labels=2).to(DEVICE)

    raw_tr = load_dataset("json", data_files=train_json)["train"]
    if exclude_ids:
        raw_tr = raw_tr.filter(lambda ex: ex["doc_id"] not in exclude_ids)

    ds_tr = (raw_tr.shuffle(seed=SEED)
                   .map(enc_factory(tok, cfg["max_len"]),
                        batched=True,
                        remove_columns=["text_a", "text_b", "doc_id"]))
    ds_va = (load_dataset("json", data_files=valid_json)["train"]
                   .map(enc_factory(tok, cfg["max_len"]),
                        batched=True,
                        remove_columns=["text_a", "text_b", "doc_id"]))

    exp_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(exp_dir),
        seed=SEED,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = CETrainer(
        model=mdl,
        args=targs,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        compute_metrics=metric_fn,
    )
    trainer.train()

    # ── best 모델 저장 ─────────────────────────────────────
    best_dir = exp_dir / "best"
    best_dir.mkdir(exist_ok=True)
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)

    f1_val = trainer.evaluate()["eval_f1"]

    # ── >>> 메모리 즉시 해제 <<< ────────────────────────────
    del trainer, mdl, tok, ds_tr, ds_va, raw_tr
    torch.cuda.empty_cache()
    gc.collect()

    return f1_val


# ────────────────────── 테스트 예측 util ───────────────────
def encode_test(tok, path: str, max_len: int):
    raw = load_dataset("json", data_files=path)["train"]
    def _enc(b):
        enc = tok(b["text_a"], b["text_b"],
                  truncation=True, max_length=max_len,
                  return_token_type_ids=False)
        enc["doc_id"] = b["doc_id"]
        return enc
    return raw.map(_enc, batched=True)


def ensemble_logits(ck_root: pathlib.Path, prefix: str,
                    tok, ds_tok, batch: int = 256) -> np.ndarray:
    bests = sorted((p / "best") for p in ck_root.glob(f"{prefix}_fold*/"))
    if not bests:
        bests = [ck_root / prefix / "best"]

    loader = torch.utils.data.DataLoader(
        ds_tok.remove_columns(["doc_id", "text_a", "text_b"]),
        batch_size=batch,
        collate_fn=DataCollatorWithPadding(tok),
    )
    tot = torch.zeros(len(ds_tok), 2)
    for bd in bests:
        mdl = AutoModelForSequenceClassification.from_pretrained(bd).to(DEVICE).eval()
        preds = []
        for bt in loader:
            preds.append(mdl(**{k:v.to(DEVICE) for k,v in bt.items()}).logits.cpu())
        tot += torch.cat(preds, 0)
        del mdl
        torch.cuda.empty_cache()
        gc.collect()

    return (tot / len(bests)).numpy()


# ─────────────────────────── main ──────────────────────────
def main(args):
    cfg = yaml.safe_load(open(args.cfg, encoding="utf-8"))
    ck_root = pathlib.Path("first_sentence_kobert/checkpoints")

    # 1) 학습
    if args.cv:
        fold_files: List[pathlib.Path] = sorted(
            pathlib.Path(args.fold_dir).glob("pair_fold_*.jsonl"))
        if not fold_files:
            raise FileNotFoundError(f"{args.fold_dir}/pair_fold_*.jsonl not found")

        f1s = []
        for k, vf in enumerate(fold_files):
            vid = set(pd.read_json(vf, lines=True)["doc_id"])
            f1 = train_once(cfg,
                            args.train, str(vf),
                            ck_root / f"{args.exp}_fold{k}",
                            exclude_ids=vid)
            f1s.append(f1)
            print(f"Fold{k}  F1={f1:.4f}")
        print(f"▶ CV mean F1 = {np.mean(f1s):.4f}")

    else:  # single-run
        if not args.valid:
            raise ValueError("--valid 경로 필요")
        f1 = train_once(cfg, args.train, args.valid, ck_root / args.exp)
        print(f"▶ Single valid F1 = {f1:.4f}")

    # 2) 테스트 예측
    if args.test_jsonl:
        print("\n[ Predicting test … ]")
        first_best = (ck_root / f"{args.exp}_fold0/best") if args.cv else (ck_root / args.exp / "best")
        tok = AutoTokenizer.from_pretrained(first_best, use_fast=False)

        ds_test = encode_test(tok, args.test_jsonl, cfg["max_len"])
        logits  = ensemble_logits(ck_root, args.exp, tok, ds_test, batch=256)
        prob    = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

        out = []
        for p, doc in zip(prob, ds_test["doc_id"]):
            out.append(json.dumps({"doc_id": int(doc),
                                   "idx": 0 if p > 0.5 else 1},
                                  ensure_ascii=False))
        out_p = pathlib.Path(args.out_jsonl)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text("\n".join(out), encoding="utf-8")
        print("✅  Saved:", out_p)


# ──────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid")                         # single-run 전용
    ap.add_argument("--cfg",   required=True)
    ap.add_argument("--exp",   required=True)
    ap.add_argument("--cv",    action="store_true")    # K-Fold
    ap.add_argument("--fold_dir", default="data/fold_pair")
    # 테스트 예측
    ap.add_argument("--test_jsonl")
    ap.add_argument("--out_jsonl", default="data/proc/test_first_top1.jsonl")
    main(ap.parse_args())
