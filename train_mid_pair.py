#!/usr/bin/env python
"""
train_mid_pair.py
────────────────────────────────────────────────────────────
중앙 두 문장(2nd ↔ 3rd) 전용 Pair-BERT 학습 스크립트
• 입력  : pair 형식 JSONL (text_a, text_b, label, doc_id)
• 출력  : <ckpt_root>/<exp_name>_fold*/best/   체크포인트
────────────────────────────────────────────────────────────
실행 예:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 \
  train_mid_pair.py \
    --train_jsonl data/proc_mid/pair_train_mid.jsonl \
    --fold_dir    data/fold_pair_mid \
    --cfg         middle_sentence_kobert/cfg/pair_roberta.yaml \
    --ckpt_root   middle_sentence_kobert/checkpoints \
    --exp_name    pair_roberta_mid_cv
"""

# ─── import ─────────────────────────────────────────────
import argparse, os, random, gc, inspect, json, pathlib, yaml
from typing import Set, Optional

import numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ─── 환경 설정 ───────────────────────────────────────────
SEED = 42
WORLD_GPUS = torch.cuda.device_count()
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
DEVICE = torch.device("cuda", LOCAL_RANK) if WORLD_GPUS else torch.device("cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if WORLD_GPUS: torch.cuda.manual_seed_all(seed)

# transformers 인자 키 (버전별 호환)
TA_SIG = set(inspect.signature(TrainingArguments.__init__).parameters)
KEY_EVAL = "evaluation_strategy" if "evaluation_strategy" in TA_SIG else "eval_strategy"
KEY_SAVE = "save_strategy"        if "save_strategy"        in TA_SIG else "save_steps"
KEY_LOG  = "logging_strategy"     if "logging_strategy"     in TA_SIG else "logging_steps"

# ─── util ───────────────────────────────────────────────
def enc_factory(tok, max_len:int):
    def _enc(b):
        e = tok(b["text_a"], b["text_b"],
                truncation=True, max_length=max_len,
                return_token_type_ids=False)
        if "label" in b:
            e["labels"] = b["label"]
        return e
    return _enc

def metric_fn(pred):
    y, yhat = pred.label_ids, pred.predictions.argmax(-1)
    p,r,f1,_ = precision_recall_fscore_support(
        y, yhat, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y, yhat),
            "precision": p, "recall": r, "f1": f1}

class CETrainer(Trainer):
    _loss = torch.nn.CrossEntropyLoss()
    # **kwargs 로 어떤 추가 인자라도 받아 흘려보내도록 수정
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        lbl = inputs.pop("labels"); out = model(**inputs)
        loss = self._loss(out.logits, lbl)
        return (loss, out) if return_outputs else loss

# ─── fold 1회 학습 ───────────────────────────────────────
def train_once(cfg: dict, tr_json: str, va_json: str,
               out_dir: pathlib.Path, exclude_ids: Optional[Set]=None) -> float:
    set_seed()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    mdl = (AutoModelForSequenceClassification
           .from_pretrained(cfg["model_name"], num_labels=2)
           .to(DEVICE))

    # ── 전체 pair 데이터 한 번만 읽는다
    raw_all = load_dataset("json", data_files=tr_json)["train"]

    # ── 검증 ID 목록 로드
    val_ids = set(pd.read_json(va_json, lines=True)["doc_id"])

    # ── 학습 · 검증 분할
    ds_tr = (raw_all
             .filter(lambda ex: ex["doc_id"] not in val_ids)        # train
             .shuffle(seed=SEED)
             .map(enc_factory(tok, cfg["max_len"]), batched=True,
                  remove_columns=["text_a", "text_b", "doc_id"]))

    ds_va = (raw_all
             .filter(lambda ex: ex["doc_id"] in val_ids)            # valid
             .map(enc_factory(tok, cfg["max_len"]), batched=True,
                  remove_columns=["text_a", "text_b", "doc_id"]))

    # ── Trainer 세팅·학습(아래는 기존 코드 그대로) ─────────────────
    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=SEED,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        fp16=bool(WORLD_GPUS),
        load_best_model_at_end=True, metric_for_best_model="f1",
        ddp_find_unused_parameters=False,
        report_to="none",
        **{KEY_EVAL: "epoch", KEY_SAVE: "epoch", KEY_LOG: "epoch"}
    )

    trainer = CETrainer(model=mdl, args=args, tokenizer=tok,
                        data_collator=DataCollatorWithPadding(tok),
                        train_dataset=ds_tr, eval_dataset=ds_va,
                        compute_metrics=metric_fn)
    trainer.train()

    (out_dir / "best").mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir / "best"); tok.save_pretrained(out_dir / "best")
    f1 = trainer.evaluate()["eval_f1"]

    del trainer, mdl, tok, ds_tr, ds_va, raw_all
    if WORLD_GPUS: torch.cuda.empty_cache()
    gc.collect()
    return f1


# ─── CLI ────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--fold_dir",    required=True,
                    help="pair_fold_*.jsonl 들이 들어 있는 폴더")
    ap.add_argument("--cfg",         required=True,
                    help="YAML 또는 JSON 형식 설정 파일")
    ap.add_argument("--ckpt_root",   required=True)
    ap.add_argument("--exp_name",    required=True,
                    help="예: pair_roberta_mid_cv")
    args = ap.parse_args()

    # cfg 로드 (yaml / json 자동)
    cfg_path = pathlib.Path(args.cfg)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"cfg 파일을 찾을 수 없습니다: {cfg_path}")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    else:
        cfg = json.load(open(cfg_path, encoding="utf-8"))

    # CV 폴드 파일 찾기
    folds = sorted(pathlib.Path(args.fold_dir).glob("pair_fold_*.jsonl"))
    if not folds:
        raise FileNotFoundError("pair_fold_*.jsonl not found in fold_dir")

    ck_root = pathlib.Path(args.ckpt_root)
    f1s=[]
    for k, fp in enumerate(folds):
        ids = set(pd.read_json(fp, lines=True)["doc_id"])
        f1 = train_once(cfg,
                        args.train_jsonl, str(fp),
                        ck_root/f"{args.exp_name}_fold{k}",
                        exclude_ids=ids)
        print(f"Fold{k} F1 = {f1:.4f}")
        f1s.append(f1)
    print(f"▶ CV mean F1 = {np.mean(f1s):.4f}")
