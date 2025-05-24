#!/usr/bin/env python
"""
2_train_fs.py ─ KoBERT 첫 문장(binary) 분류기
  • 단일 학습 (--train / --valid)
  • K-Fold CV (--cv --fold_dir data/fold)
"""
import argparse, yaml, pathlib, random, os, json, inspect
from typing import Optional, Set, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, IntervalStrategy
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict

# ─────────────────────── 글로벌 설정 ───────────────────────────────
SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────── metrics 함수 ──────────────────────────────
def compute_metrics(pred, k: int = 2):
    """
    Top-1 F1  +  Hit@2 / MRR@2  (문서 ID 기반)
    pred.inputs 가 dict 가 아닐 경우를 대비해 fallback(4줄 연속) 로직 포함
    """
    y_true = pred.label_ids
    logits = pred.predictions
    probs  = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    # ---- doc_id 추출 (HF ≥4.30 은 dict, 그 이하/TPU 등에선 list 가능) ----
    doc_ids = None
    if isinstance(pred.inputs, dict) and "doc_id" in pred.inputs:
        doc_ids = pred.inputs["doc_id"]
    elif hasattr(pred, "inputs") and isinstance(pred.inputs, (list, tuple)):
        # older Trainer 버전: inputs = (input_ids, attention, labels, doc_id,…)
        for item in pred.inputs:
            if isinstance(item, torch.Tensor) and item.ndim == 1 and len(item) == len(y_true):
                # heuristically pick the 1-D tensor that matches length
                if item.dtype in (torch.int32, torch.int64):
                    doc_ids = item
                    break

    # ---- Hit@2 / MRR@2 -----------------------------------------------------
    if doc_ids is not None:
        bucket = defaultdict(list)
        for p, y, d in zip(probs, y_true, doc_ids):
            bucket[int(d)].append((p, y))

        hit = mrr = 0
        for arr in bucket.values():
            arr.sort(key=lambda x: -x[0])
            topk = arr[:k]
            rank = next((i for i,(_,lab) in enumerate(topk) if lab == 1), None)
            if rank is not None:
                hit += 1
                mrr += 1.0 / (rank + 1)
        n_doc = len(bucket)
    else:
        # fallback: valid set이 4줄 연속 구조라는 가정
        hit = mrr = 0
        for i in range(0, len(y_true), 4):
            idx = slice(i, i + 4)
            topk = probs[idx].argsort()[::-1][:k]
            labels4 = y_true[idx]
            rank = np.where(labels4[topk] == 1)[0]
            if rank.size:
                hit += 1
                mrr += 1.0 / (rank[0] + 1)
        n_doc = len(y_true) // 4

    # ---- Top-1 F1 ----------------------------------------------------------
    y_pred = (probs > 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": p, "recall": r, "f1": f1,
        "hit@2":     hit / n_doc,
        "mrr@2":     mrr / n_doc,
    }

# ─────────────────────── Trainer 확장 ──────────────────────────────
class WeightedCETrainer(Trainer):
    def __init__(self, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight)
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self._loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ─────────────────────── strategy 키 자동 감지 ─────────────────────
def pick_param(long, short):
    return long if long in inspect.signature(TrainingArguments.__init__).parameters else short
STRAT_EVAL = pick_param("evaluation_strategy", "eval_strategy")
STRAT_SAVE = pick_param("save_strategy",        "save_steps")
STRAT_LOG  = pick_param("logging_strategy",     "logging_steps")

# ─────────────────────── 학습 1회 함수 ─────────────────────────────
def train_once(cfg: dict,
               train_path: str,
               valid_path: str,
               exp_path: pathlib.Path,
               exclude_texts: Optional[Set[str]] = None):

    set_seed()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2).to(device)

    def encode(ex):
        enc = tok(ex["text"],
                  truncation=True,
                  max_length=cfg.get("max_len", 96),
                  return_token_type_ids=False)
        enc["labels"] = ex["label"]
        enc["doc_id"] = torch.tensor(ex["doc_id"], dtype=torch.int64)  # tensor화
        return enc

    raw_train = load_dataset("json", data_files=train_path)["train"]
    if exclude_texts:
        raw_train = raw_train.filter(lambda ex: ex["text"] not in exclude_texts)

    ds_train = raw_train.shuffle(seed=SEED).map(encode, batched=True)
    ds_valid = load_dataset("json", data_files=valid_path)["train"].map(encode, batched=True)

    exp_path.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(exp_path),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        include_inputs_for_metrics=True,           # ★ doc_id 전달 필수
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        **{STRAT_EVAL: IntervalStrategy.EPOCH,
           STRAT_SAVE:  IntervalStrategy.EPOCH,
           STRAT_LOG:   IntervalStrategy.EPOCH}
    )

    class_w = torch.tensor([1.0, cfg.get("pos_weight", 3.0)], device=device)
    trainer = WeightedCETrainer(
        class_weight=class_w,
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    result = trainer.train()
    best_dir = exp_path / "best"
    trainer.save_model(str(best_dir)); tok.save_pretrained(str(best_dir))

    metrics = trainer.evaluate()
    metrics.update({"train_runtime": result.metrics["train_runtime"]})
    return metrics, best_dir

# ─────────────────────── main ──────────────────────────────────────
def main(args):
    cfg = yaml.safe_load(open(args.cfg, encoding="utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs

    ckpt_root = pathlib.Path("first_sentence_kobert/checkpoints")

    # ── 단일 run ───────────────────────────────────────────────────
    if not args.cv:
        m, best = train_once(cfg, args.train, args.valid,
                             ckpt_root / args.exp)
        print("✅ Single run finished:", json.dumps(m, indent=2, ensure_ascii=False))
        print("Best model →", best)
        return

    # ── K-Fold CV ─────────────────────────────────────────────────
    fold_files = sorted(pathlib.Path(args.fold_dir).glob("fold_*.jsonl"))
    assert fold_files, f"{args.fold_dir} 에 fold_*.jsonl 이 없습니다."

    results = []
    for k, vf in enumerate(fold_files):
        valid_ds = load_dataset("json", data_files=str(vf))["train"]
        exclude_set = set(valid_ds["text"])

        print(f"\n── Fold {k} ──")
        m, _ = train_once(cfg, args.train, str(vf),
                          ckpt_root / f"{args.exp}_fold{k}",
                          exclude_texts=exclude_set)
        m["fold"] = k
        results.append(m)
        print("metrics:", m)

    f1s = [m["f1"] for m in results]
    print(f"\n📊 {len(results)}-Fold avg F1 = {np.mean(f1s):.4f}")
    with open(ckpt_root / f"{args.exp}_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ─────────────────────── CLI ───────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="fs_train.jsonl")
    p.add_argument("--valid", help="single-run valid jsonl")
    p.add_argument("--cfg",   required=True)
    p.add_argument("--exp",   required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--cv", action="store_true")
    p.add_argument("--fold_dir", default="data/fold")
    args = p.parse_args()

    if args.cv:
        assert pathlib.Path(args.fold_dir).is_dir(), "--fold_dir 경로가 없습니다."
    else:
        if not args.valid:
            raise ValueError("single-run 모드에서는 --valid 를 지정해야 합니다.")
    main(args)
