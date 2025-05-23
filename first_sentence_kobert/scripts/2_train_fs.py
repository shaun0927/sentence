#!/usr/bin/env python
"""
2_train_fs.py
-------------
KoBERT 첫 문장(binary) 분류기 파인튜닝
  • 단일 학습        : 기본 (--train / --valid 지정)
  • 5-Fold CV 학습   : --cv --fold_dir data/fold  (fold_0.jsonl … fold_4.jsonl)
"""

import argparse, yaml, pathlib, random, os, numpy as np, inspect, json, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, IntervalStrategy
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ───────────────────────────── 기본 설정 ─────────────────────────────
SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────── metric 함수 ────────────────────────────────
def compute_metrics(pred):
    y_true, y_pred = pred.label_ids, pred.predictions.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": p, "recall": r, "f1": f1}


# ─────────────────────── strategy 키 자동 감지 ───────────────────────
def pick_param(long: str, short: str) -> str:
    params = inspect.signature(TrainingArguments.__init__).parameters
    return long if long in params else short


STRAT_EVAL = pick_param("evaluation_strategy", "eval_strategy")
STRAT_SAVE = pick_param("save_strategy",        "save_steps")
STRAT_LOG  = pick_param("logging_strategy",     "logging_steps")


# ────────────────────────── Trainer 서브클래스 ───────────────────────
class WeightedCETrainer(Trainer):
    def __init__(self, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self._loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ────────────────────────── 학습 함수 ────────────────────────────────
def train_once(cfg: dict, train_path: str, valid_path: str, exp_path: pathlib.Path):
    set_seed()

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2
    ).to(device)

    def encode(ex):
        return tok(ex["text"], truncation=True,
                   max_length=cfg.get("max_len", 96),
                   return_token_type_ids=False)

    ds_train = load_dataset("json", data_files=train_path)["train"] \
               .shuffle(seed=SEED).map(encode, batched=True)
    ds_valid = load_dataset("json", data_files=valid_path)["train"] \
               .map(encode, batched=True)

    exp_path.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(exp_path),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        **{
            STRAT_EVAL: IntervalStrategy.EPOCH,
            STRAT_SAVE: IntervalStrategy.EPOCH,
            STRAT_LOG : IntervalStrategy.EPOCH,
        }
    )

    class_w = torch.tensor([1.0, cfg.get("pos_weight", 3.0)], device=device)
    trainer = WeightedCETrainer(
        class_weight=class_w,
        model=model,
        args=train_args,
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


# ────────────────────────── main ────────────────────────────────────
def main(args):
    cfg = yaml.safe_load(open(args.cfg, encoding="utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs

    checkpoints_root = pathlib.Path("first_sentence_kobert/checkpoints")

    # ── 1) 단일 학습 ────────────────────────────────────────────────
    if not args.cv:
        exp_path = checkpoints_root / args.exp
        metrics, best_dir = train_once(cfg, args.train, args.valid, exp_path)
        print("✅ Single run finished:", json.dumps(metrics, indent=2, ensure_ascii=False))
        print("Best model saved ->", best_dir)
        return

    # ── 2) 5-Fold CV ───────────────────────────────────────────────
    fold_dir = pathlib.Path(args.fold_dir)
    results = []

    for k in range(5):
        valid_file = fold_dir / f"fold_{k}.jsonl"
        if not valid_file.exists():
            raise FileNotFoundError(valid_file)
        exp_path = checkpoints_root / f"{args.exp}_fold{k}"
        print(f"\n─── Fold {k} training ───")
        m, _ = train_once(cfg, args.train, str(valid_file), exp_path)
        m["fold"] = k
        results.append(m)
        print(f"Fold {k} metrics:", m)

    # 평균/표준편차 요약
    f1s = [m["eval_f1"] for m in results]
    print("\n📊 5-Fold CV summary:  avg F1 = %.4f  ±  %.4f" %
          (np.mean(f1s), np.std(f1s)))
    # 저장
    with open(checkpoints_root / f"{args.exp}_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Metrics logged to", f.name)


# ────────────────────────── CLI ─────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="fs_train.jsonl")
    p.add_argument("--valid", required=False, help="single-run valid jsonl")
    p.add_argument("--cfg",   required=True)
    p.add_argument("--exp",   required=True)
    p.add_argument("--epochs", type=int, help="override epochs")
    # CV 옵션
    p.add_argument("--cv", action="store_true", help="enable 5-fold training")
    p.add_argument("--fold_dir", default="data/fold",
                   help="directory containing fold_0.jsonl … fold_4.jsonl")
    args = p.parse_args()

    if args.cv:
        # valid 인자 필요 없음
        assert pathlib.Path(args.fold_dir).is_dir(), "--fold_dir 경로가 존재하지 않습니다."
    else:
        if not args.valid:
            raise ValueError("single-run 모드에서는 --valid 경로를 지정해야 합니다.")
    main(args)
