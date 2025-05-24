#!/usr/bin/env python
"""
2_train_fs.py
-------------
KoBERT 첫 문장(binary) 분류기 파인튜닝
  • 단일 학습 (--train / --valid)
  • K-Fold CV  (--cv  --fold_dir data/fold)  ← fold_0.jsonl … fold_N.jsonl
     ↳ 검증 문서가 학습 세트에 섞이지 않도록 누수 차단
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

# ─────────────────────── 기본 설정 ─────────────────────────────────
SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────── metric 함수 ───────────────────────────────
def compute_metrics(pred, k=2):
    y_true = pred.label_ids
    logits = pred.predictions          # [N, 2]

    # ── Top-1 F1 기존 계산 ─────────────────────────
    y_pred = logits.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # ── Hit@2 계산 ────────────────────────────────
    # 확률이 양성일 top-2 문장 기준이 아니라,
    # "첫 문장일 확률" 값 상위 k 에 정답(라벨=1)이 있나?
    probs_first = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    # 그룹(문서)별로 top-k 보기 → 여기선 배치가 문서 섞여 있으니
    # valid 데이터셋이 문서 단위 4문장씩 순서대로라는 전제를 사용
    hit2, mrr2 = 0, 0.0
    for i in range(0, len(y_true), 4):
        idx = slice(i, i+4)
        topk = probs_first[idx].argsort()[::-1][:k]
        labels4 = y_true[idx]
        # 정답 위치
        rank = np.where(labels4[topk] == 1)[0]
        if rank.size:                 # 성공
            hit2 += 1
            mrr2 += 1.0 / (rank[0] + 1)   # rank[0]=0 → 1.0
    n_docs = len(y_true) // 4

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p, "recall": r, "f1": f1,
        "hit@2": hit2 / n_docs,
        "mrr@2": mrr2 / n_docs,
    }


# ───────────────── strategy 키 자동 감지 ────────────────────────────
def pick_param(long: str, short: str) -> str:
    params = inspect.signature(TrainingArguments.__init__).parameters
    return long if long in params else short


STRAT_EVAL = pick_param("evaluation_strategy", "eval_strategy")
STRAT_SAVE = pick_param("save_strategy",        "save_steps")
STRAT_LOG  = pick_param("logging_strategy",     "logging_steps")


# ─────────────────────── Trainer 확장 (가중치 CE) ──────────────────
class WeightedCETrainer(Trainer):
    def __init__(self, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self._loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────── 학습 1회 함수 ──────────────────────────────
def train_once(cfg: dict,
               train_path: str,
               valid_path: str,
               exp_path: pathlib.Path,
               exclude_texts: Optional[Set[str]] = None):
    """
    exclude_texts : 학습 세트에서 제거할 문장 set (누수 차단용)
    """
    set_seed()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2
    ).to(device)

    # ── 데이터 로드 & 전처리 ───────────────────────────────────────
    def encode(ex):
        return tok(ex["text"],
                   truncation=True,
                   max_length=cfg.get("max_len", 96),
                   return_token_type_ids=False)

    raw_train = load_dataset("json", data_files=train_path)["train"]
    if exclude_texts:
        raw_train = raw_train.filter(lambda ex: ex["text"] not in exclude_texts)

    ds_train = raw_train.shuffle(seed=SEED).map(encode, batched=True)
    ds_valid = load_dataset("json", data_files=valid_path)["train"] \
               .map(encode, batched=True)

    # ── 학습 준비 ─────────────────────────────────────────────────
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


# ───────────────────────────── main ────────────────────────────────
def main(args):
    cfg = yaml.safe_load(open(args.cfg, encoding="utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs

    ckpt_root = pathlib.Path("first_sentence_kobert/checkpoints")

    # ── 단일 학습 모드 ────────────────────────────────────────────
    if not args.cv:
        exp_path = ckpt_root / args.exp
        metrics, best_dir = train_once(cfg, args.train, args.valid, exp_path)
        print("✅ Single run finished:", json.dumps(metrics, indent=2, ensure_ascii=False))
        print("Best model saved ->", best_dir)
        return

    # ── K-Fold CV 모드 ───────────────────────────────────────────
    fold_dir = pathlib.Path(args.fold_dir)
    fold_files: List[pathlib.Path] = sorted(fold_dir.glob("fold_*.jsonl"))
    assert fold_files, f"{fold_dir} 에 fold_*.jsonl 이 없습니다."
    K = len(fold_files)
    print(f"▶ Detected {K} fold files")

    results = []
    for k, valid_file in enumerate(fold_files):
        # ── 누수 차단용 exclude set ────────────────────────────
        valid_ds = load_dataset("json", data_files=str(valid_file))["train"]
        exclude_set = set(valid_ds["text"])

        exp_path = ckpt_root / f"{args.exp}_fold{k}"
        print(f"\n─── Fold {k} training ───")
        m, _ = train_once(cfg,
                          train_path=args.train,
                          valid_path=str(valid_file),
                          exp_path=exp_path,
                          exclude_texts=exclude_set)
        m["fold"] = k
        results.append(m)
        print(f"Fold {k} metrics:", m)

    f1s = [m["eval_f1"] for m in results]
    print(f"\n📊 {K}-Fold CV summary:  avg F1 = {np.mean(f1s):.4f}  ±  {np.std(f1s):.4f}")

    with open(ckpt_root / f"{args.exp}_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Metrics logged to", f.name)


# ───────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="fs_train.jsonl (전체)")
    p.add_argument("--valid", help="single-run valid jsonl")
    p.add_argument("--cfg",   required=True, help="yaml config")
    p.add_argument("--exp",   required=True, help="experiment name")
    p.add_argument("--epochs", type=int, help="override epochs")

    # CV 옵션
    p.add_argument("--cv", action="store_true",
                   help="enable K-Fold cross-validation (fold_*.jsonl 자동 탐색)")
    p.add_argument("--fold_dir", default="data/fold",
                   help="directory containing fold_0.jsonl … fold_N.jsonl")
    args = p.parse_args()

    if args.cv:
        assert pathlib.Path(args.fold_dir).is_dir(), "--fold_dir 경로가 존재하지 않습니다."
    else:
        if not args.valid:
            raise ValueError("single-run 모드에서는 --valid 경로를 지정해야 합니다.")
    main(args)
