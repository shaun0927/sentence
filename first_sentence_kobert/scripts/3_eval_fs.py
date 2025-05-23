#!/usr/bin/env python
"""
3_eval_fs.py
------------
*폴드별* checkpoint 를 모두 순회하여
 - 각 fold-valid 에 대해 예측
 - OOF 메트릭(F1·정확도 등) 계산
 - oof_proba / oof_pred csv 저장
"""

import argparse, json, pathlib, torch, numpy as np, yaml
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
torch.manual_seed(SEED)


# ──────────────────── metric util ──────────────────────
def metrics(y_true, y_prob):
    y_pred = (y_prob[:, 1] > 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p, "recall": r, "f1": f1,
    }


# ──────────────────── main ─────────────────────────────
def main(args):
    ckpt_root = pathlib.Path(args.ckpt_dir)
    fold_ckpts = sorted(ckpt_root.glob("*_fold*/best"))
    assert fold_ckpts, f"no fold checkpoints found under {ckpt_root}"
    print("▶ detected", len(fold_ckpts), "fold checkpoints")

    fold_dir  = pathlib.Path(args.fold_dir)
    out_dir   = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_oof_prob, all_oof_label = [], []
    per_fold_scores            = []

    for ckpt in fold_ckpts:
        fold_idx = int(ckpt.parent.name.split("_fold")[-1])
        valid_file = fold_dir / f"fold_{fold_idx}.jsonl"
        assert valid_file.exists(), f"missing {valid_file}"

        tok   = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
                    ckpt).to(DEVICE).eval()
        trainer = Trainer(model=model, tokenizer=tok)

        # ── valid dataset ───────────────────────────────
        valid_ds = load_dataset("json", data_files=str(valid_file))["train"]
        def _enc(ex): return tok(ex["text"], truncation=True,
                                  max_length=128, return_tensors=None)
        valid_ds = valid_ds.map(_enc, batched=True)

        # ── inference ───────────────────────────────────
        preds = []
        for chunk in tqdm(np.array_split(range(len(valid_ds)), 
                                         max(1, len(valid_ds)//1024)),
                          desc=f"fold{fold_idx}", leave=False):
            outputs = trainer.predict(valid_ds.select(chunk))
            preds.append(outputs.predictions)   # [N,2]
        prob = np.concatenate(preds, 0)
        label = np.array(valid_ds["label"])

        all_oof_prob.append(prob)
        all_oof_label.append(label)

        sc = metrics(label, prob)
        sc["fold"] = fold_idx
        per_fold_scores.append(sc)
        print(f"fold {fold_idx} F1={sc['f1']:.4f}")

    # ── aggregate OOF ───────────────────────────────────
    oof_prob   = np.concatenate(all_oof_prob,   0)
    oof_label  = np.concatenate(all_oof_label,  0)
    overall_sc = metrics(oof_label, oof_prob)
    print("\n📊 OOF summary:",
          ", ".join(f"{k}={v:.4f}" for k, v in overall_sc.items()))

    # ── save ────────────────────────────────────────────
    np.save(out_dir / "oof_prob.npy",  oof_prob)
    np.save(out_dir / "oof_label.npy", oof_label)
    with open(out_dir / "oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"fold_scores": per_fold_scores,
                   "overall": overall_sc}, f, ensure_ascii=False, indent=2)
    print("✅ saved OOF outputs to", out_dir)


# ──────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="parent dir that contains *_fold*/best checkpoints")
    ap.add_argument("--fold_dir", required=True,
                    help="data/fold/  (fold_0.jsonl …)")
    ap.add_argument("--out_dir",  default="data/proc",
                    help="directory to write oof_* files")
    main(ap.parse_args())
