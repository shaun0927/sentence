#!/usr/bin/env python
"""
3_eval_fs.py
────────────
• *_fold*/best 체크포인트를 순회하며 각 fold-valid를 예측
• OOF 확률·라벨·메타데이터를 저장
      oof_prob.npy   : (N, 2) float32
      oof_label.npy  : (N,)   int8
      oof_meta.npy   : (N, 2) int32   ← [doc_id, sent_idx]
"""

import argparse, json, pathlib, sys, numpy as np, torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ────────────────── metric util ────────────────────
def metric_dict(y_true, y_prob):
    y_pred = (y_prob[:, 1] > 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": p, "recall": r, "f1": f1}

# ───────────────────── main ─────────────────────────
def main(a):
    ck_root = pathlib.Path(a.ckpt_dir)
    ck_dirs = sorted(ck_root.glob("*_fold*/best"))
    if not ck_dirs:
        sys.exit(f"[ERR] no *_fold*/best found under {ck_root}")
    print("▶ detected", len(ck_dirs), "fold checkpoints")

    fold_dir = pathlib.Path(a.fold_dir)
    out_dir  = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_prob, all_lbl, all_meta, per_fold = [], [], [], []

    # ─────────── fold loop ───────────
    for ckpt in ck_dirs:
        fidx = int(ckpt.parent.name.split("_fold")[-1])
        vfile = fold_dir / f"fold_{fidx}.jsonl"
        if not vfile.exists():
            sys.exit(f"[ERR] {vfile} missing")

        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEVICE).eval()
        trainer = Trainer(model=model, tokenizer=tok)

        # ── valid 데이터 ───────────────────────────
        vds = load_dataset("json", data_files=str(vfile))["train"]
        def _enc(ex): return tok(ex["text"], truncation=True,
                                 max_length=128, return_tensors=None)
        vds = vds.map(_enc, batched=True)

        # ── inference ─────────────────────────────
        prob_chunks = []
        idx_splits = np.array_split(np.arange(len(vds)),
                                    max(1, len(vds)//1024))
        for idx in tqdm(idx_splits, desc=f"fold{fidx}", leave=False):
            prob_chunks.append(trainer.predict(vds.select(idx)).predictions)
        prob = np.concatenate(prob_chunks, 0)        # (n,2) logits
        lbl  = np.asarray(vds["label"],   dtype=np.int8)
        did  = np.asarray(vds["doc_id"],  dtype=np.int32)
        sid  = np.asarray(vds.get("sent_idx",
                   np.tile([0,1,2,3], len(did)//4)), dtype=np.int32)  # fallback

        # ── 기록 ──────────────────────────────────
        all_prob.append(prob); all_lbl.append(lbl)
        all_meta.append(np.vstack([did, sid]).T)

        sc = metric_dict(lbl, prob); sc["fold"] = fidx
        per_fold.append(sc); print(f"fold {fidx} F1={sc['f1']:.4f}")

    # ─────────── aggregate & save ───────────
    oof_prob  = np.concatenate(all_prob, 0).astype(np.float32)
    oof_label = np.concatenate(all_lbl,  0).astype(np.int8)
    oof_meta  = np.concatenate(all_meta, 0).astype(np.int32)

    np.save(out_dir/"oof_prob.npy",  oof_prob)
    np.save(out_dir/"oof_label.npy", oof_label)
    np.save(out_dir/"oof_meta.npy",  oof_meta)

    overall = metric_dict(oof_label, oof_prob)
    print("\n📊 OOF summary:",
          ", ".join(f"{k}={v:.4f}" for k,v in overall.items()))

    with open(out_dir/"oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"fold_scores": per_fold, "overall": overall},
                  f, indent=2, ensure_ascii=False)
    print("✅ saved OOF files to", out_dir)

# ───────────────────── CLI ───────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="parent dir containing *_fold*/best/")
    p.add_argument("--fold_dir", required=True,
                   help="fold_0.jsonl … fold_k.jsonl location")
    p.add_argument("--out_dir", default="data/proc",
                   help="directory to write OOF files")
    main(p.parse_args())
