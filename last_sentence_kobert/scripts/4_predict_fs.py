#!/usr/bin/env python
"""
4_predict_fs.py
---------------
✓ `test.csv` 4 문장 각각에 대해 ‘첫 문장일 확률’ 예측
✓ ckpt_dir 안에
     • best/               ← 1-model
     • *_fold?/best/ …     ← 5-fold
  둘 다 자동 인식 → logits 평균 앙상블
✓ 결과를 ID 단위 jsonl -OR- csv 두 형태로 저장
"""

import argparse, json, math, pathlib, sys
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


# -------------------------------------------------------------------
def find_best_dirs(ckpt_root: pathlib.Path) -> List[pathlib.Path]:
    """fold0/best … fold4/best or single best/ 를 모두 반환"""
    if (ckpt_root / "best").is_dir():          # 단일 모델
        return [ckpt_root / "best"]

    bests = sorted((p / "best") for p in ckpt_root.glob("*_fold*") if (p / "best").is_dir())
    if not bests:
        sys.exit(f"[ERR] no checkpoint found under {ckpt_root}")
    return bests


def build_dataset(df: pd.DataFrame, tokenizer, max_len: int) -> Dataset:
    rows = []
    for _, r in df.iterrows():
        for i in range(4):
            rows.append({"ID": r.ID, "idx": i, "text": r[f"sentence_{i}"]})
    ds = Dataset.from_pandas(pd.DataFrame(rows))
    return ds.map(
        lambda ex: tokenizer(
            ex["text"],
            truncation=True,
            max_length=max_len,
            return_token_type_ids=False,
        ),
        batched=True,
        remove_columns=["text"],
    )


def model_logits(model, dl):
    outs = []
    for batch in dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits          # (bs, 2)
        outs.append(logits.cpu())
    return torch.cat(outs, 0)                   # (N, 2)


# -------------------------------------------------------------------
def main(a):
    ckpt_root = pathlib.Path(a.ckpt_dir)
    best_dirs = find_best_dirs(ckpt_root)

    # ── tokenizer / dataset --------------------------------------------------
    tok = AutoTokenizer.from_pretrained(best_dirs[0], use_fast=False)
    df_test = pd.read_csv(a.test_csv)
    ds_test = build_dataset(df_test, tok, a.max_len)
    dl_test = DataLoader(
        ds_test.remove_columns(["ID", "idx"]),
        batch_size=a.batch_size,
        collate_fn=DataCollatorWithPadding(tok),
    )

    # ── ensemble -------------------------------------------------------------
    logits_sum = torch.zeros(len(ds_test), 2)
    for bd in best_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(bd).to(DEVICE).eval()
        logits_sum += model_logits(model, dl_test)
        del model
        torch.cuda.empty_cache()

    probs = torch.softmax(logits_sum / len(best_dirs), dim=-1)[:, 1].numpy()  # P(first)

    # ── per-ID top-k ---------------------------------------------------------
    k = a.top_k
    offset = 0
    jsonl_lines, csv_rows = [], []
    for _, r in df_test.iterrows():
        p4 = probs[offset : offset + 4]
        offset += 4
        order = p4.argsort()[::-1][:k]
        jsonl_lines.append(
            json.dumps(
                {
                    "ID": r.ID,
                    "rank": order.tolist(),
                    "prob": p4[order].round(6).tolist(),
                },
                ensure_ascii=False,
            )
        )
        for i in range(4):
            csv_rows.append({"ID": r.ID, "idx": i, "prob_first": float(p4[i])})

    # ── save -----------------------------------------------------------------
    out_jsonl = pathlib.Path(a.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(jsonl_lines), encoding="utf-8")

    out_csv = pathlib.Path(a.out_csv)
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    print(f"✅ saved  jsonl:{out_jsonl}  csv:{out_csv}")



# ---------------------------------------------------------------------------
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--ckpt_dir", required=True,
                   help="ex) first_sentence_kobert/checkpoints/roberta-large")
    P.add_argument("--test_csv", required=True, help="data/raw/test.csv")
    P.add_argument("--out_jsonl", default="data/proc/test_first_topk.jsonl")
    P.add_argument("--out_csv",   default="data/proc/test_first_probs.csv")
    P.add_argument("--top_k", type=int, default=1)
    P.add_argument("--max_len", type=int, default=96)
    P.add_argument("--batch_size", type=int, default=128)
    main(P.parse_args())
