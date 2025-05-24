#!/usr/bin/env python
"""
build_submission.py  (all-in-one)
────────────────────────────────────────────────────────────
사용 예:
python build_submission.py \
  --test_csv            data/raw/test.csv \
  --first_last_jsonl    data/proc/first_last_final.jsonl \
  --pair_ckpt_dir       middle_sentence_kobert/checkpoints/pair_roberta_mid_cv \
  --out_csv             submission.csv
"""

import argparse, json, pathlib, gc
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────── Pair-BERT 앙상블 ─────────────────────
def pair_logits(ckpt_root: pathlib.Path, ds_tok: Dataset,
                tok, bs: int = 128) -> np.ndarray:
    """fold 앙상블 soft-logit 반환 (N,2)"""
    best_dirs = sorted(ckpt_root.glob("*_fold*/best"))
    if not best_dirs and (ckpt_root / "best").is_dir():       # single
        best_dirs = [ckpt_root / "best"]
    if not best_dirs:
        raise FileNotFoundError(f"'best' 디렉터리를 {ckpt_root}에서 찾지 못했습니다.")

    dl = torch.utils.data.DataLoader(
        ds_tok, batch_size=bs, collate_fn=DataCollatorWithPadding(tok)
    )

    total = torch.zeros(len(ds_tok), 2)
    for bd in best_dirs:
        mdl = (AutoModelForSequenceClassification
               .from_pretrained(bd).to(DEVICE).eval())
        outs = []
        for bt in dl:
            bt = {k: v.to(DEVICE) for k, v in bt.items()}
            outs.append(mdl(**bt).logits.cpu())
        total += torch.cat(outs, 0)
        del mdl; torch.cuda.empty_cache(); gc.collect()

    return (total / len(best_dirs)).numpy()


# ─────────────────────────── main ───────────────────────
def main(a):
    # 0) 데이터 로드 ──────────────────────────────────────
    df_test = pd.read_csv(a.test_csv, encoding="utf-8-sig")
    first_last = {j["ID"]: j for j in map(json.loads,
                                         open(a.first_last_jsonl, encoding="utf-8"))}

    # 1) 가운데 두 문장 쌍 만들기 ───────────────────────
    pair_rows, meta = [], []          # meta: (ID, idx_a, idx_b, cand)
    for _, row in df_test.iterrows():
        aid = row.ID
        first_i, last_i = first_last[aid]["first"], first_last[aid]["last"]
        mids = [i for i in range(4) if i not in (first_i, last_i)]
        ia, ib = mids  # 2개

        pair_rows.append({"text": row[f"sentence_{ia}"],
                          "text_pair": row[f"sentence_{ib}"]})
        meta.append((aid, ia, ib, 0))          # cand0 = (ia, ib)

        pair_rows.append({"text": row[f"sentence_{ib}"],
                          "text_pair": row[f"sentence_{ia}"]})
        meta.append((aid, ia, ib, 1))          # cand1 = (ib, ia)

    ds_pair = Dataset.from_pandas(pd.DataFrame(pair_rows))

    # 2) 토크나이즈 + Pair-BERT 추론 ─────────────────────
    ckpt_root = pathlib.Path(a.pair_ckpt_dir)
    # 토크나이저: best 디렉터리 탐색
    tok_dir = sorted(ckpt_root.glob("*_fold*/best"))
    tok_dir = tok_dir[0] if tok_dir else (ckpt_root / "best")
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=False)

    ds_tok = ds_pair.map(
        lambda b: tok(b["text"], b["text_pair"],
                      truncation=True, max_length=128,
                      return_token_type_ids=False),
        batched=True,
        remove_columns=["text", "text_pair"],
    )

    logits = pair_logits(ckpt_root, ds_tok, tok)          # (N,2)
    prob_first = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    # 3) 문서별 cand 0 vs 1 중 큰 쪽 선택 ───────────────
    best_mid = defaultdict(lambda: (-1, -1.0, -1, -1))   # ID → (cand, p, ia, ib)
    for (aid, ia, ib, cand), p in zip(meta, prob_first):
        if p > best_mid[aid][1]:
            best_mid[aid] = (cand, p, ia, ib)

    # 4) 최종 순서 & CSV 작성 ───────────────────────────
    rows_out = []
    for _, row in df_test.iterrows():
        aid = row.ID
        first_i, last_i = first_last[aid]["first"], first_last[aid]["last"]
        ia, ib = best_mid[aid][2], best_mid[aid][3]
        if best_mid[aid][0] == 1:      # cand1이면 (ib, ia) 순
            ia, ib = ib, ia
        order = [first_i, ia, ib, last_i]
        rows_out.append({"ID": aid,
                         "answer_0": order[0],
                         "answer_1": order[1],
                         "answer_2": order[2],
                         "answer_3": order[3]})

    out_path = pathlib.Path(a.out_csv)
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print("✅ submission 저장:", out_path)


# ────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv",         required=True)
    p.add_argument("--first_last_jsonl", required=True,
                   help="first_last_final.jsonl 경로")
    p.add_argument("--pair_ckpt_dir",    required=True,
                   help="가운데 두 문장 Pair-BERT 체크포인트 root")
    p.add_argument("--out_csv",          default="submission.csv")
    main(p.parse_args())
