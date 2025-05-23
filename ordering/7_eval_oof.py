#!/usr/bin/env python
"""
7_eval_oof.py
─────────────
OOF jsonl + train.csv → f1 / accuracy 로컬 평가
"""

import argparse, json, pathlib, pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def main(a):
    # ① 정답
    train = pd.read_csv(a.train_csv).set_index("ID")
    y_true = []
    y_pred = []
    # ② 예측
    for l in pathlib.Path(a.oof_jsonl).read_text().splitlines():
        obj = json.loads(l)
        y_true.append(train.loc[obj["ID"]][["answer_0","answer_1","answer_2","answer_3"]].tolist())
        y_pred.append(obj["best_order"])
    # ③ metric (완전 일치 여부로 accuracy / f1)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"🪄  OOF accuracy={acc:.4f}   macro-f1={f1:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="data/raw/train.csv")
    p.add_argument("--oof_jsonl", required=True,
                   help="data/proc/oof_ppl_scores.jsonl")
    main(p.parse_args())
