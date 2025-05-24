#!/usr/bin/env python
"""
1_make_dataset.py ─ 문서 단위 K-Fold 지원
────────────────────────────────────────────────────────────
train.csv (각 행 = 4문장 + 정답 순서) →
 • fs_train.jsonl     : 첫 문장(binary) 전체 학습
 • pw_train.jsonl     : 문장쌍 순서(binary) 전체 학습
 • fold/fold_i.jsonl  : 문서 단위 Stratified K-Fold
                        (첫 문장 index 비율 유지, 같은 문서 4줄 연속 기록)
각 jsonl 예)
{"text":"문장", "label":1, "doc_id":17}
"""
import argparse, json, pathlib, random
from typing import List

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# sklearn ≥1.1 -> StratifiedGroupKFold 우선 사용
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except ImportError:
    HAS_SGK = False

SEED = 42
random.seed(SEED)

# ───────────────── jsonl util ───────────────────────────────────────
def _write_jsonl(rows: List[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ───────────────── 전체 세트 ────────────────────────────────────────
def build_first_samples(df: pd.DataFrame, out_path: pathlib.Path):
    rows = []
    for doc_id, row in df.iterrows():                 # doc_id = 행 인덱스
        gold_first = int(row["answer_0"])
        for i in range(4):
            rows.append({
                "text":   row[f"sentence_{i}"],
                "label":  int(i == gold_first),
                "doc_id": int(doc_id)
            })
    random.shuffle(rows)
    _write_jsonl(rows, out_path)

def build_pair_samples(df: pd.DataFrame, out_path: pathlib.Path):
    rows = []
    for _, row in df.iterrows():
        gold_order = [int(row[f"answer_{k}"]) for k in range(4)]
        rank = {idx: pos for pos, idx in enumerate(gold_order)}
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                label = int(rank[i] < rank[j])
                text  = f"{row[f'sentence_{i}']} [SEP] {row[f'sentence_{j}']}"
                rows.append({"text": text, "label": label})
    random.shuffle(rows)
    _write_jsonl(rows, out_path)

# ───────────────── K-Fold ───────────────────────────────────────────
def make_folds_doc(csv_path: pathlib.Path, k: int):
    df = pd.read_csv(csv_path)

    X      = df.index.values
    y      = df["answer_0"].values
    groups = df["ID"].values if "ID" in df.columns else df.index.values

    if HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                        random_state=SEED)
        splits = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=k, shuffle=True,
                                   random_state=SEED)
        splits = splitter.split(X, y)

    fold_dir = csv_path.parent / "fold"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for f_idx, (_, val_idx) in enumerate(splits):
        for_rows = df.iloc[sorted(val_idx)]           # 4줄 연속 보장
        rows = []
        for doc_id, row in for_rows.iterrows():
            gold_first = int(row["answer_0"])
            for i in range(4):
                rows.append({
                    "text":   row[f"sentence_{i}"],
                    "label":  int(i == gold_first),
                    "doc_id": int(doc_id)
                })
        fp = fold_dir / f"fold_{f_idx}.jsonl"
        _write_jsonl(rows, fp)
        print(f"[fold] saved {fp.name:<12} ({len(rows)} lines)")

# ───────────────── CLI ──────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train.csv 경로")
    ap.add_argument("--out_dir", default="data/proc",
                    help="jsonl 출력 디렉터리")
    ap.add_argument("--make_fold", action="store_true",
                    help="문서 단위 K-Fold jsonl 생성")
    ap.add_argument("--k", type=int, default=7, help="fold 개수")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = pathlib.Path(args.out_dir)

    build_first_samples(df, out_dir / "fs_train.jsonl")
    build_pair_samples(df,  out_dir / "pw_train.jsonl")
    print("✅ JSONL 생성 완료")

    if args.make_fold:
        make_folds_doc(pathlib.Path(args.csv), k=args.k)
