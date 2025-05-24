#!/usr/bin/env python
"""
1_make_dataset_last.py  ─  마지막 문장용 데이터셋 & K-Fold 생성
────────────────────────────────────────────────────────────
train.csv  (행 = 4문장 + 정답 순서(answer_0~3))   →
 • ls_train.jsonl          : 마지막 문장(binary) 전체 학습 세트
 • pw_train.jsonl          : 문장쌍 순서(binary)  ← first 버전 재사용
 • fold_last/fold_i.jsonl  : 문서 단위 Stratified K-Fold (4줄 연속)
jsonl 예) {"text":"문장", "label":1, "doc_id":17}
"""
import argparse, json, pathlib, random
from typing import List
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except ImportError:
    HAS_SGK = False

SEED = 42
random.seed(SEED)

# ───────────── util ──────────────
def _write_jsonl(rows: List[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ───────────── 마지막 문장 세트 ────
def build_last_samples(df: pd.DataFrame, out_path: pathlib.Path):
    rows = []
    for doc_id, row in df.iterrows():
        gold_last = int(row["answer_3"])          # ★ 마지막 문장 인덱스
        for i in range(4):
            rows.append({
                "text":   row[f"sentence_{i}"],
                "label":  int(i == gold_last),    # ★ label = 1 if last
                "doc_id": int(doc_id)
            })
    random.shuffle(rows)
    _write_jsonl(rows, out_path)

# (문장쌍 세트는 first 버전 그대로 사용)
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

# ───────────── K-Fold ─────────────
def make_folds_doc(csv_path: pathlib.Path, k: int):
    df = pd.read_csv(csv_path)
    X      = df.index.values
    y      = df["answer_3"].values                 # ★ stratify by last-idx
    groups = df["ID"].values if "ID" in df.columns else df.index.values

    splitter = (StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
                if HAS_SGK else
                StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED))
    splits = splitter.split(X, y, groups) if HAS_SGK else splitter.split(X, y)

    fold_dir = csv_path.parent / "fold_last"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, val_idx) in enumerate(splits):
        rows = []
        for doc_id, row in df.iloc[sorted(val_idx)].iterrows():
            gold_last = int(row["answer_3"])
            for j in range(4):
                rows.append({
                    "text":   row[f"sentence_{j}"],
                    "label":  int(j == gold_last),
                    "doc_id": int(doc_id)
                })
        fp = fold_dir / f"fold_{i}.jsonl"
        _write_jsonl(rows, fp)
        print(f"[fold] saved {fp.name:<12} ({len(rows)} lines)")

# ───────────── CLI ────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="data/proc_last")
    ap.add_argument("--make_fold", action="store_true")
    ap.add_argument("--k", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = pathlib.Path(args.out_dir)

    build_last_samples(df, out / "ls_train.jsonl")
    build_pair_samples(df, out / "pw_train.jsonl")     # optional 재사용
    print("✅ JSONL 생성 완료")

    if args.make_fold:
        make_folds_doc(pathlib.Path(args.csv), k=args.k)
