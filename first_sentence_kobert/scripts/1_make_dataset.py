#!/usr/bin/env python
"""
1_make_dataset.py  (문서 단위  K-Fold 지원 버전)
─────────────────────────────────────────────────────────────
train.csv  (각 행 = 4문장 + 정답 순서)  →  
 • fs_train.jsonl  : 첫 문장(binary) 전체 학습 세트
 • pw_train.jsonl  : 문장쌍 순서(binary) 전체 학습 세트
 • fold/fold_0.jsonl … fold_{k-1}.jsonl :  문서 단위 Stratified K-Fold  
                                           (첫 문장 인덱스 비율 유지)

jsonl 1줄 예시  
{"text": "<문장>", "label": 1}
{"text": "<문장A> [SEP] <문장B>", "label": 0}
"""
import argparse, json, pathlib, random
from typing import List

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# sklearn ≥1.1 인 경우 StratifiedGroupKFold 사용
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except ImportError:                     # fallback
    HAS_SGK = False

SEED = 42
random.seed(SEED)


# ───────────────────────────── jsonl 작성 유틸 ──────────────────────────────
def _write_jsonl(rows: List[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ───────────────────────────── 전체 세트 생성 ──────────────────────────────
def build_first_samples(df: pd.DataFrame, out_path: pathlib.Path):
    """첫 문장 여부 binary 라벨 샘플 생성 (전체)"""
    rows = []
    for _, row in df.iterrows():
        gold_first = int(row["answer_0"])          # 0~3
        for i in range(4):
            rows.append({"text": row[f"sentence_{i}"],
                         "label": int(i == gold_first)})
    random.shuffle(rows)
    _write_jsonl(rows, out_path)


def build_pair_samples(df: pd.DataFrame, out_path: pathlib.Path):
    """모든 (A,B) 순서 쌍 binary 라벨 샘플 생성 (전체)"""
    rows = []
    for _, row in df.iterrows():
        gold_order = [int(row[f"answer_{k}"]) for k in range(4)]
        rank = {idx: pos for pos, idx in enumerate(gold_order)}   # 인덱스 → 순서
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                label = int(rank[i] < rank[j])                    # i 가 앞?
                text  = f"{row[f'sentence_{i}']} [SEP] {row[f'sentence_{j}']}"
                rows.append({"text": text, "label": label})
    random.shuffle(rows)
    _write_jsonl(rows, out_path)


# ───────────────────────────── K-Fold (문서 단위) ──────────────────────────
def make_folds_doc(csv_path: pathlib.Path, k: int = 7):
    """
    문서 단위 Stratified K-Fold → fold_{i}.jsonl 작성  
    • label  : answer_0  (첫 문장 인덱스)  
    • group  : ID 열이 있으면 ID, 없으면 행 인덱스
    """
    df = pd.read_csv(csv_path)
    X      = df.index.values                      # dummy
    y      = df["answer_0"].values                # stratify target
    groups = df["ID"].values if "ID" in df.columns else df.index.values

    if HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=k,
                                        shuffle=True,
                                        random_state=SEED)
        splits = splitter.split(X, y, groups)
    else:
        # fallback : 그룹 고려 없이 StratifiedKFold,
        #           단, 같은 문서가 다른 fold 로 흩어지지 않도록
        #           shuffle 후 순서대로 균등 분할
        splitter = StratifiedKFold(n_splits=k,
                                   shuffle=True,
                                   random_state=SEED)
        splits = splitter.split(X, y)

    fold_dir = csv_path.parent / "fold"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for f_idx, (_, val_idx) in enumerate(splits):
        rows = []
        for _, row in df.iloc[val_idx].iterrows():
            gold_first = int(row["answer_0"])
            for i in range(4):
                rows.append({"text": row[f"sentence_{i}"],
                             "label": int(i == gold_first)})
        fold_path = fold_dir / f"fold_{f_idx}.jsonl"
        _write_jsonl(rows, fold_path)
        print(f"[fold] saved {fold_path.name:<10} ({len(rows)} lines)")


# ──────────────────────────────── CLI ────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train.csv 경로")
    ap.add_argument("--out_dir", default="data/proc", help="jsonl 출력 디렉터리")
    ap.add_argument("--make_fold", action="store_true",
                    help="문서 단위 Stratified K-Fold jsonl 생성")
    ap.add_argument("--k", type=int, default=7, help="fold 개수 (default 7)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = pathlib.Path(args.out_dir)

    # 전체 세트 작성
    build_first_samples(df, out_dir / "fs_train.jsonl")
    build_pair_samples(df,  out_dir / "pw_train.jsonl")
    print("✅ JSONL 생성 완료")

    # K-Fold 작성 (옵션)
    if args.make_fold:
        make_folds_doc(pathlib.Path(args.csv), k=args.k)
