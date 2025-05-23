#!/usr/bin/env python
"""
1_make_dataset.py
-----------------
train.csv (4 문장 순서·정답 포함) →  
 • fs_train.jsonl  : 첫 문장(binary) 학습 세트
 • pw_train.jsonl  : 문장쌍 순서(binary) 학습 세트
 • (선택) k-fold split 파일

각 JSONL 한 줄 형식
{"text": "<문장>", "label": 1}
{"text": "<문장A> [SEP] <문장B>", "label": 0}
"""
import argparse, json, pathlib, random
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
random.seed(SEED)


def build_first_samples(df, out_path: pathlib.Path):
    """첫 문장 여부 binary 라벨 샘플 생성"""
    rows = []
    for _, row in df.iterrows():
        gold_first = int(row["answer_0"])  # 0~3
        for i in range(4):
            rows.append(
                {"text": row[f"sentence_{i}"],
                 "label": int(i == gold_first)}
            )
    random.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_pair_samples(df, out_path: pathlib.Path):
    """모든 (A,B) 쌍 → A가 앞이면 1, 아니면 0"""
    rows = []
    for _, row in df.iterrows():
        gold_order = [int(row[f"answer_{k}"]) for k in range(4)]
        rank = {idx: pos for pos, idx in enumerate(gold_order)}
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                label = int(rank[i] < rank[j])  # i 앞?
                text = f"{row[f'sentence_{i}']} [SEP] {row[f'sentence_{j}']}"
                rows.append({"text": text, "label": label})
    random.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_folds(jsonl_path: pathlib.Path, k: int = 5):
    """Stratified K-fold 파일 저장 (first-sentence 세트 전용)"""
    df = pd.read_json(jsonl_path, lines=True)
    X = df["text"].values
    y = df["label"].values
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_dir = jsonl_path.parent.parent / "fold"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, val_idx) in enumerate(skf.split(X, y)):
        fold_path = fold_dir / f"fold_{i}.jsonl"
        df.iloc[val_idx].to_json(fold_path, orient="records",
                                 lines=True, force_ascii=False)
        print(f"[fold] saved {fold_path.name}  ({len(val_idx)} lines)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train.csv 경로")
    ap.add_argument("--out_dir", default="data/proc",
                    help="jsonl 출력 디렉터리")
    ap.add_argument("--make_fold", action="store_true",
                    help="Stratified 5-fold jsonl 생성")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = pathlib.Path(args.out_dir)
    build_first_samples(df, out_dir / "fs_train.jsonl")
    build_pair_samples(df,  out_dir / "pw_train.jsonl")
    print("✅ JSONL 생성 완료")

    if args.make_fold:
        make_folds(out_dir / "fs_train.jsonl")
