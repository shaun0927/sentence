#!/usr/bin/env python
# make_mid_folds.py  ─────────────────────────────────────
# 중앙 Pair 학습 JSONL을 K-fold로 나눠
#   fold_dir/pair_fold_0.jsonl … pair_fold_{K-1}.jsonl
# 를 만듭니다.

import argparse, json, pathlib, random
from collections import defaultdict

def main(a):
    random.seed(42)
    # (1) doc_id 리스트 수집
    doc_ids = []
    for ex in map(json.loads, open(a.train_jsonl, encoding="utf-8")):
        if ex["label"] == 1:           # 쌍 하나당 label=1 한 줄만 쓰면 충분
            doc_ids.append(ex["doc_id"])
    doc_ids = sorted(set(doc_ids))
    random.shuffle(doc_ids)

    # (2) Kfold 분할
    K = a.k
    fold_ids = [doc_ids[i::K] for i in range(K)]

    out_dir = pathlib.Path(a.fold_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, ids in enumerate(fold_ids):
        path = out_dir / f"pair_fold_{k}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for did in ids:
                f.write(json.dumps({"doc_id": did}, ensure_ascii=False) + "\n")
        print(f"✅ fold{k}: {len(ids)}개  → {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True,
                    help="data/proc_mid/pair_train_mid.jsonl")
    ap.add_argument("--fold_dir",    required=True,
                    help="생성될 폴더 경로")
    ap.add_argument("-k", type=int, default=5,
                    help="fold 개수 (기본 5)")
    main(ap.parse_args())
