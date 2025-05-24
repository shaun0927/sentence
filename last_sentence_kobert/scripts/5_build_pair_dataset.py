#!/usr/bin/env python
"""
5_build_pair_dataset.py
────────────────────────────────────────────────────────────
첫 문장 BERT OOF 결과  ➜  Pair-BERT 학습용 top-2 쌍 생성 스크립트

▸ 입력
    --train_csv   : train.csv  (4문장 + answer_0~3)
    --oof_prob    : oof_prob.npy  (N,2)  soft-logit 혹은 softmax
    --oof_meta    : oof_meta.npy  (N,2)  [doc_id , sent_idx]
    --fold_dir    : (선택) 첫 문장 Fold jsonl 들이 있는 디렉터리

▸ 출력
    --out_pair      pair_train.jsonl          (기본: data/proc/pair_train.jsonl)
    --fold_pair_dir pair_fold_i.jsonl 저장    (기본: data/fold_pair)
"""
import argparse, json, pathlib, numpy as np, pandas as pd
from collections import defaultdict

# ------------------------------------------------------------------ utilities
def write_jsonl(rows, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_fold_doc_ids(fs_fold_dir: pathlib.Path):
    """fold_i.jsonl → set(doc_id)  매핑"""
    mapping = {}
    for fp in sorted(fs_fold_dir.glob("fold_*.jsonl")):
        fold_idx = int(fp.stem.split("_")[-1])
        ids = {json.loads(l)["doc_id"] for l in fp.open(encoding="utf-8")}
        mapping[fold_idx] = ids
    return mapping

# ------------------------------------------------------------------ main
def main(args):
    prob = np.load(args.oof_prob)                 # (N,2)
    meta = np.load(args.oof_meta).astype(int)     # (N,2)  [doc_id , sent_idx]
    assert prob.shape[0] == meta.shape[0], "prob/meta 길이가 다릅니다."
    assert (prob.shape[0] % 4) == 0, "N 이 4의 배수가 아닙니다."

    # ── ① 문서별 softmax 상위 2 문장 추출 ────────────────────────
    per_doc = defaultdict(list)      # doc_id → [(idx , P_last)]
    for (doc, idx), p_last in zip(meta, prob[:, 1]):
        per_doc[doc].append((idx, float(p_last)))

    # ── ② train.csv 원문 로드 ────────────────────────────────────
    df = pd.read_csv(args.train_csv)

    rows              = []           # pair_train.jsonl
    fold_rows         = defaultdict(list)
    fold_id_mapping   = (load_fold_doc_ids(pathlib.Path(args.fold_dir))
                         if args.fold_dir else {})

    for doc_id, lst in per_doc.items():
        lst.sort(key=lambda x: -x[1])          # 확률 desc
        (idx_a, p_a), (idx_b, p_b) = lst[:2]   # 항상 2개 존재 (4문장 기준)

        row_src = df.iloc[doc_id]
        text_a, text_b = row_src[f"sentence_{idx_a}"], row_src[f"sentence_{idx_b}"]
        gold_last     = int(row_src["answer_3"])
        label          = int(idx_a == gold_last)

        ex = {
            "text_a": text_a,
            "text_b": text_b,
            "delta_soft": round(p_a - p_b, 6),   # 부동소수 안정
            "label": label,
            "doc_id": int(doc_id)
        }
        rows.append(ex)

        # Fold 상속 (선택)
        for k, id_set in fold_id_mapping.items():
            if doc_id in id_set:
                fold_rows[k].append(ex)
                break

    # ── ③ 저장 ─────────────────────────────────────────────────
    write_jsonl(rows, pathlib.Path(args.out_pair))
    print(f"pair_train.jsonl  저장 완료 ▶ {len(rows):,} 줄")

    if fold_rows:
        fold_pair_dir = pathlib.Path(args.fold_pair_dir)
        for k, r in fold_rows.items():
            fp = fold_pair_dir / f"pair_fold_{k}.jsonl"
            write_jsonl(r, fp)
            print(f"pair_fold_{k}.jsonl  ({len(r):,} 줄)")

# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="raw train.csv")
    ap.add_argument("--oof_prob",  required=True, help="oof_prob.npy")
    ap.add_argument("--oof_meta",  required=True, help="oof_meta.npy")
    ap.add_argument("--out_pair",  default="data/proc/pair_train.jsonl")
    ap.add_argument("--fold_dir",  help="fs fold_* dir (optional)")
    ap.add_argument("--fold_pair_dir", default="data/fold_pair")
    main(ap.parse_args())
