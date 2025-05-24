#!/usr/bin/env python
"""
make_mid_pair.py
────────────────────────────────────────────────────────────
첫·마지막 문장이 확정된 JSONL → 중앙 두 문장 pair 학습 데이터 생성

사용 예:
python make_mid_pair.py \
    --csv               data/raw/train.csv \
    --first_last_jsonl  data/proc/first_last_final.jsonl \
    --out_jsonl         data/proc_mid/pair_train_mid.jsonl
"""

import argparse, json, pathlib
import pandas as pd

def main(a):
    df = pd.read_csv(a.csv, encoding="utf-8-sig")
    fl_map = {j["ID"]: j for j in map(json.loads,
                                      open(a.first_last_jsonl, encoding="utf-8"))}

    out_lines = []
    for _, row in df.iterrows():
        aid = row.ID
        first_i = fl_map[aid]["first"]
        last_i  = fl_map[aid]["last"]
        mids    = [i for i in range(4) if i not in (first_i, last_i)]
        ia, ib  = mids   # 정확히 두 개

        # (ia, ib) 순서가 맞으면 label = 1
        out_lines.append({
            "doc_id": aid,
            "text_a": row[f"sentence_{ia}"],
            "text_b": row[f"sentence_{ib}"],
            "label": 1
        })
        # (ib, ia) 순서가 바뀌면 label = 0
        out_lines.append({
            "doc_id": aid,
            "text_a": row[f"sentence_{ib}"],
            "text_b": row[f"sentence_{ia}"],
            "label": 0
        })

    out_path = pathlib.Path(a.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for j in out_lines:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"✅ 저장 완료: {out_path}  (총 {len(out_lines)} 라인)")

# ────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",               required=True,
                   help="원본 문장 CSV (sentence_0 … sentence_3 열 포함)")
    p.add_argument("--first_last_jsonl",  required=True,
                   help="first_last_final.jsonl 경로")
    p.add_argument("--out_jsonl",         required=True,
                   help="생성될 중앙 Pair 학습 JSONL")
    main(p.parse_args())
