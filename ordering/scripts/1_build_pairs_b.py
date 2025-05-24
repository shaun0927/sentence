#!/usr/bin/env python
# ordering/scripts/1_build_pairs_b.py
# ------------------------------------------------------------------
# test.csv → pairs.jsonl (네 문장 모두 입력)
#
# {"ID":"...", "idx_a":0,"sent_a":"...", "idx_b":1,"sent_b":"...",
#  "idx_c":2,"sent_c":"...", "idx_d":3,"sent_d":"..."}

"""
python ordering/scripts/1_build_pairs_b.py --test_csv data/raw/test.csv --out_jsonl ordering/data/pairs/test_all4_pairs.jsonl
"""

import argparse, json, pathlib, pandas as pd

def main(a):
    df = pd.read_csv(a.test_csv, encoding="utf-8-sig")

    out = []
    for _, row in df.iterrows():
        out.append({
            "ID": row.ID,
            "idx_a": 0, "sent_a": row["sentence_0"],
            "idx_b": 1, "sent_b": row["sentence_1"],
            "idx_c": 2, "sent_c": row["sentence_2"],
            "idx_d": 3, "sent_d": row["sentence_3"],
        })

    p = pathlib.Path(a.out_jsonl)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out),
                 encoding="utf-8")
    print(f"✅ pairs saved → {p}  ({len(out)} records)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--out_jsonl", required=True)
    main(ap.parse_args())
