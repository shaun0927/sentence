#!/usr/bin/env python
# ordering/scripts/1_build_pairs.py
# ---------------------------------------------------------------------
# first_last_final.jsonl + test.csv  →  두 번째 · 세 번째 문장 쌍 JSONL
# {"ID": "...", "first": i, "last": j, "idx_a": k, "idx_b": l,
#  "sent_a": "...", "sent_b": "..."}

import argparse, json, pathlib, pandas as pd

def main(a):
    df = pd.read_csv(a.test_csv, encoding="utf-8-sig")
    fl = {j["ID"]: j for j in map(json.loads,
                                  open(a.first_last_jsonl, encoding="utf-8"))}

    lines = []
    for _, row in df.iterrows():
        rid = row.ID
        first_i = fl[rid]["first"]; last_i = fl[rid]["last"]
        mids = [i for i in range(4) if i not in (first_i, last_i)]
        ia, ib = mids                     # 정확히 2개 남음

        lines.append({
            "ID": rid,
            "first_idx": first_i,
            "last_idx":  last_i,
            "first_sent": row[f"sentence_{first_i}"],
            "last_sent":  row[f"sentence_{last_i}"],
            "idx_a": ia,
            "idx_b": ib,
            "sent_a": row[f"sentence_{ia}"],
            "sent_b": row[f"sentence_{ib}"],
        })

    out = pathlib.Path(a.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines),
                   encoding="utf-8")
    print(f"✅ pairs saved → {out}  ({len(lines)} records)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv",           required=True)
    ap.add_argument("--first_last_jsonl",   required=True)
    ap.add_argument("--out_jsonl",          required=True)
    main(ap.parse_args())
