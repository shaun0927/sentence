#!/usr/bin/env python
# ordering/scripts/1_build_pairs_a.py
# ------------------------------------------------------------------
# first_jsonl + test.csv → 첫 문장 고정 + A·B·C 3-문장 pairs.jsonl
#
# 출력 한 줄 예
# {"ID":"TEST_0000",
#  "first_idx":0,"first_sent":"첫 문장 …",
#  "idx_a":1,"sent_a":"…",
#  "idx_b":2,"sent_b":"…",
#  "idx_c":3,"sent_c":"…" }

"""
python ordering/scripts/1_build_pairs_a.py \
  --test_csv    data/raw/test.csv \
  --first_jsonl data/proc/first_last_final.jsonl \
  --out_jsonl   ordering/data/pairs/test_mid3_pairs.jsonl

python ordering/scripts/1_build_pairs_a.py --test_csv data/raw/test.csv --first_jsonl data/proc/first_only.jsonl --out_jsonl ordering/data/pairs/test_mid3_pairs.jsonl

"""


import argparse, json, pathlib, pandas as pd

def main(a):
    # 1) 고정된 첫 문장 인덱스 로드  ---------------------------------
    first_map = {}
    for obj in map(json.loads, open(a.first_jsonl, encoding="utf-8")):
        # 키 이름이 first 또는 first_idx 어느 쪽이든 허용
        first_map[obj["ID"]] = obj.get("first", obj.get("first_idx"))

    # 2) test.csv 읽기 ---------------------------------------------
    df = pd.read_csv(a.test_csv, encoding="utf-8-sig")

    out_lines = []
    for _, row in df.iterrows():
        rid  = row.ID
        fidx = first_map[rid]                 # 고정 첫 문장 idx

        # 남은 세 문장 인덱스
        rest = [i for i in range(4) if i != fidx]
        ia, ib, ic = rest                     # 정확히 3개

        out_lines.append({
            "ID": rid,
            "first_idx": fidx,
            "first_sent": row[f"sentence_{fidx}"],
            "idx_a": ia, "sent_a": row[f"sentence_{ia}"],
            "idx_b": ib, "sent_b": row[f"sentence_{ib}"],
            "idx_c": ic, "sent_c": row[f"sentence_{ic}"],
        })

    # 3) JSONL 저장 -------------------------------------------------
    out_path = pathlib.Path(a.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(l, ensure_ascii=False)
                                  for l in out_lines),
                        encoding="utf-8")

    print(f"✅ pairs saved → {out_path}  ({len(out_lines)} records)")

# ───────────── CLI ───────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv",   required=True,
                   help="data/raw/test.csv")
    p.add_argument("--first_jsonl", required=True,
                   help="first_sentence 확정 JSONL (first 또는 first_idx 키)")
    p.add_argument("--out_jsonl",  required=True,
                   help="ordering/data/pairs/test_mid3_pairs.jsonl")
    main(p.parse_args())
