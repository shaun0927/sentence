#!/usr/bin/env python
# ordering/scripts/3_make_submission_a.py
# ------------------------------------------------------------------
"""
python ordering/scripts/3_make_submission_a.py 
--votes_jsonl ordering/data/votes/test_mid3_votes.jsonl 
--first_jsonl data/proc/first_only.jsonl 
--out_csv submission_a.csv

python ordering/scripts/3_make_submission_a.py --votes_jsonl ordering/data/votes/test_mid3_votes.jsonl --first_jsonl data/proc/first_only.jsonl --out_csv submission_final.csv
"""
# votes_jsonl( order_mid 3개 ) + first_jsonl → submission.csv

import argparse, json, pathlib, pandas as pd

def main(a):
    # 첫 문장만 고정된 JSONL ── 키 이름이 first / first_idx 둘 다 허용
    first_map = {}
    for j in map(json.loads, open(a.first_jsonl, encoding="utf-8")):
        first_map[j["ID"]] = j.get("first", j.get("first_idx"))

    votes_map = {j["ID"]: j for j in
                 map(json.loads, open(a.votes_jsonl, encoding="utf-8"))}

    rows = []
    for rid, first_i in first_map.items():
        rec = votes_map.get(rid)
        if rec is None:
            raise KeyError(f"missing vote for ID {rid}")

        order_mid = rec["order_mid"]           # [idx?, idx?, idx?] or null
        if order_mid is None:                  # LLM 전부 실패 → 기본 A,B,C
            order_mid = [rec["idx_a"], rec["idx_b"], rec["idx_c"]]

        assert len(order_mid) == 3, f"order_mid len≠3 for {rid}"
        answer = [first_i, *order_mid]         # 총 4개

        rows.append({
            "ID": rid,
            "answer_0": answer[0],
            "answer_1": answer[1],
            "answer_2": answer[2],
            "answer_3": answer[3],
        })

    out = pathlib.Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("ID").to_csv(out, index=False)
    print("✅ submission saved →", out)

# ─────────── CLI ────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--votes_jsonl",  required=True,
                   help="2_query_llm_a.py 결과 JSONL")
    p.add_argument("--first_jsonl",  required=True,
                   help="first_sentence 확정 JSONL (first 또는 first_idx 키)")
    p.add_argument("--out_csv",      required=True,
                   help="내보낼 submission.csv")
    main(p.parse_args())
