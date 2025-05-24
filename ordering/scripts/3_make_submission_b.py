#!/usr/bin/env python
# ordering/scripts/3_make_submission_b.py
# ---------------------------------------------------------------
# votes_jsonl(order_all 포함) → submission.csv(ID, answer_0~3)

"""
python ordering/scripts/3_make_submission_b.py --votes_jsonl ordering/data/votes/test_all4_votes.jsonl --out_csv submission_b.csv
"""

import argparse, json, pathlib, pandas as pd

def main(a):
    votes = {j["ID"]: j for j in map(json.loads,
                                     open(a.votes_jsonl, encoding="utf-8"))}
    rows=[]
    for rid, rec in votes.items():
        ord4 = rec["order_all"]
        if ord4 is None:
            raise ValueError(f"order_all missing for {rid}")
        rows.append({"ID": rid,
                     "answer_0": ord4[0],
                     "answer_1": ord4[1],
                     "answer_2": ord4[2],
                     "answer_3": ord4[3]})

    out = pathlib.Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("ID").to_csv(out,index=False)
    print("✅ submission saved →", out)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--votes_jsonl",required=True)
    p.add_argument("--out_csv",required=True)
    main(p.parse_args())
