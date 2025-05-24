#!/usr/bin/env python
# ordering/scripts/3_make_submission.py
# ---------------------------------------------------------------------
# votes_jsonl + first_last_final.jsonl → submission.csv(ID, answer_0~3)

import argparse, json, pathlib, pandas as pd

def main(a):
    fl = {j["ID"]: j for j in map(json.loads,
                                  open(a.first_last_jsonl, encoding="utf-8"))}
    votes = {j["ID"]: j for j in map(json.loads,
                                     open(a.votes_jsonl, encoding="utf-8"))}

    rows=[]
    for rid, info in fl.items():
        first_i, last_i = info["first"], info["last"]
        order_mid = votes[rid]["order_mid"]          # [idx1, idx2]
        order = [first_i, *order_mid, last_i]
        rows.append({"ID": rid,
                     "answer_0": order[0],
                     "answer_1": order[1],
                     "answer_2": order[2],
                     "answer_3": order[3]})

    out = pathlib.Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("ID").to_csv(out, index=False)
    print("✅ submission saved →", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--votes_jsonl",      required=True)
    p.add_argument("--first_last_jsonl", required=True)
    p.add_argument("--out_csv",          required=True)
    main(p.parse_args())
