#!/usr/bin/env python
"""
7_make_submission.py
────────────────────
test_ppl_scores.jsonl → 제출용 CSV(ID, answer_0~3)
"""

import argparse, json, pathlib, pandas as pd

def main(a):
    rows = []
    for line in pathlib.Path(a.ppl_jsonl).read_text().splitlines():
        obj = json.loads(line)
        ans = obj["best_order"]
        rows.append({
            "ID": obj["ID"],
            "answer_0": ans[0],
            "answer_1": ans[1],
            "answer_2": ans[2],
            "answer_3": ans[3],
        })
    sub = pd.DataFrame(rows).sort_values("ID")
    out = pathlib.Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print("✅ submission saved →", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ppl_jsonl", required=True,
                   help="data/proc/test_ppl_scores.jsonl")
    p.add_argument("--out_csv",   required=True,
                   help="submission/submission.csv")
    main(p.parse_args())
