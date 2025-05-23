#!/usr/bin/env python
"""
5_build_candidates.py
─────────────────────
• input  : test.csv  (ID, sentence_0 … sentence_3)
• input  : first_jsonl  (ID, rank=[best1,best2,…])  ← 4_predict_fs.py 산출물
• output : candidates_jsonl  (ID, candidates=[ [4개 idx], … ])
         └ 각 ID 당 최대 top_k * 6  (= 남은 3! ) 개 후보

예시:
{"ID":"TEST_0000",
 "candidates":[[1,0,2,3],[1,0,3,2],[1,2,0,3] … ]}
"""
import argparse, json, itertools, pathlib, pandas as pd
from tqdm.auto import tqdm


def build_for_one(best_first, top_k=2):
    """
    best_first : [idx0, idx1, …]  (첫 문장 후보 TOP-k, 확률 높은 순)
    반환: 여러 permutation(list[int])
    """
    rest = [x for x in range(4) if x not in best_first[:top_k]]
    cand_all = []
    for first in best_first[:top_k]:
        others = [x for x in range(4) if x != first]
        for perm in itertools.permutations(others, 3):
            cand_all.append([first, *perm])
    return cand_all


def main(args):
    test_df = pd.read_csv(args.test_csv)
    id2rank = {}
    for line in open(args.first_jsonl, encoding="utf-8"):
        obj = json.loads(line)
        id2rank[obj["ID"]] = obj["rank"]          # e.g. [1,0]

    out_lines, too = [], 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        rid = row["ID"]
        rank = id2rank.get(rid)
        assert rank, f"missing ID {rid} in first-sentence jsonl"

        cands = build_for_one(rank, top_k=args.top_k)
        out_lines.append(json.dumps({"ID": rid, "candidates": cands},
                                    ensure_ascii=False))
        too += len(cands)

    out_p = pathlib.Path(args.out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(out_lines), encoding="utf-8")

    print(f"✅ saved → {out_p}   ({len(test_df)} IDs, {too} candidates total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv",    required=True,
                    help="data/raw/test.csv")
    ap.add_argument("--first_jsonl", required=True,
                    help="data/proc/test_first_top2.jsonl")
    ap.add_argument("--out_jsonl",   required=True,
                    help="data/proc/test_candidates.jsonl")
    ap.add_argument("--top_k", type=int, default=2,
                    help="How many first-sentence candidates to expand")
    main(ap.parse_args())
