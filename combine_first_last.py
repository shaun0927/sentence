#!/usr/bin/env python
# combine_first_last.py
# ---------------------------------------------------------
# usage:
#   python combine_first_last.py \
#     --first test_first_top2.jsonl \
#     --last  test_first_topk.jsonl \
#     --out   first_last_final.jsonl
#
# output (JSONL):
#   {"ID": "TEST_0000", "first": 0, "last": 3}

import argparse, json, pathlib

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def main(a):
    first_js = load_jsonl(a.first)
    last_js  = load_jsonl(a.last)

    # ID → (rank, prob) 로 사전화
    first_map = {d["ID"]: (d["rank"], d["prob"]) for d in first_js}
    last_map  = {d["ID"]: (d["rank"], d["prob"]) for d in last_js}

    out_lines = []
    for _id in sorted(first_map.keys()):
        fr_rank, fr_prob = first_map[_id]
        la_rank, la_prob = last_map [_id]

        # 1) 각 파일에서 prob 큰 쪽 선택
        fr_idx_primary = fr_rank[0] if fr_prob[0] >= fr_prob[1] else fr_rank[1]
        fr_idx_second  = fr_rank[1] if fr_idx_primary == fr_rank[0] else fr_rank[0]
        fr_p_primary   = max(fr_prob)

        la_idx_primary = la_rank[0] if la_prob[0] >= la_prob[1] else la_rank[1]
        la_idx_second  = la_rank[1] if la_idx_primary == la_rank[0] else la_rank[0]
        la_p_primary   = max(la_prob)

        first_idx, last_idx = fr_idx_primary, la_idx_primary

        # 2) 중복 시 교체
        if first_idx == last_idx:
            if fr_p_primary >= la_p_primary:
                last_idx = la_idx_second
            else:
                first_idx = fr_idx_second

        out_lines.append(json.dumps(
            {"ID": _id, "first": int(first_idx), "last": int(last_idx)},
            ensure_ascii=False
        ))

    out_path = pathlib.Path(a.out)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"✅ saved {len(out_lines)} records →", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--first", required=True, help="test_first_top2.jsonl")
    p.add_argument("--last",  required=True, help="test_first_topk.jsonl (last prob)")
    p.add_argument("--out",   required=True, help="output jsonl path")
    main(p.parse_args())
