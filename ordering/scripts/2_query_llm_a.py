#!/usr/bin/env python
# ordering/scripts/2_query_llm_a.py
# ------------------------------------------------------------------
# 첫 문장 고정 + 3-문장 순서 맞추기 (편향 완화 버전)
#
#   python 2_query_llm_a.py --pairs_jsonl ordering/data/pairs/test_mid3_pairs.jsonl --out_jsonl ordering/data/votes/test_mid3_votes.jsonl
# ------------------------------------------------------------------
import argparse, json, pathlib, random, re, statistics
import yaml, requests
from tqdm.auto import tqdm

# ────────────────────────── 프롬프트 템플릿 ───────────────────────────
BASE_TEMPLATE = """\
다음 네 문장은 하나의 글을 이룹니다. 첫 문장은 이미 확정되어 있으며
{L1},{L2},{L3} 세 문장을 가장 자연스러운 순서(앞→뒤)로 배열하세요.

첫 문장: "{FIRST}"

문장 {L1}: "{S1}"
문장 {L2}: "{S2}"
문장 {L3}: "{S3}"

정답을 {L1}{L2}{L3}, {L1}{L3}{L2} … 처럼
***3글자 문자열*** 한 줄로만 출력하세요.  (예: {L2}{L1}{L3})
"""

PERM_ABC = [''.join(p) for p in ("ABC","ACB","BAC","BCA","CAB","CBA")]

ans_pat = re.compile(r"\b([ABC]{3})\b", re.I)

# ─────────────── util ──────────────────────────────────────────────
def load_pairs(path):
    txt = pathlib.Path(path).read_text("utf-8").splitlines()
    return [json.loads(l) for l in txt]

def build_prompt(rec, lbl_order):
    """lbl_order=['C','A','B'] 처럼 무작위 라벨 순서"""
    lbl_map = dict(zip(lbl_order, "ABC"))          # 무작위 → 고정
    sents = {l: rec[f"sent_{lbl_map[l].lower()}"] for l in lbl_order}
    return BASE_TEMPLATE.format(
        L1=lbl_order[0], L2=lbl_order[1], L3=lbl_order[2],
        FIRST = rec["first_sent"],
        S1 = sents[lbl_order[0]],
        S2 = sents[lbl_order[1]],
        S3 = sents[lbl_order[2]],
    ), lbl_map

def extract_ans(txt):
    m = ans_pat.search(txt.strip())
    return m.group(1).upper() if m and m.group(1).upper() in PERM_ABC else None

def majority(votes):
    good = [v for v in votes if v]
    if not good:
        return None
    try:
        return statistics.mode(good)
    except statistics.StatisticsError:       # tie
        return random.choice(good)

def call_llm(server, model, prompts, n, temp, top_p, timeout=120):
    body = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 12,           # 짧게
        "temperature": temp,
        "top_p": top_p,
        "n": n,
    }
    url = server.rstrip("/") + "/v1/completions"
    r   = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    blocks = data if isinstance(data, list) else [data]
    return [[c["text"] for c in blk["choices"]] for blk in blocks]

def order_from_str(perm_str, lbl_map, rec):
    letter2idx = {"A": rec["idx_a"], "B": rec["idx_b"], "C": rec["idx_c"]}
    canonical  = ''.join(lbl_map[ch] for ch in perm_str)    # → ABC 기준
    return [letter2idx[ch] for ch in canonical]

# ─────────────────────────────── main ──────────────────────────────
def main(a):
    models = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))["models"]
    pairs  = load_pairs(a.pairs_jsonl)

    out_lines = []
    for m in models:
        name   = m["name"];  model = m["hf_id"];  server = m["server_url"]
        bs     = int(m.get("batch_size", 1))
        n_sample = int(m.get("n_sample", 32))
        n_view   = int(m.get("n_view",   5))      # 라벨 무작위 view 수
        temp   = float(m.get("temperature", 0.8)); top_p = float(m.get("top_p", 0.9))

        print(f"\n⚙️  {model} | batch={bs} | n_sample={n_sample} | view={n_view}")

        for i in tqdm(range(0, len(pairs), bs),
                      total=(len(pairs) + bs - 1)//bs,
                      desc=name):
            batch_recs = pairs[i:i+bs]

            prompts, maps = [], []
            for rec in batch_recs:
                for _ in range(n_view):
                    lbl_order = random.sample("ABC", 3)
                    ptxt, mp  = build_prompt(rec, lbl_order)
                    prompts.append(ptxt); maps.append((rec, mp))

            # 호출
            reps = call_llm(server, model, prompts,
                            n_sample, temp, top_p)
            reps_flat = [t for sub in reps for t in sub]

            # 집계
            ptr = 0
            for rec in batch_recs:
                votes_str, votes_idx = [], []
                for _ in range(n_view):
                    chunk = reps_flat[ptr : ptr + n_sample]
                    rec_, mp = maps[ptr // n_sample]
                    ptr += n_sample
                    for txt in chunk:
                        s = extract_ans(txt)
                        if s:
                            votes_str.append(s)
                            votes_idx.append(order_from_str(s, mp, rec_))

                best = majority(votes_str)
                order_mid = order_from_str(best, mp, rec) if best else None

                out_lines.append(json.dumps({
                    "ID": rec["ID"],
                    "order_mid": order_mid,         # [idx2, idx3, idx4] or null
                    "votes": votes_str,
                    "model": name
                }, ensure_ascii=False))

    out_p = pathlib.Path(a.out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text('\n'.join(out_lines), encoding="utf-8")
    print("✅ votes saved →", out_p)

# ───────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--models_yaml", default="ordering/6_models.yaml")
    ap.add_argument("--out_jsonl",   required=True)
    main(ap.parse_args())
