#!/usr/bin/env python
# ordering/scripts/2_query_llm_b.py
# -------------------------------------------------------------
# 4-문장 전체 순열(24) → LLM self-consistency   편향 최소화 버전
#
#   python 2_query_llm_b.py --pairs_jsonl ordering/data/pairs/test_all4_pairs.jsonl --out_jsonl ordering/data/votes/test_all4_votes.jsonl
# -------------------------------------------------------------
import argparse, json, pathlib, random, re, statistics, itertools
import yaml, requests
from tqdm.auto import tqdm

# ───────────────────────────────── 프롬프트 ────────────────────────────────
BASE_TEMPLATE = """\
아래 네 문장은 하나의 글을 이룹니다. 문장 {L1},{L2},{L3},{L4} 를
가장 자연스러운 순서(앞→뒤)로 배열하세요.

문장 {L1}: "{S1}"
문장 {L2}: "{S2}"
문장 {L3}: "{S3}"
문장 {L4}: "{S4}"

정답을 {L1}{L2}{L3}{L4} • {L1}{L2}{L4}{L3} … 과 같이
{L1}{L2}{L3}{L4},{L1}{L2}{L4}{L3}, … ,{L4}{L3}{L2}{L1}
***4글자 문자열*** 로 한 줄만 출력하세요.
(예: {L2}{L1}{L3}{L4})
"""

PERM_4 = [''.join(p) for p in itertools.permutations("ABCD")]

perm_pat = re.compile(r"\b([ABCD]{4})\b")

# ───────────────────────────── util 함수 ───────────────────────────────────
def load_pairs(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text("utf-8").splitlines()]

def build_prompt(rec, lbl_order):
    """lbl_order=['C','A','D','B'] 처럼 임의 순서"""
    lbl_map = dict(zip(lbl_order, "ABCD"))      # 실제→고정
    sents = {l: rec[f"sent_{lbl_map[l].lower()}"] for l in lbl_order}
    return BASE_TEMPLATE.format(
        L1=lbl_order[0], L2=lbl_order[1],
        L3=lbl_order[2], L4=lbl_order[3],
        S1=sents[lbl_order[0]],
        S2=sents[lbl_order[1]],
        S3=sents[lbl_order[2]],
        S4=sents[lbl_order[3]],
    ), lbl_map

def extract_ans(txt):
    m = perm_pat.search(txt.strip())
    return m.group(1) if m and m.group(1) in PERM_4 else None

def majority(votes):
    valid = [v for v in votes if v]
    if not valid:
        return None
    try:
        return statistics.mode(valid)
    except statistics.StatisticsError:          # tie
        return random.choice(valid)

def call_llm(server, model, prompts, n, temp, top_p, timeout=120):
    body = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 12,
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

def order_from_str(order_str, lbl_map, rec):
    """order_str 예: 'CADB' → 원본 인덱스 순서(list[int])"""
    letter2idx = {"A": rec["idx_a"], "B": rec["idx_b"],
                  "C": rec["idx_c"], "D": rec["idx_d"]}
    # order_str 은 무작위 라벨을 그대로 사용 → lbl_map 로 변환
    canonical = ''.join(lbl_map[ch] for ch in order_str)
    return [letter2idx[ch] for ch in canonical]

# ─────────────────────────────── main ─────────────────────────────────────
def main(a):
    models = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))["models"]
    pairs  = load_pairs(a.pairs_jsonl)

    out_lines = []
    for m in models:
        name = m["name"]; model = m["hf_id"]; server = m["server_url"]
        bs = int(m.get("batch_size", 1))
        n_sample = int(m.get("n_sample", 32))
        n_view   = int(m.get("n_view",   5))      # 라벨 무작위 view 수
        temp = float(m.get("temperature", 0.8)); top_p = float(m.get("top_p", 0.9))

        print(f"\n⚙️ {model} | batch={bs} | n_sample={n_sample} | view={n_view}")
        for i in tqdm(range(0, len(pairs), bs),
                      total=(len(pairs) + bs - 1)//bs,
                      desc=name):
            batch_recs = pairs[i:i+bs]

            # view 별 프롬프트 생성
            prompts, view_maps = [], []
            for rec in batch_recs:
                for v in range(n_view):
                    lbl_order = random.sample("ABCD", 4)
                    ptxt, mp = build_prompt(rec, lbl_order)
                    prompts.append(ptxt)
                    view_maps.append((rec, mp))

            # LLM 호출 (view*bs 개)
            reps = call_llm(server, model, prompts,
                            n_sample, temp, top_p)
            # 납작 리스트로 정렬
            flat_reps = [t for sub in reps for t in sub]

            # 결과 집계
            ptr = 0
            for rec in batch_recs:
                votes = []
                for _ in range(n_view):
                    ans_list = flat_reps[ptr: ptr + n_sample]
                    rec_, mp = view_maps[ptr // n_sample]
                    ptr += n_sample
                    for txt in ans_list:
                        s = extract_ans(txt)
                        if s:
                            votes.append(order_from_str(s, mp, rec_))
                # Canonical tuple 로 majority
                vote_strs = ['-'.join(map(str, v)) for v in votes]
                best = majority(vote_strs)
                order_all = list(map(int, best.split('-'))) if best else None

                out_lines.append(json.dumps({
                    "ID": rec["ID"],
                    "order_all": order_all,
                    "votes": vote_strs,
                    "model": name
                }, ensure_ascii=False))

    p = pathlib.Path(a.out_jsonl)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(out_lines), encoding="utf-8")
    print("✅ votes saved →", p)

# ─────────────────────────── CLI ──────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--models_yaml", default="ordering/6_models.yaml")
    ap.add_argument("--out_jsonl",   required=True)
    main(ap.parse_args())
