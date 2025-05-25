#!/usr/bin/env python
# ordering/scripts/0_choose_first_llm.py
# ------------------------------------------------------------------
# test_first_top2.jsonl + test.csv  →  first_only.jsonl({ID, first})
#
#  • 두 후보( rank[0], rank[1] ) 중 실제 첫 문장을 LLM-SC(무작위 라벨)로 선택
#  • 편향 완화 : ① A/B 라벨 무작위, ② 문자열 답(‘A’/‘B’) 사용,
#                ③ n_sample=32 · view=3, ④ max_tokens=10
#
# 사용 예:
#   python ordering/scripts/0_choose_first_llm.py --test_csv data/raw/test.csv --top2_jsonl data/proc/test_first_top2.jsonl --out_jsonl data/proc/first_only.jsonl
# ------------------------------------------------------------------
import argparse, json, pathlib, random, re, statistics
import yaml, requests, pandas as pd
from tqdm.auto import tqdm

# ────────────────────────────── 프롬프트 ────────────────────────────────
BASE_PROMPT = """\
다음 두 문장 가운데 **글의 첫 문장**(도입부)에 더 자연스러운 문장을 고르십시오.

문장 {L1}: "{S1}"
문장 {L2}: "{S2}"

첫 문장에 어울리는 쪽의 라벨만 ***한 글자*** 로 답하십시오.
가능한 정답: {L1} 또는 {L2}

예) {L1}
"""

# ────────────────────── 정규식 & 집계 함수 ──────────────────────────────
ans_pat = re.compile(r"\b([AB])\b", re.I)   # A or B  (대소문자 허용)

def extract(txt: str):
    m = ans_pat.search(txt.strip())
    return m.group(1).upper() if m else None

def majority(votes):
    valid = [v for v in votes if v]
    if not valid:
        return None
    try:
        return statistics.mode(valid)
    except statistics.StatisticsError:      # tie
        return random.choice(valid)

# ────────────────────────── LLM 호출 래퍼 ───────────────────────────────
def call_llm(server, model, prompts, n, temp, top_p, timeout=120):
    url  = server.rstrip("/") + "/v1/completions"
    body = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 10,
        "temperature": temp,
        "top_p": top_p,
        "n": n,
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    blocks = data if isinstance(data, list) else [data]
    return [[c["text"] for c in blk["choices"]] for blk in blocks]

# ───────────────────────────────── main ─────────────────────────────────
def main(a):
    # 0. 데이터 로드 ------------------------------------------------------
    df = pd.read_csv(a.test_csv, encoding="utf-8-sig")
    pool = {row.ID: [row[f"sentence_{i}"] for i in range(4)]
            for _, row in df.iterrows()}

    top2_list = [json.loads(l) for l in open(a.top2_jsonl, encoding="utf-8")]
    models = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))["models"]

    # 1. LLM 평가 ---------------------------------------------------------
    results = {}             # ID → (first_idx, votes)
    for m in models:
        name   = m["name"];  model = m["hf_id"];  server = m["server_url"]
        n_sample = int(m.get("n_sample", 32))
        n_view   = int(m.get("n_view", 3))        # 라벨 무작위 view 수
        temp   = float(m.get("temperature", 0.8));  top_p = float(m.get("top_p", 0.9))

        print(f"\n⚙️  {model} | n_sample={n_sample} | view={n_view}")
        for obj in tqdm(top2_list, desc=name):
            rid, (i, j) = obj["ID"], obj["rank"]
            sa, sb = pool[rid][i], pool[rid][j]

            all_votes = []
            for _ in range(n_view):
                # 라벨 무작위: 절반 확률로 A/B 스왑
                if random.random() < 0.5:
                    L1, S1, idx1 = "A", sa, i
                    L2, S2, idx2 = "B", sb, j
                else:
                    L1, S1, idx1 = "A", sb, j
                    L2, S2, idx2 = "B", sa, i

                prompt = BASE_PROMPT.format(L1=L1, L2=L2, S1=S1, S2=S2)
                texts  = call_llm(server, model, [prompt],
                                  n_sample, temp, top_p)[0]
                votes  = [extract(t) for t in texts]
                # 라벨 → 인덱스로 바꾸어 저장
                all_votes.extend(idx1 if v == "A" else idx2
                                 for v in votes if v)

            choice = majority(all_votes)
            # fallback: 불확실 → prob 상위 0(=i)
            if choice is None:
                choice = i

            results[rid] = (choice, all_votes)

    # 2. 저장 -------------------------------------------------------------
    out_lines = [
        json.dumps({"ID": rid, "first": idx, "votes": v}, ensure_ascii=False)
        for rid, (idx, v) in results.items()
    ]
    out_p = pathlib.Path(a.out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(out_lines), encoding="utf-8")
    print("✅ first_only saved →", out_p)

# -----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv",    required=True)
    ap.add_argument("--top2_jsonl",  required=True,
                    help="data/proc/test_first_top2.jsonl")
    ap.add_argument("--models_yaml", default="ordering/6_models.yaml")
    ap.add_argument("--out_jsonl",   required=True,
                    help="data/proc/first_only.jsonl")
    main(ap.parse_args())
