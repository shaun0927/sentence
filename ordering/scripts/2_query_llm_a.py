#!/usr/bin/env python
# ordering/scripts/2_query_llm_a.py
# ------------------------------------------------------------------
# 첫 문장만 고정된 3-문장 순서를 LLM self-consistency로 맞추는 스크립트

"""
python ordering/scripts/2_query_llm_a.py
--pairs_jsonl ordering/data/pairs/test_mid3_pairs.jsonl
--out_jsonl ordering/data/votes/test_mid3_votes.jsonl

python ordering/scripts/2_query_llm_a.py --pairs_jsonl ordering/data/pairs/test_mid3_pairs.jsonl --out_jsonl ordering/data/votes/test_mid3_votes.jsonl
"""

import argparse, json, pathlib, re, statistics, random, yaml, requests
from tqdm.auto import tqdm

# ───────────── 내장 프롬프트 ─────────────
PROMPT_TEMPLATE = """\
다음 네 문장은 하나의 글을 이룹니다.
첫 문장은 이미 결정되어 있고,
문장 A, B, C 세 개는 그 뒤를 이어갈 **중간·마지막 문장**입니다.

첫 문장: "{first_sent}"

문장 A: "{sent_a}"
문장 B: "{sent_b}"
문장 C: "{sent_c}"

문장 A, B, C를 자연스러운 순서로 배열하세요.
가능한 배열은 아래 6가지입니다.
1) A → B → C
2) A → C → B
3) B → A → C
4) B → C → A
5) C → A → B
6) C → B → A

정답을 \\boxed{{{{번호}}}} 한 줄로만 출력하세요.
반드시 첫 토큰이 ‘\\boxed{{’ 로 시작해야 합니다.
1~6의 정수 답만 가능합니다.

예) B → A → C이면 \\boxed{{{{3}}}}
    C → A → B이면 \\boxed{{{{5}}}}
"""


# ───────────── helpers ────────────────────────
def load_pairs(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text("utf-8").splitlines()]

def render_prompt(rec):
    return PROMPT_TEMPLATE.format(
        first_sent = rec["first_sent"],
        sent_a     = rec["sent_a"],
        sent_b     = rec["sent_b"],
        sent_c     = rec["sent_c"],
    )

# 1~6 다수결, 동률이면 랜덤
def majority_numeric(votes):
    # None 제거 후 숫자 필터링
    numeric = [v for v in votes if isinstance(v, str) and v in "123456"]
    if not numeric:
        return None
    try:
        return statistics.mode(numeric)
    except statistics.StatisticsError:   # 동률
        return random.choice(numeric)

boxed_pat  = re.compile(r"\\boxed\{\s*([1-6])\s*\}")
backup_pat = re.compile(r"\b([1-6])\b")

def extract_final(txt):
    if m := boxed_pat.search(txt):
        return m.group(1)
    if m := backup_pat.search(txt.strip()):
        return m.group(1)
    return None

# 6개 번호 → 인덱스 순서 매핑
ORDER_MAP = {
    "1": lambda r: [r["idx_a"], r["idx_b"], r["idx_c"]],
    "2": lambda r: [r["idx_a"], r["idx_c"], r["idx_b"]],
    "3": lambda r: [r["idx_b"], r["idx_a"], r["idx_c"]],
    "4": lambda r: [r["idx_b"], r["idx_c"], r["idx_a"]],
    "5": lambda r: [r["idx_c"], r["idx_a"], r["idx_b"]],
    "6": lambda r: [r["idx_c"], r["idx_b"], r["idx_a"]],
}

def call_llm(server, model, prompts, n, temp, top_p, timeout=120):
    url  = server.rstrip("/") + "/v1/completions"
    body = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 40,
        "temperature": temp,
        "top_p": top_p,
        "n": n,
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    blocks = data if isinstance(data, list) else [data]
    return [[c["text"] for c in blk["choices"]] for blk in blocks]

# ───────── retry / self-consistency ─────────
MAX_RETRY = 8

def query_one(server, model, prompt, n, temp, top_p):
    reps  = call_llm(server, model, [prompt], n, temp, top_p)[0]
    votes = [extract_final(t) for t in reps]
    choice = majority_numeric(votes)
    if choice:
        return choice, [v for v in votes if v]

    for _ in range(MAX_RETRY):
        txt = call_llm(server, model, [prompt], 1, 0.0, 1.0)[0][0]
        v = extract_final(txt)
        if v:
            return v, [v]

    return None, []                       # 완전 실패

# ───────────── main ─────────────────────────
def main(a):
    cfg_models = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))["models"]
    pairs      = load_pairs(a.pairs_jsonl)

    out_lines=[]
    for m in cfg_models:
        name   = m["name"];  model = m["hf_id"];  server = m["server_url"]
        bs     = int(m.get("batch_size", 1))
        n_samp = int(m.get("n_sample", 3))
        temp   = float(m.get("temperature", 0.7)); top_p = float(m.get("top_p", 0.9))

        print(f"\n⚙️  {model} | batch={bs} | n_sample={n_samp}")

        for i in tqdm(range(0, len(pairs), bs),
                      total=(len(pairs)+bs-1)//bs,
                      desc=name):
            batch = pairs[i:i+bs]
            prompts = [render_prompt(r) for r in batch]

            try:
                batch_texts = call_llm(server, model, prompts, n_samp, temp, top_p)
            except Exception as e:
                print("❗ 호출 실패:", e)
                batch_texts = [[""]*n_samp for _ in prompts]

            for rec, texts, prompt in zip(batch, batch_texts, prompts):
                votes  = [extract_final(t) for t in texts]
                choice = majority_numeric(votes)

                if choice is None:
                    choice, votes = query_one(server, model, prompt,
                                              n_samp, temp, top_p)

                order_mid = ORDER_MAP[choice](rec) if choice else None

                out_lines.append(json.dumps({
                    "ID": rec["ID"],
                    "order_mid": order_mid,          # null → 후처리
                    "votes": [v for v in votes if v],
                    "model": name
                }, ensure_ascii=False))

    out_p = pathlib.Path(a.out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(out_lines), encoding="utf-8")
    print("✅ votes saved →", out_p)

# ───────────── CLI ───────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--models_yaml", default="ordering/6_models.yaml")
    ap.add_argument("--out_jsonl",   required=True)
    main(ap.parse_args())
