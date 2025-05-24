#!/usr/bin/env python
# ordering/scripts/2_query_llm.py
# ------------------------------------------------------------------
# 두 문장 쌍 JSONL + ordering/6_models.yaml → LLM 질의 → 다수결 결과 저장


import argparse, json, pathlib, re, statistics, random, yaml, requests
from tqdm.auto import tqdm

# ───────────── 내장 프롬프트 템플릿 ─────────────
PROMPT_TEMPLATE = """\
다음 네 문장은 하나의 글을 이룹니다.
첫 문장과 마지막 문장은 이미 결정되어 있고,
문장 A, B 두 개는 그 사이에 들어갈 **중간 문장**입니다.

첫 문장: "{first_sent}"
마지막 문장: "{last_sent}"

문장 A: "{sent_a}"
문장 B: "{sent_b}"

문장 A와 문장 B를 자연스러운 순서로 배열하세요.
1) A → B
2) B → A

정답을 \\boxed{{1}} 또는 \\boxed{{2}} 형태로만 출력하십시오.
예) \\boxed{{1}}
"""

# ───────────── helpers ────────────────────────
def load_pairs(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text("utf-8").splitlines()]

def render_prompt(rec):
    return PROMPT_TEMPLATE.format(
        first_sent = rec["first_sent"],
        last_sent  = rec["last_sent"],
        sent_a     = rec["sent_a"],
        sent_b     = rec["sent_b"],
    )

def majority_numeric(votes):
    """빈칸(None) 제외 후 다수결, 동률이면 임의 선택."""
    numeric = [v for v in votes if v in ("1", "2")]
    if not numeric:
        return None
    try:
        return statistics.mode(numeric)
    except statistics.StatisticsError:      # tie
        return random.choice(numeric)

boxed_pat  = re.compile(r"\\boxed\{\s*([12])\s*\}")
backup_pat = re.compile(r"\b([12])\b")

def extract_final(text):
    if m := boxed_pat.search(text):
        return m.group(1)
    if m := backup_pat.search(text.strip()):
        return m.group(1)
    return None

def call_llm(server, model, prompts, n, temp, top_p, timeout=120):
    url  = server.rstrip("/") + "/v1/completions"
    body = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 30,
        "temperature": temp,
        "top_p": top_p,
        "n": n,
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    blocks = data if isinstance(data, list) else [data]
    return [[c["text"] for c in blk["choices"]] for blk in blocks]

# ───────── retry / self-consistency ───────────
MAX_RETRY = 8

def query_one(server, model, prompt, n, temp, top_p):
    # ① self-consistency
    replies = call_llm(server, model, [prompt], n, temp, top_p)[0]
    votes   = [extract_final(r) for r in replies]
    choice  = majority_numeric(votes)
    if choice:
        return choice, [v for v in votes if v]

    # ② deterministic 재시도
    for _ in range(MAX_RETRY):
        txt = call_llm(server, model, [prompt], 1, 0.0, 1.0)[0][0]
        v = extract_final(txt)
        if v:
            return v, [v]

    # ③ 완전 실패 → None
    return None, []

# ───────────── main ───────────────────────────
def main(a):
    models_cfg = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))["models"]
    pairs      = load_pairs(a.pairs_jsonl)

    out_lines=[]
    for m in models_cfg:
        name   = m["name"]
        model  = m["hf_id"]
        server = m["server_url"]
        bs     = int(m.get("batch_size", 1))
        n_samp = int(m.get("n_sample", 3))
        temp   = float(m.get("temperature", 0.7))
        top_p  = float(m.get("top_p", 0.9))

        print(f"\n⚙️  {model} | batch={bs} | n_sample={n_samp}")

        for i in tqdm(range(0, len(pairs), bs),
                      total=(len(pairs)+bs-1)//bs,
                      desc=name):
            batch_recs = pairs[i:i+bs]
            prompts    = [render_prompt(r) for r in batch_recs]

            try:
                batch_texts = call_llm(server, model, prompts,
                                       n_samp, temp, top_p)
            except Exception as e:
                print("❗ 배치 호출 실패:", e)
                batch_texts = [[""]*n_samp for _ in prompts]

            for rec, texts, prompt in zip(batch_recs, batch_texts, prompts):
                votes   = [extract_final(t) for t in texts]
                choice  = majority_numeric(votes)

                if choice is None:                  # 전부 빈칸 → 재시도
                    choice, votes = query_one(server, model, prompt,
                                              n_samp, temp, top_p)

                if choice is None:
                    order_mid = None               # 끝까지 실패
                else:
                    order_mid = ([rec["idx_a"], rec["idx_b"]]
                                 if choice == "1"
                                 else [rec["idx_b"], rec["idx_a"]])

                out_lines.append(json.dumps({
                    "ID": rec["ID"],
                    "order_mid": order_mid,         # null 은 후단계에서 처리
                    "votes": [v for v in votes if v],
                    "model": name
                }, ensure_ascii=False))

    out_path = pathlib.Path(a.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print("✅ votes saved →", out_path)

# ───────────── CLI ────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--models_yaml", default="ordering/6_models.yaml")
    ap.add_argument("--out_jsonl",   required=True)
    main(ap.parse_args())
