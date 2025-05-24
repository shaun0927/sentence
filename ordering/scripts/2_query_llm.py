#!/usr/bin/env python
# ordering/scripts/2_query_llm.py  (self-contained prompt)
# ------------------------------------------------------------------
# 두 문장 쌍 JSONL + ordering/6_models.yaml → LLM 질의 → 다수결 결과 저장

import argparse, json, pathlib, re, statistics, yaml, requests

# ───────────── 내장 프롬프트 템플릿 ─────────────
PROMPT_TEMPLATE = """\
첫 문장: "{first_sent}"
마지막 문장: "{last_sent}"

문장 A: "{sent_a}"
문장 B: "{sent_b}"

자연스러운 순서는?
1) A → B
2) B → A
정답 번호만 숫자로 출력:"""

# ───────────── helpers ────────────────────────
def load_pairs(path):
    txt = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in txt]

def render_prompt(rec):
    return PROMPT_TEMPLATE.format(first_sent = rec["first_sent"],
                                  last_sent  = rec["last_sent"],
                                  sent_a = rec["sent_a"],
                                  sent_b = rec["sent_b"])

def majority(votes):
    try:
        return statistics.mode(votes)
    except statistics.StatisticsError:
        return votes[0] if votes else "1"

def call_llm(server, model, prompts, n, temp, top_p, timeout=120):
    url  = server.rstrip("/") + "/v1/completions"
    body = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 5,
        "temperature": temp,
        "top_p": top_p,
        "n": n,
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    blocks = data if isinstance(data, list) else [data]
    return [[c["text"] for c in blk["choices"]] for blk in blocks]

def parse_vote(txt):
    m = re.search(r"[12]", txt)
    return m.group(0) if m else None

# ───────────── main ───────────────────────────
def main(a):
    models_cfg = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))["models"]
    pairs      = load_pairs(a.pairs_jsonl)

    out_lines=[]
    for m in models_cfg:
        model  = m["hf_id"]
        server = m["server_url"]
        bs     = int(m.get("batch_size", 4))
        n_samp = int(m.get("n_sample", 1))
        temp   = float(m.get("temperature", 0.0))
        top_p  = float(m.get("top_p", 1.0))

        print(f"\n⚙️  {model} | batch={bs} | n_sample={n_samp}")

        for i in range(0, len(pairs), bs):
            batch_recs = pairs[i:i+bs]
            prompts = [render_prompt(r) for r in batch_recs]

            try:
                replies = call_llm(server, model, prompts,
                                   n_samp, temp, top_p)
            except Exception as e:
                print("❗ LLM 요청 실패, fallback 1회:", e)
                replies = [["1"]*n_samp for _ in prompts]

            for rec, texts in zip(batch_recs, replies):
                votes = [parse_vote(t) for t in texts if parse_vote(t)]
                choice = majority(votes)
                order_mid = ([rec["idx_a"], rec["idx_b"]] if choice == "1"
                             else [rec["idx_b"], rec["idx_a"]])
                out_lines.append(json.dumps({
                    "ID": rec["ID"],
                    "order_mid": order_mid,
                    "votes": votes,
                    "model": m["name"]
                }, ensure_ascii=False))

    out = pathlib.Path(a.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print("✅ votes saved →", out)

# ───────────── CLI ────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_jsonl", required=True)
    p.add_argument("--models_yaml", default="ordering/6_models.yaml")
    p.add_argument("--out_jsonl",   required=True)
    main(p.parse_args())
