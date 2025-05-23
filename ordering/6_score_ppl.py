#!/usr/bin/env python
"""
6_score_ppl.py  ― completions-echo 버전 (fixed)
───────────────────────────────────────────────
배치 입력을 보내면 vLLM 0.8.x 는 /v1/completions 응답을
• 단일 prompt  →  {"choices":[…]}
• 다중 prompt  →  [ {"choices":[…]}, … ]
두 가지로 나눠 반환합니다. 이 코드는 양쪽 모두 지원합니다.
"""

import argparse, json, math, pathlib, yaml, collections
import requests, pandas as pd, tqdm.auto as tq

# ───────────── util ─────────────
def ppl_from_lp(lp: list[float]) -> float:
    """logprob 리스트 → perplexity (None 값은 건너뜀)"""
    vals = [x for x in lp if x is not None]
    if not vals:                 # 모두 None이면 무한대 취급
        return float("inf")
    nll = -sum(vals) / len(vals)
    return math.exp(nll)

def load_test(csv):
    df = pd.read_csv(csv)
    return {r.ID: [r[f"sentence_{i}"] for i in range(4)] for _, r in df.iterrows()}

def load_jsonl(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]

def completions_score(server: str, model: str, prompts: list[str], timeout=120):
    """Return list[list[logprob]] per prompt using echo+logprobs trick."""
    url  = server.rstrip("/") + "/v1/completions"
    body = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 0,
        "echo": True,
        "logprobs": 0,
        "temperature": 0.0,
    }
    resp = requests.post(url, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"/v1/completions {resp.status_code} → {resp.text[:200]}")

    data = resp.json()
    # 단일 → dict / 복수 → list[dict]
    if isinstance(data, dict):          # 1-prompt
        objs = [data]
    elif isinstance(data, list):        # n-prompt
        objs = data
    else:
        raise RuntimeError(f"Unexpected JSON type: {type(data)}")

    lp_lists = []
    for obj in objs:
        try:
            lp_lists.append(obj["choices"][0]["logprobs"]["token_logprobs"])
        except (KeyError, TypeError):
            raise RuntimeError(f"Missing logprobs in response: {obj}") from None
    return lp_lists

# ───────────── main ─────────────
def main(a):
    test  = load_test(a.test_csv)
    cand  = load_jsonl(a.cands_jsonl)
    cfg   = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))
    if "models" not in cfg:
        raise ValueError("'models' key missing in YAML")

    results = collections.defaultdict(list)

    for m in cfg["models"]:
        model = m["hf_id"]; bs = int(m.get("batch_size", 4))
        print(f"\n⚙️  Scoring with {model} (batch={bs})")

        texts, meta = [], []            # meta = (ID, order)
        for row in cand:
            sents = test[row["ID"]]
            for order in row["candidates"]:
                texts.append(" ".join(sents[i] for i in order))
                meta.append((row["ID"], order))

        for i in tq.tqdm(range(0, len(texts), bs)):
            batch = texts[i:i+bs]
            try:
                lp_lists = completions_score(a.server_url, model, batch)
            except RuntimeError:
                # 서버가 리스트 입력을 아직 지원하지 않을 때 단건 fallback
                lp_lists = [completions_score(a.server_url, model, [t])[0] for t in batch]

            # zip 길이 불일치 방어
            if len(lp_lists) != len(batch):
                print(f"❗ batch={len(batch)}  got={len(lp_lists)} → 일부 prompt drop")

            for (ID, order), lp in zip(meta[i:i+bs], lp_lists):
                results[ID].append((ppl_from_lp(lp), order))

            # 남은 meta(응답 못 받은 것)에 dummy PPL
            for ID, order in meta[i+len(lp_lists): i+bs]:
                results[ID].append((float("inf"), order))

    out_lines = [
        json.dumps({"ID": ID,
                    "best_order": min(lst, key=lambda x: x[0])[1],
                    "ppl": round(min(lst, key=lambda x: x[0])[0], 6)},
                   ensure_ascii=False)
        for ID, lst in results.items()
    ]
    out_path = pathlib.Path(a.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n✅ saved → {out_path}")

# ───────────── CLI ─────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_yaml", required=True)
    ap.add_argument("--test_csv",    required=True)
    ap.add_argument("--cands_jsonl", required=True)
    ap.add_argument("--out_jsonl",   required=True)
    ap.add_argument("--server_url",  required=True)
    main(ap.parse_args())
