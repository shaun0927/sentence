#!/usr/bin/env python
"""
6_score_ppl.py
──────────────
・후보 순열 JSONL(5_build_candidates.py 결과)을 불러와
  vLLM 'score' 모드로 PPL을 계산하고 순위화.
・YAML(6_models.yaml)에 적힌 모델들을 차례로 실행할 수 있도록 작성.
───────────────────────────────────────────────────────────────
예시 실행
$ python ordering/6_score_ppl.py \
      --models_yaml ordering/6_models.yaml \
      --test_csv    data/raw/test.csv \
      --cands_jsonl data/proc/test_candidates.jsonl \
      --out_jsonl   data/proc/test_scored.jsonl
"""

import argparse, yaml, json, math, pathlib, itertools, collections
import numpy as np, pandas as pd, tqdm.auto as tq
from vllm import LLM

# ────────────────────────── util ──────────────────────────
def ppl_from_logprobs(log_probs: list[float]) -> float:
    """token log-prob list → perplexity"""
    nll = -sum(log_probs) / max(len(log_probs), 1)
    return math.exp(nll)

def load_test_sentences(csv_path: pathlib.Path) -> dict[str, list[str]]:
    df = pd.read_csv(csv_path)
    return {
        row.ID: [row[f"sentence_{i}"] for i in range(4)]
        for _, row in df.iterrows()
    }

def load_candidates(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

# ────────────────────────── main ───────────────────────────
def main(a: argparse.Namespace) -> None:
    test_sents = load_test_sentences(pathlib.Path(a.test_csv))
    cand_rows  = load_candidates(pathlib.Path(a.cands_jsonl))

    results_all_models = collections.defaultdict(list)  # {ID:[(ppl,order),…]}

    # ── YAML 모델 목록 순회 ───────────────────────────────
    cfg = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))
    for m in cfg["models"]:
        print(f"\n⚙️  Loading {m['hf_id']}  (name={m['name']})")
        llm = LLM(
            model=m["hf_id"],
            task="score",
            dtype=m.get("dtype", "auto"),
            gpu_memory_utilization=m.get("gpu_memory_utilization", 0.9),
            max_model_len=m.get("max_model_len", 4096),
        )
        tokenizer = llm.get_tokenizer()
        bs = m.get("batch_size", 4)

        # ── 후보 시퀀스 → 텍스트 리스트 ───────────────────
        texts, meta = [], []   # meta: (ID, order_list)
        for row in cand_rows:
            sent_list = test_sents[row["ID"]]
            order     = row["order"] if "order" in row else row["rank"]
            text      = " ".join([sent_list[i] for i in order])
            texts.append(text)
            meta.append( (row["ID"], order) )

        # ── 배치별 PPL 계산 ─────────────────────────────
        for b in tq.tqdm(range(0, len(texts), bs)):
            batch_texts = texts[b:b+bs]
            outs = llm.score(batch_texts)
            for out, (ID, order) in zip(outs, meta[b:b+bs]):
                ppl = ppl_from_logprobs(out.logprobs)
                results_all_models[ID].append( (ppl, order) )

    # ── ID마다 최저 PPL 1순위 선택 ───────────────────────
    out_lines = []
    for ID, lst in results_all_models.items():
        best_ppl, best_order = min(lst, key=lambda x: x[0])
        out_lines.append(json.dumps({
            "ID": ID,
            "best_order": best_order,
            "ppl": round(best_ppl, 6),
        }, ensure_ascii=False))

    out_path = pathlib.Path(a.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n✅ saved scored candidates → {out_path.relative_to(pathlib.Path().resolve())}")

# ────────────────────────── CLI ───────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_yaml", required=True)
    ap.add_argument("--test_csv",    required=True)
    ap.add_argument("--cands_jsonl", required=True)
    ap.add_argument("--out_jsonl",   required=True)
    main(ap.parse_args())
