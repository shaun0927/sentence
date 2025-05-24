#!/usr/bin/env python
# ordering/scripts/0_choose_first_llm.py
# ------------------------------------------------------------------
# test_first_top2.jsonl + test.csv → first_only.jsonl({ID,first})

"""
python ordering/scripts/0_choose_first_llm.py --test_csv data/raw/test.csv --top2_jsonl data/proc/test_first_top2.jsonl --out_jsonl data/proc/first_only.jsonl
"""

import argparse, json, pathlib, re, statistics, random, yaml, requests, pandas as pd
from tqdm.auto import tqdm

# ───── 프롬프트 ────────────────────────────────────────────────────
PROMPT = """\
다음 두 문장 중 글의 **첫 문장**(도입부)으로 더 자연스러운 쪽을 고르세요.

문장 A: "{sent_a}"
문장 B: "{sent_b}"

1) 문장 A가 첫 문장
2) 문장 B가 첫 문장

정답을 \\boxed{{{{번호}}}} 형태로 한 줄로만 출력하세요.
예) \\boxed{{{{1}}}}
"""

# ───── util --------------------------------------------------------
boxed_pat = re.compile(r"\\boxed\{\s*([12])\s*\}")
def parse(txt):
    m = boxed_pat.search(txt) or re.search(r"\b([12])\b", txt.strip())
    return m.group(1) if m else None

def majority(votes):
    vs = [v for v in votes if v]
    if not vs:
        return None
    try:
        return statistics.mode(vs)
    except statistics.StatisticsError:
        return random.choice(vs)

def call_llm(url, model, prompts, n, temp, top_p):
    body={"model":model,"prompt":prompts if len(prompts)>1 else prompts[0],
          "max_tokens":20,"temperature":temp,"top_p":top_p,"n":n}
    r=requests.post(url.rstrip("/")+"/v1/completions",json=body,timeout=120)
    r.raise_for_status()
    blk=r.json(); blk=[blk] if isinstance(blk,dict) else blk
    return [[c["text"] for c in b["choices"]] for b in blk]

# ───── main --------------------------------------------------------
def main(a):
    # 0) 데이터 읽기 -------------------------------------------------
    df = pd.read_csv(a.test_csv, encoding="utf-8-sig")
    sent_pool = {row.ID:[row[f"sentence_{i}"] for i in range(4)]
                 for _,row in df.iterrows()}

    top2 = [json.loads(l) for l in open(a.top2_jsonl,encoding="utf-8")]
    models = yaml.safe_load(open(a.models_yaml,encoding="utf-8"))["models"]

    results = {}
    for m in models:
        name,model,server = m["name"],m["hf_id"],m["server_url"]
        bs,n,temp,top_p = 1,int(m.get("n_sample",5)),float(m.get("temperature",0.7)),float(m.get("top_p",0.9))
        print(f"\n⚙️  {model} batch={bs} n={n}")

        # -- 모든 ID 한번에 처리 (batch 1) ---------------------------
        for obj in tqdm(top2, desc=name):
            rid = obj["ID"]; i,j = obj["rank"]           # 두 후보 idx
            sents = sent_pool[rid]
            prompt=PROMPT.format(sent_a=sents[i], sent_b=sents[j])

            txts = call_llm(server,model,[prompt],n,temp,top_p)[0]
            votes=[parse(t) for t in txts]; choice=majority(votes)

            # numeric → 실제 인덱스
            first_idx = i if choice=="1" else j
            results[rid]=(first_idx, votes)

    # 1) JSONL 저장 --------------------------------------------------
    out_lines=[json.dumps({"ID":rid,"first":idx,"votes":v},
                          ensure_ascii=False)
               for rid,(idx,v) in results.items()]
    p=pathlib.Path(a.out_jsonl); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("\n".join(out_lines),encoding="utf-8")
    print("✅ first_only saved →",p)

# ───── CLI ---------------------------------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--test_csv",    required=True)
    ap.add_argument("--top2_jsonl",  required=True,
                   help="data/proc/test_first_top2.jsonl")
    ap.add_argument("--models_yaml", default="ordering/6_models.yaml")
    ap.add_argument("--out_jsonl",   required=True,
                   help="data/proc/first_only.jsonl")
    main(ap.parse_args())
