#!/usr/bin/env python
# ordering/scripts/2_query_llm_b.py
# ------------------------------------------------------------------
# pairs.jsonl + 6_models.yaml → votes.jsonl  (order_all 길이 4)

import argparse, json, pathlib, re, statistics, random, itertools, yaml, requests
from tqdm.auto import tqdm

"""
python ordering/scripts/2_query_llm_b.py --pairs_jsonl ordering/data/pairs/test_all4_pairs.jsonl --out_jsonl  ordering/data/votes/test_all4_votes.jsonl
"""

# ───── 프롬프트 템플릿 ──────────────────────────────────────────────
PROMPT_TEMPLATE = """\
다음 네 문장은 하나의 글을 이룹니다. 문장 A, B, C, D 네 개를
자연스러운 순서로 배열하세요.

문장 A: "{sent_a}"
문장 B: "{sent_b}"
문장 C: "{sent_c}"
문장 D: "{sent_d}"

가능한 배열은 아래 24가지입니다.
1) A → B → C → D
2) A → B → D → C
3) A → C → B → D
4) A → C → D → B
5) A → D → B → C
6) A → D → C → B
7) B → A → C → D
8) B → A → D → C
9) B → C → A → D
10) B → C → D → A
11) B → D → A → C
12) B → D → C → A
13) C → A → B → D
14) C → A → D → B
15) C → B → A → D
16) C → B → D → A
17) C → D → A → B
18) C → D → B → A
19) D → A → B → C
20) D → A → C → B
21) D → B → A → C
22) D → B → C → A
23) D → C → A → B
24) D → C → B → A

정답을 \\boxed{{{{번호}}}} 한 줄로만 출력하세요.
첫 토큰은 반드시 \\boxed{{ 로 시작해야 합니다. 
(가능한 정답은 정수이며, 1~24 범위)

예) A → B → D → C 이면 \\boxed{{{{2}}}}
"""

# ───── util ────────────────────────────────────────────────────────
def load_pairs(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text("utf-8").splitlines()]

def render_prompt(r):              # str.format 삽입
    return PROMPT_TEMPLATE.format(
        sent_a=r["sent_a"], sent_b=r["sent_b"],
        sent_c=r["sent_c"], sent_d=r["sent_d"]
    )

boxed_pat  = re.compile(r"\\boxed\{\s*([1-9]|1[0-9]|2[0-4])\s*\}")
backup_pat = re.compile(r"\b([1-9]|1[0-9]|2[0-4])\b")

def extract_num(txt):
    if m := boxed_pat.search(txt):
        return m.group(1)
    if m := backup_pat.search(txt.strip()):
        return m.group(1)
    return None

def majority(votes):
    nums = [v for v in votes if v]
    if not nums:
        return None
    try:
        return statistics.mode(nums)
    except statistics.StatisticsError:
        return random.choice(nums)

# 번호 → 인덱스 배열 매핑
PERMS = list(itertools.permutations("ABCD"))
ORDER_MAP = {str(i+1): p for i, p in enumerate(PERMS)}   # '1'~'24'

def order_from_choice(choice, rec):
    letter2idx = {"A": rec["idx_a"], "B": rec["idx_b"],
                  "C": rec["idx_c"], "D": rec["idx_d"]}
    return [letter2idx[ch] for ch in ORDER_MAP[choice]]

# LLM 호출
def call_llm(server, model, prompts, n, temp, top_p, timeout=120):
    url  = server.rstrip("/") + "/v1/completions"
    body = {"model": model, "prompt": prompts if len(prompts)>1 else prompts[0],
            "max_tokens": 40, "temperature": temp, "top_p": top_p, "n": n}
    return requests.post(url, json=body, timeout=timeout).json()

def texts_from_resp(resp):
    blocks = resp if isinstance(resp, list) else [resp]
    return [[c["text"] for c in blk["choices"]] for blk in blocks]

# ───── main ────────────────────────────────────────────────────────
def main(a):
    models = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))["models"]
    pairs  = load_pairs(a.pairs_jsonl)

    out=[]
    for m in models:
        name, model, server = m["name"], m["hf_id"], m["server_url"]
        bs = int(m.get("batch_size",1)); n = int(m.get("n_sample",5))
        temp=float(m.get("temperature",0.7)); top_p=float(m.get("top_p",0.9))

        print(f"\n⚙️ {model} | batch={bs} | n_sample={n}")
        for i in tqdm(range(0,len(pairs),bs),
                      total=(len(pairs)+bs-1)//bs, desc=name):
            chunk = pairs[i:i+bs]
            prompts=[render_prompt(r) for r in chunk]
            try:
                resp = call_llm(server,model,prompts,n,temp,top_p)
                batch_txt = texts_from_resp(resp)
            except Exception as e:
                print("❗ 호출 실패:",e); batch_txt=[[""]*n for _ in prompts]

            for rec, txts in zip(chunk, batch_txt):
                votes=[extract_num(t) for t in txts]
                choice=majority(votes)
                order_all=order_from_choice(choice,rec) if choice else None

                out.append(json.dumps({
                    "ID":rec["ID"],
                    "order_all":order_all,   # [idx0,idx1,idx2,idx3] 또는 null
                    "votes":[v for v in votes if v],
                    "model":name
                }, ensure_ascii=False))

    p=pathlib.Path(a.out_jsonl)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text("\n".join(out),encoding="utf-8")
    print("✅ votes saved →",p)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl",required=True)
    ap.add_argument("--models_yaml",default="ordering/6_models.yaml")
    ap.add_argument("--out_jsonl",required=True)
    main(ap.parse_args())
