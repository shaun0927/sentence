#!/usr/bin/env python
"""
6c_gen_submit_no_first.py
─────────────────────────
· raw test.csv(4 문장)만 사용
· self-consistency(다수결)로 문장 4개의 ‘완전 순서’ 예측
· submission.csv 생성
"""

import argparse, json, re, collections, pathlib, pandas as pd, requests, tqdm.auto as tq

# “0 3 1 2” 처럼 0-3 사이 숫자 4개 공백 구분
ORDER_PAT = re.compile(r"^[0-3](?: [0-3]){3}$")

def build_prompt(sentences):
    body = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return (
        "다음은 한 단락을 이루는 4개의 한국어 문장입니다.\n"
        "가장 자연스러운 문단이 되도록 문장의 순서를 재배열하십시오.\n"
        "아래 문장 인덱스(0~3) 네 개를 공백으로 구분해 한 줄로만 답하세요.\n"
        "예시) 2 0 1 3\n\n"
        + body + "\n\nAnswer:"
    )

def query_llm(url, model, prompt, n):
    body = {
        "model": model,
        "messages":[{"role":"user","content":prompt}],
        "temperature":0.7, "top_p":0.9,
        "n": n, "max_tokens":8
    }
    r = requests.post(url.rstrip("/")+"/v1/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    return [c["message"]["content"].strip() for c in r.json()["choices"]]

def parse_order(txt):
    if not ORDER_PAT.match(txt): 
        return None
    idx = list(map(int, txt.split()))
    return idx if len(set(idx))==4 else None

def majority_vote(orders):
    cnt = collections.Counter(tuple(o) for o in orders)
    return list(cnt.most_common(1)[0][0])

# ────────────────────────
def main(a):
    df = pd.read_csv(a.test_csv)
    rows_csv = []

    for _, row in tq.tqdm(df.iterrows(), total=len(df)):
        ID = row["ID"]
        sents = [row[f"sentence_{i}"] for i in range(4)]
        prompt = build_prompt(sents)

        gens = query_llm(a.server_url, a.model, prompt, a.n_sample)
        parsed = [p for g in gens if (p:=parse_order(g)) is not None]

        if not parsed:                       # 전부 파싱 실패 → 0-3 기본
            order = [0,1,2,3]
        else:
            order = majority_vote(parsed)

        rows_csv.append({
            "ID":ID,
            "answer_0":order[0],
            "answer_1":order[1],
            "answer_2":order[2],
            "answer_3":order[3],
        })

    out = pd.DataFrame(rows_csv)
    out.to_csv(a.out_csv, index=False)
    print("✅ submission saved →", a.out_csv)

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--test_csv",   required=True, help="data/raw/test.csv")
    P.add_argument("--server_url", required=True, help="http://localhost:8000")
    P.add_argument("--model",      required=True, help="moreh/Llama-3-Motif-102B")
    P.add_argument("--n_sample",   type=int, default=16, help="self-consistency N")
    P.add_argument("--out_csv",    default="submission_no_first.csv")
    main(P.parse_args())
