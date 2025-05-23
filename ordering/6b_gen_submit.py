#!/usr/bin/env python
"""
6c_gen_submit_strict.py
───────────────────────
• 입력
  ├─ test.csv (4 문장)
  └─ *_first_top2.jsonl (첫 문장 index)
• 과정
  ① 엄격한 프롬프트 + 예시 + system 지시
  ② T=0.3 / top_p=0.7 / n=32  self-consistency
  ③ 파싱·수정·다수결  ⇒ 최종 순서
• 출력
  submission.csv

로그로 파싱 성공률 · fallback 빈도를 콘솔에 출력한다.
"""

import argparse, json, re, collections, pathlib, random, pandas as pd, requests
import tqdm.auto as tq

# ───────────────────────────────────────────────────────────────────────
IDX_RE   = re.compile(r"^[0-3](?:\s+[0-3]){2}$")      # "3 1 2"
CLEAN_RE = re.compile(r"[^\d\s]")                     # 숫자/공백 외 제거

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are an expert Korean writer. "
        "Return **ONLY** three distinct integers (0-3) "
        "representing the natural order of the remaining sentences. "
        "Format: `<a> <b> <c>` (single spaces). "
        "No other text."
    ),
}

# few-shot 예시 3개
FEW_SHOT = [
    {
        "role": "user",
        "content":
            "다음은 한 문단을 이루는 4개의 한국어 문장입니다.\n"
            "문장 [2] 는 이미 첫 문장으로 확정되었습니다.\n"
            "남은 문장을 자연스러운 순서로 재배열하여 인덱스만 공백으로 구분해 주세요.\n\n"
            "[2] 커피를 내리는 시간은 약 4분이 소요된다.\n"
            "[0] 먼저 커피 원두를 분쇄한다.\n"
            "[1] 그런 다음 필터에 원두를 담는다.\n"
            "[3] 마지막으로 물을 부어 추출한다.\n\n"
            "Answer:"
    },
    {"role": "assistant", "content": "0 1 3"},
    {
        "role": "user",
        "content":
            "문장 [0] 는 이미 첫 문장으로 확정되었습니다.\n"
            "[0] 봄이 오면 벚꽃이 만발한다.\n"
            "[1] 사람들은 꽃구경을 위해 공원을 찾는다.\n"
            "[2] 사진을 찍으며 즐거워한다.\n"
            "[3] 저녁이 되면 조명이 켜져 더욱 아름답다.\n\n"
            "Answer:"
    },
    {"role": "assistant", "content": "1 2 3"},
    {
        "role": "user",
        "content":
            "문장 [3] 는 이미 첫 문장으로 확정되었습니다.\n"
            "[3] 스마트폰은 현대인의 필수품이다.\n"
            "[0] 통화뿐만 아니라 인터넷도 사용할 수 있다.\n"
            "[1] 다양한 앱으로 업무 효율을 높인다.\n"
            "[2] 그러나 과도한 사용은 건강에 해롭다.\n\n"
            "Answer:"
    },
    {"role": "assistant", "content": "0 1 2"},
]

# ───────────────────────────────────────────────────────────────────────
def load_first(path):
    """jsonl → {ID: first_idx}"""
    return {j["ID"]: j["rank"][0] for j in map(json.loads, pathlib.Path(path).read_text().splitlines())}


def build_prompt(first_idx: int, sents: list[str]) -> str:
    remain = [f"[{i}] {s}" for i, s in enumerate(sents) if i != first_idx]
    return (
        "다음은 한 문단을 이루는 4개의 한국어 문장입니다.\n"
        f"문장 [{first_idx}] 는 이미 첫 문장으로 확정되었습니다.\n"
        "남은 세 문장을 자연스러운 순서로 재배열하여 **숫자 세 개만** 공백으로 구분해 주세요.\n\n"
        f"[{first_idx}] {sents[first_idx]}\n" + "\n".join(remain) + "\n\nAnswer:"
    )


def query_chat(url, model, messages, n, temp=0.3, top_p=0.7):
    body = {
        "model": model,
        "messages": messages,
        "n": n,
        "temperature": temp,
        "top_p": top_p,
        "max_tokens": 6,
    }
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=180)
    r.raise_for_status()
    return [c["message"]["content"].strip() for c in r.json()["choices"]]


def parse_or_repair(txt: str):
    """clean → parse; attempt repair if invalid"""
    txt = CLEAN_RE.sub(" ", txt).strip()
    txt = re.sub(r"\s+", " ", txt)
    if IDX_RE.fullmatch(txt):
        idx = list(map(int, txt.split()))
    else:
        # ✧ repair : 추출 가능한 숫자 세 개면 사용
        nums = [int(t) for t in txt.split() if t.isdigit() and 0 <= int(t) <= 3]
        if len(set(nums)) == 3:
            idx = nums[:3]
        else:
            return None
    return idx


def majority(seq_list):
    cnt = collections.Counter(map(tuple, seq_list))
    return list(cnt.most_common(1)[0][0])


# ───────────────────────────────────────────────────────────────────────
def main(args):
    random.seed(42)
    first_map = load_first(args.first_jsonl)
    df = pd.read_csv(args.test_csv)

    good, repaired, fail = 0, 0, 0
    submission = []

    for _, row in tq.tqdm(df.iterrows(), total=len(df)):
        ID = row["ID"]
        sents = [row[f"sentence_{i}"] for i in range(4)]
        first_idx = first_map[ID]

        prompt = build_prompt(first_idx, sents)
        messages = [SYSTEM_MSG, *FEW_SHOT, {"role": "user", "content": prompt}]

        gens = query_chat(args.server_url, args.model, messages, args.n_sample)
        parsed = []
        for g in gens:
            p = parse_or_repair(g)
            if p is None:
                fail += 1
            elif IDX_RE.fullmatch(" ".join(map(str, p))):
                if g.strip() != " ".join(map(str, p)):
                    repaired += 1
                parsed.append(p)
                good += 1
        if not parsed:
            # fallback: 남은 인덱스를 original 순서
            final = [i for i in range(4) if i != first_idx]
        else:
            final = majority(parsed)

        full = [first_idx] + final
        submission.append({
            "ID": ID,
            "answer_0": full[0],
            "answer_1": full[1],
            "answer_2": full[2],
            "answer_3": full[3],
        })

    # ─── 결과 저장 ────────────────────────────────────────────────────
    out_p = pathlib.Path(args.out_csv)
    pd.DataFrame(submission).to_csv(out_p, index=False)

    total = good + repaired + fail
    print(
        "\n📈 Parsing stats:"
        f"\n   valid   : {good}  ({good/total:.1%})"
        f"\n   repaired: {repaired}  ({repaired/total:.1%})"
        f"\n   fail    : {fail}  ({fail/total:.1%})"
    )
    print("✅ submission saved →", out_p.resolve())


# ─── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv",    required=True, help="data/raw/test.csv")
    p.add_argument("--first_jsonl", required=True, help="*_first_top2.jsonl")
    p.add_argument("--server_url",  required=True, help="http://localhost:8000")
    p.add_argument("--model",       required=True, help="moreh/Llama-3-Motif-102B")
    p.add_argument("--n_sample",    type=int, default=32)
    p.add_argument("--out_csv",     default="submission.csv")
    main(p.parse_args())
