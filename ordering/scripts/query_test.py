#!/usr/bin/env python
# --------------------------------------------------------------------
# ordering/scripts/query_test.py
# --------------------------------------------------------------------
# • 4-문장 정렬(Self-Consistency) 실험용 드라이버
# • 특정 ID 집합 / 무작위 샘플 / 실시간 echo 옵션 포함
# --------------------------------------------------------------------
# 사용 예시
#   python ordering/scripts/query_test.py --pairs_jsonl ordering/data/pairs/test_all4_pairs.jsonl --out_jsonl ordering/data/votes/debug10.jsonl --ids TEST_0000,TEST_0001,TEST_0002,TEST_0003,TEST_0004,TEST_0005,TEST_0006,TEST_0007,TEST_0008,TEST_0009 --echo
#
#   python ordering/scripts/query_test.py \
#       --pairs_jsonl ordering/data/pairs/test_all4_pairs.jsonl \
#       --out_jsonl   ordering/data/votes/sample10.jsonl \
#       --sample_n 10 --echo
# --------------------------------------------------------------------
import argparse
import json
import math
import pathlib
import random
import re
import statistics
import sys
import itertools

from collections import Counter

import requests
import yaml
from tqdm.auto import tqdm

# ────────────────────── 정규식 / 상수 ─────────────────────────
PERM_4 = [''.join(p) for p in itertools.permutations("ABCD")]
perm_pat = re.compile(r"\b([ABCD]{4})\b")

# 베이스 프롬프트 템플릿
BASE_TEMPLATE = """\
아래 네 문장은 하나의 글을 이룹니다. 문장 {L1},{L2},{L3},{L4} 를
가장 자연스러운 순서(앞→뒤)로 배열하세요.

문장 {L1}: "{S1}"
문장 {L2}: "{S2}"
문장 {L3}: "{S3}"
문장 {L4}: "{S4}"

정답을 {L1}{L2}{L3}{L4} • {L1}{L2}{L4}{L3} … 과 같이
{L1}{L2}{L3}{L4},{L1}{L2}{L4}{L3}, … ,{L4}{L3}{L2}{L1}
***4글자 문자열*** 로 한 줄만 출력하세요.
(예: {L2}{L1}{L3}{L4})
"""

# ────────────────────── 헬퍼 함수 ────────────────────────────
def load_pairs(path: str):
    """JSONL → list[dict]"""
    txt = pathlib.Path(path).read_text("utf-8").splitlines()
    return [json.loads(l) for l in txt]


def build_prompt(rec: dict, lbl_order):
    """
    lbl_order: e.g. ['C','A','D','B']  (무작위 라벨 순서)
    반환: (prompt_text, lbl_map)
      * lbl_map : 무작위 라벨→고정라벨(A/B/C/D) 매핑(dict)
    """
    lbl_map = dict(zip(lbl_order, "ABCD"))              # 실제 → 고정
    sents = {l: rec[f"sent_{lbl_map[l].lower()}"] for l in lbl_order}

    ptxt = BASE_TEMPLATE.format(
        L1=lbl_order[0], L2=lbl_order[1],
        L3=lbl_order[2], L4=lbl_order[3],
        S1=sents[lbl_order[0]],
        S2=sents[lbl_order[1]],
        S3=sents[lbl_order[2]],
        S4=sents[lbl_order[3]],
    )
    return ptxt, lbl_map


def extract_ans(txt: str):
    """LLM 출력에서 ‘ABCD’ 패턴 추출 → 유효하면 반환"""
    m = perm_pat.search(txt.strip())
    if not m:
        return None
    seq = m.group(1)
    return seq if seq in PERM_4 else None


def majority(votes):
    """
    간단한 최빈값(다중동률 시 random choice)
    votes: list[hashable] (None 제외)
    """
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    try:
        return statistics.mode(valid)
    except statistics.StatisticsError:          # tie
        return random.choice(valid)


def call_llm(server_url: str, model_name: str,
             prompts, n, temp, top_p, timeout=120):
    """
    OpenAI-compatible 서버 / vLLM 서버 등 호출
    prompts: list[str]  또는 단일 str
    반환: list[list[str]]  (batch × n responses)
    """
    body = {
        "model": model_name,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 12,
        "temperature": temp,
        "top_p": top_p,
        "n": n,
    }
    url = server_url.rstrip("/") + "/v1/completions"
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    blocks = data if isinstance(data, list) else [data]
    # blocks[i]["choices"][j]["text"]
    return [[c["text"] for c in blk["choices"]] for blk in blocks]


def order_from_str(order_str, lbl_map, rec):
    """
    order_str : 'CADB' (무작위 view 라벨)
    lbl_map   : 무작위→기준 매핑
    rec       : pair record (idx_a …)
    반환: [문장 인덱스]  ex) [1,0,2,3]
    """
    letter2idx = {"A": rec["idx_a"], "B": rec["idx_b"],
                  "C": rec["idx_c"], "D": rec["idx_d"]}
    canonical = ''.join(lbl_map[ch] for ch in order_str)   # 고정 라벨열
    return [letter2idx[ch] for ch in canonical]

# ────────────────────── 메인 루틴 ───────────────────────────
def main(a):
    # ---------- 1) 모델 설정  ----------
    models_cfg = yaml.safe_load(open(a.models_yaml, encoding="utf-8"))["models"]

    # ---------- 2) pairs 로드 ----------
    pairs = load_pairs(a.pairs_jsonl)

    # ---------- 3) 테스트용 샘플링 ----------
    if a.ids:                                      # ID 리스트 지정
        keep = set(a.ids.split(','))
        pairs = [p for p in pairs if p["ID"] in keep]
        if not pairs:
            print("⚠️  지정한 ID가 데이터에 없습니다.", file=sys.stderr)
            return
    elif a.sample_n:
        random.seed(42)                            # 재현성
        pairs = random.sample(pairs, min(a.sample_n, len(pairs)))

    # ---------- 4) 결과 수집 ----------
    out_lines = []
    for m in models_cfg:
        name   = m["name"]
        model  = m["hf_id"]
        server = m["server_url"]

        bs      = int(m.get("batch_size", 1))
        n_sample= int(m.get("n_sample", 32))
        n_view  = int(m.get("n_view",   5))
        temp    = float(m.get("temperature", 0.8))
        top_p   = float(m.get("top_p", 0.9))

        print(f"\n⚙️ {model} | pairs={len(pairs)} | "
              f"view={n_view} × n={n_sample} | batch={bs}",
              file=sys.stderr)

        # ---- pair batch ----
        for i in tqdm(range(0, len(pairs), bs),
                      total=math.ceil(len(pairs)/bs),
                      desc=name, file=sys.stderr):
            batch_recs = pairs[i:i+bs]

            # ---- view 별 프롬프트 ----
            prompts, view_maps = [], []            # 길이 = bs * n_view
            for rec in batch_recs:
                for _ in range(n_view):
                    lbl_order = random.sample("ABCD", 4)
                    ptxt, mp = build_prompt(rec, lbl_order)
                    prompts.append(ptxt)
                    view_maps.append((rec, mp))

            # ---- LLM 호출 ----
            resp_blocks = call_llm(server, model, prompts,
                                   n_sample, temp, top_p)
            flat_resp = [txt
                         for blk in resp_blocks
                         for txt in blk]           # 길이 = prompts * n_sample

            # ---- 결과 해석 ----
            ptr = 0
            for rec in batch_recs:
                view_votes = []                    # list[list[int]]
                for _ in range(n_view):
                    answers = flat_resp[ptr: ptr + n_sample]
                    rec_, mp = view_maps[ptr // n_sample]
                    ptr += n_sample
                    for txt in answers:
                        seq = extract_ans(txt)
                        if seq:
                            view_votes.append(order_from_str(seq, mp, rec_))

                # Canonical 문자열로 majority
                vote_strs = ['-'.join(map(str, v)) for v in view_votes]
                best_str  = majority(vote_strs)
                order_all = list(map(int, best_str.split('-'))) if best_str else None

                rec_json = {
                    "ID": rec["ID"],
                    "order_all": order_all,
                    "votes": vote_strs,
                    "model": name,
                }
                line = json.dumps(rec_json, ensure_ascii=False)
                out_lines.append(line)
                if a.echo:
                    print(line)

    # ---------- 5) 저장 ----------
    out_path = pathlib.Path(a.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out_lines), encoding="utf-8")
    print(f"✅ 결과 저장 → {out_path}", file=sys.stderr)

# ────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True,
                    help="입력 pairs JSONL 파일")
    ap.add_argument("--models_yaml", default="ordering/6_models.yaml",
                    help="모델 리스트 YAML")
    ap.add_argument("--out_jsonl",   required=True,
                    help="저장할 votes JSONL")

    # --- 실험용 옵션 ---
    ap.add_argument("--ids", help="콤마로 구분한 특정 ID 리스트")
    ap.add_argument("--sample_n", type=int,
                    help="무작위 N개 pairs 샘플링 (ids 옵션보다 우선순위 낮음)")
    ap.add_argument("--echo", action="store_true",
                    help="한 줄 결과를 STDOUT 에도 실시간 출력")

    main(ap.parse_args())
