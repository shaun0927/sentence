#!/usr/bin/env python
# ordering/scripts/2_query_llm_b.py
# ------------------------------------------------------------------
# 4 문장 전순열(24) → LLM self-consistency · 편향 최소화 버전
#
# 사용 예
# python ordering/scripts/2_query_llm_b.py --pairs_jsonl ordering/data/pairs/test_all4_pairs.jsonl --out_jsonl ordering/data/votes/test_all4_votes.jsonl
# ------------------------------------------------------------------
import argparse, json, pathlib, random, re, statistics, itertools
import yaml, requests
from collections import defaultdict
from tqdm.auto import tqdm

# ────────────────────────────── 프롬프트 ────────────────────────────────
BASE_TEMPLATE = """\
아래 네 문장은 하나의 글을 구성합니다. {L1},{L2},{L3},{L4} 를
읽는 사람이 가장 매끄럽게 이해할 수 있는 **앞→뒤** 순서로 재배열하세요.

{L1}: "{S1}"
{L2}: "{S2}"
{L3}: "{S3}"
{L4}: "{S4}"

정답은 네 글자의 문자열 한 줄로만 적어주세요.
예) {L2}{L1}{L3}{L4}
"""

PERM_4     = {''.join(p) for p in itertools.permutations("ABCD")}
perm_regex = re.compile(r"\b([ABCD]{4})\b")

# ────────────────────────── 헬퍼 함수들 ────────────────────────────────
def load_pairs(path: str):
    txt = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in txt]

def build_prompt(rec, lbl_order):
    """
    lbl_order 예: ['C','A','D','B']
    반환: prompt, lbl_map(랜덤→정형 'ABCD')
    """
    lbl_map = {rnd: canon for rnd, canon in zip(lbl_order, "ABCD")}
    prompt  = BASE_TEMPLATE.format(
        L1=lbl_order[0], L2=lbl_order[1], L3=lbl_order[2], L4=lbl_order[3],
        S1=rec[f"sent_{lbl_map[lbl_order[0]].lower()}"],
        S2=rec[f"sent_{lbl_map[lbl_order[1]].lower()}"],
        S3=rec[f"sent_{lbl_map[lbl_order[2]].lower()}"],
        S4=rec[f"sent_{lbl_map[lbl_order[3]].lower()}"],
    )
    return prompt, lbl_map

def extract_perm(text: str):
    """4-letter 순열을 찾아 PERM_4 안에 있으면 반환"""
    m = perm_regex.search(text.strip())
    return m.group(1) if m and m.group(1) in PERM_4 else None

def majority(lst):
    """동률이면 무작위"""
    if not lst:
        return None
    try:
        return statistics.mode(lst)
    except statistics.StatisticsError:
        return random.choice(lst)

def call_llm(server, model, prompts, n, temp, top_p, timeout=120):
    """vLLM /v1/completions 호출"""
    url  = server.rstrip('/') + '/v1/completions'
    body = {
        "model": model,
        "prompt": prompts if len(prompts) > 1 else prompts[0],
        "max_tokens": 20,                 # ### NEW: 여유 확보
        "temperature": temp,
        "top_p": top_p,
        "n": n
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    blocks = data if isinstance(data, list) else [data]
    return [[c["text"] for c in blk["choices"]] for blk in blocks]

def order_from_perm(perm_str, lbl_map, rec):
    """LLM 출력 perm_str → 원본 인덱스 리스트"""
    letter2idx = {"A": rec["idx_a"], "B": rec["idx_b"],
                  "C": rec["idx_c"], "D": rec["idx_d"]}
    canonical  = ''.join(lbl_map[ch] for ch in perm_str)  # 무작위→정형
    return [letter2idx[ch] for ch in canonical]

# ──────────────────────────── 메인 루프 ────────────────────────────────
def main(args):
    # 데이터 로드
    pairs  = load_pairs(args.pairs_jsonl)
    models = yaml.safe_load(open(args.models_yaml, encoding="utf-8"))["models"]

    out_lines = []

    for m in models:
        name      = m["name"]
        model     = m["hf_id"]
        server    = m["server_url"]
        bs        = int(m.get("batch_size", 1))
        n_sample  = int(m.get("n_sample", 8))   # ### NEW: 뷰·샘플 균형
        n_view    = int(m.get("n_view",   8))   # ### NEW: view ↑
        temp      = float(m.get("temperature", 0.8))
        top_p     = float(m.get("top_p", 0.95))

        print(f"\n⚙️ {model} | batch={bs} | n_sample={n_sample} | view={n_view}")

        for i in tqdm(range(0, len(pairs), bs),
                      total=(len(pairs)+bs-1)//bs,
                      desc=name):
            batch_recs = pairs[i:i+bs]

            # ------ 프롬프트 생성 (rec × view) ----------------------
            prompts       = []
            prompt_infos  = []   # [(rec obj, lbl_map)]  순서 일치
            for rec in batch_recs:
                for _ in range(n_view):
                    lbl_order = random.sample("ABCD", 4)
                    ptxt, mp  = build_prompt(rec, lbl_order)
                    prompts.append(ptxt)
                    prompt_infos.append((rec, mp))      ### NEW: 정확 매핑

            # ------ LLM 호출 ---------------------------------------
            replies_nested = call_llm(server, model, prompts,
                                      n_sample, temp, top_p)
            # 납작 리스트 (prompt 순서 유지)
            flat_reps = [txt for block in replies_nested for txt in block]

            # ------ 결과 집계 ---------------------------------------
            votes_per_id = defaultdict(list)

            for pi, txt in enumerate(flat_reps):
                rec, mp = prompt_infos[pi // n_sample]   # ### FIX: 인덱스 정확
                perm    = extract_perm(txt)
                if perm:
                    order = order_from_perm(perm, mp, rec)
                    votes_per_id[rec["ID"]].append('-'.join(map(str, order)))

            # ---- batch_recs 순서대로 저장 ----
            for rec in batch_recs:
                id_   = rec["ID"]
                votes = votes_per_id[id_]
                choice = majority(votes)
                order_all = list(map(int, choice.split('-'))) if choice else None

                out_lines.append(json.dumps({
                    "ID": id_,
                    "order_all": order_all,
                    "votes": votes,
                    "model": name
                }, ensure_ascii=False))

    out_path = pathlib.Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out_lines), encoding='utf-8')
    print("✅ votes saved →", out_path)

# ─────────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--models_yaml", default="ordering/6_models.yaml")
    ap.add_argument("--out_jsonl",   required=True)
    main(ap.parse_args())
