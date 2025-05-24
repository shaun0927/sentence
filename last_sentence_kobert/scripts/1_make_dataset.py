#!/usr/bin/env python
"""
1_make_dataset.py ─ 마지막 문장용 K-Fold 데이터 생성
────────────────────────────────────────────────────────────
train.csv → 
 • fs_train.jsonl     : 마지막 문장(binary)
 • pw_train.jsonl     : 문장쌍 순서(binary)
 • fold/fold_i.jsonl  : Stratified  K-Fold  (answer_3 기준)
"""
import argparse, json, pathlib, random
from typing import List
import pandas as pd
from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except ImportError:
    HAS_SGK = False

SEED = 42
random.seed(SEED)

# ---------- util ------------------------------------------------------------
def _write(rows:List[dict], path:pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

# ---------- 전체 샘플 --------------------------------------------------------
def build_last_samples(df:pd.DataFrame, out:pathlib.Path):
    rows=[]
    for doc, row in df.iterrows():
        gold_last = int(row["answer_3"])
        for i in range(4):
            rows.append({"text":row[f"sentence_{i}"],
                         "label":int(i==gold_last),
                         "doc_id":int(doc),
                         "sent_idx":i})
    random.shuffle(rows); _write(rows, out)

def build_pair_samples(df:pd.DataFrame, out:pathlib.Path):
    rows=[]
    for _,row in df.iterrows():
        order=[int(row[f"answer_{k}"]) for k in range(4)]  # 0→3
        rank={idx:pos for pos,idx in enumerate(order)}
        for i in range(4):
            for j in range(4):
                if i==j: continue
                label=int(rank[i] < rank[j])    # 앞에 오면 1
                rows.append({"text":f"{row[f'sentence_{i}']} [SEP] {row[f'sentence_{j}']}",
                             "label":label})
    random.shuffle(rows); _write(rows, out)

# ---------- fold ------------------------------------------------------------
def make_folds(csv:pathlib.Path, k:int):
    df=pd.read_csv(csv)
    X=df.index.values
    y=df["answer_3"].values                         # ★ answer_3
    groups=df["ID"].values if "ID" in df.columns else X
    splitter=(StratifiedGroupKFold if HAS_SGK else StratifiedKFold)(
        n_splits=k, shuffle=True, random_state=SEED)
    folds=splitter.split(X,y,groups) if HAS_SGK else splitter.split(X,y)

    fold_dir=csv.parent/"fold"; fold_dir.mkdir(parents=True, exist_ok=True)
    for fi,(_,vi) in enumerate(folds):
        vdf=df.iloc[sorted(vi)]
        rows=[]
        for doc,row in vdf.iterrows():
            gold=int(row["answer_3"])              # ★
            for i in range(4):
                rows.append({"text":row[f"sentence_{i}"],
                             "label":int(i==gold),
                             "doc_id":int(doc),
                             "sent_idx":i})
        _write(rows, fold_dir/f"fold_{fi}.jsonl")
        print(f"[fold] fold_{fi}.jsonl ({len(rows)})")

# ---------- CLI -------------------------------------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="data/proc")
    ap.add_argument("--make_fold", action="store_true")
    ap.add_argument("--k", type=int, default=7)
    ar=ap.parse_args()

    df=pd.read_csv(ar.csv); out=pathlib.Path(ar.out_dir)
    build_last_samples(df, out/"fs_train.jsonl")
    build_pair_samples(df,out/"pw_train.jsonl")
    print("✅ JSONL 생성 완료")
    if ar.make_fold: make_folds(pathlib.Path(ar.csv), ar.k)
