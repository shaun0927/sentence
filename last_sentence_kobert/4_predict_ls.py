#!/usr/bin/env python
"""
4_predict_ls.py ─ 마지막 문장 확률 예측 & Top-k=2 추출
"""
import argparse, json, pathlib, sys, torch, numpy as np, pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ─────────── util ────────────
def find_best_dirs(root: pathlib.Path):
    if (root/"best").is_dir(): return [root/"best"]
    dirs=sorted((p/"best") for p in root.glob("*_fold*") if (p/"best").is_dir())
    if not dirs: sys.exit(f"[ERR] no checkpoint in {root}"); return dirs

def build_dataset(df, tok, max_len):
    rows=[{"ID":r.ID,"idx":i,"text":r[f"sentence_{i}"]} for _,r in df.iterrows() for i in range(4)]
    ds=Dataset.from_pandas(pd.DataFrame(rows))
    return ds.map(lambda ex: tok(ex["text"],truncation=True,max_length=max_len,
                                 return_token_type_ids=False),
                  batched=True,remove_columns=["text"])

def model_logits(model,dl):
    out=[]
    for b in dl:
        b={k:v.to(DEVICE) for k,v in b.items()}; out.append(model(**b).logits.cpu())
    return torch.cat(out,0)

# ─────────── main ────────────
def main(a):
    ck_root=pathlib.Path(a.ckpt_dir); bests=find_best_dirs(ck_root)
    tok=AutoTokenizer.from_pretrained(bests[0],use_fast=False)

    df=pd.read_csv(a.test_csv)
    ds=build_dataset(df,tok,a.max_len)
    dl=DataLoader(ds.remove_columns(["ID","idx"]),batch_size=a.batch_size,
                  collate_fn=DataCollatorWithPadding(tok))

    logits_sum=torch.zeros(len(ds),2)
    for bd in bests:
        m=AutoModelForSequenceClassification.from_pretrained(bd).to(DEVICE).eval()
        logits_sum += model_logits(m,dl); del m; torch.cuda.empty_cache()

    probs=torch.softmax(logits_sum/len(bests),dim=-1)[:,1].numpy()  # P(last)

    k=a.top_k; off=0; js,rows=[],[]
    for _,r in df.iterrows():
        p4=probs[off:off+4]; off+=4
        order=p4.argsort()[::-1][:k]                 # Top-k 높은 확률 ⇒ last 후보
        js.append(json.dumps({"ID":r.ID,"rank":order.tolist(),
                              "prob":p4[order].round(6).tolist()},
                             ensure_ascii=False))
        rows.extend({"ID":r.ID,"idx":i,"prob_last":float(p4[i])} for i in range(4))

    pathlib.Path(a.out_jsonl).write_text("\n".join(js),encoding="utf-8")
    pd.DataFrame(rows).to_csv(a.out_csv,index=False)
    print(f"✅ saved jsonl:{a.out_jsonl}  csv:{a.out_csv}")

if __name__=="__main__":
    P=argparse.ArgumentParser()
    P.add_argument("--ckpt_dir",required=True)
    P.add_argument("--test_csv",required=True)
    P.add_argument("--out_jsonl",default="data/proc/test_last_top2.jsonl")
    P.add_argument("--out_csv",default="data/proc/test_last_probs.csv")
    P.add_argument("--top_k",type=int,default=2)        # ★ default 2
    P.add_argument("--max_len",type=int,default=96)
    P.add_argument("--batch_size",type=int,default=128)
    main(P.parse_args())
