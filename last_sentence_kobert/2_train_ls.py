#!/usr/bin/env python
"""
2_train_ls.py ─ KoBERT 마지막 문장(binary) 분류기 (first 버전과 코드 동일)
"""
import argparse, pathlib, os, json, random, inspect, yaml
from typing import Optional, Set, List
import numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          IntervalStrategy)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict

SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ───────────── metrics (doc_id 기반) ─────────────
def compute_metrics(pred, k=2):
    y_true = pred.label_ids
    logits = pred.predictions
    probs  = torch.softmax(torch.tensor(logits), dim=-1)[:,1].numpy()
    doc_ids = pred.inputs.get("doc_id") if isinstance(pred.inputs, dict) else None
    if doc_ids is None:                       # fallback 4줄 연속
        hit=mrr=0
        for i in range(0,len(y_true),4):
            idx = slice(i,i+4)
            topk = probs[idx].argsort()[::-1][:k]
            lab4 = y_true[idx]
            rank = np.where(lab4[topk]==1)[0]
            if rank.size: hit+=1; mrr += 1/(rank[0]+1)
        n_doc=len(y_true)//4
    else:
        bucket=defaultdict(list)
        for p,y,d in zip(probs,y_true,doc_ids): bucket[int(d)].append((p,y))
        hit=mrr=0
        for arr in bucket.values():
            arr.sort(key=lambda x:-x[0])
            topk=arr[:k]
            rank=next((i for i,(_,l) in enumerate(topk) if l==1),None)
            if rank is not None: hit+=1; mrr+=1/(rank+1)
        n_doc=len(bucket)
    y_pred=(probs>0.5).astype(int)
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average="binary",zero_division=0)
    return {"accuracy":accuracy_score(y_true,y_pred),"precision":p,"recall":r,"f1":f1,
            "hit@2":hit/n_doc,"mrr@2":mrr/n_doc}

# ───────────── Trainer 확장 ─────────────
class WeightedCETrainer(Trainer):
    def __init__(self,class_weight,*args,**kw):
        super().__init__(*args,**kw)
        self._loss_fct=torch.nn.CrossEntropyLoss(weight=class_weight)
    def compute_loss(self,model,inputs,return_outputs=False,**_):
        labels=inputs.pop("labels")
        out=model(**inputs)
        loss=self._loss_fct(out.logits,labels)
        return (loss,out) if return_outputs else loss

# ───────────── 학습 1회 ─────────────
def train_once(cfg,train_path,valid_path,exp_path,exclude:Optional[Set[str]]=None):
    set_seed()
    tok=AutoTokenizer.from_pretrained(cfg["model_name"],use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],num_labels=2).to(device)

    def encode(ex):
        enc=tok(ex["text"],truncation=True,max_length=cfg.get("max_len",96),
                return_token_type_ids=False)
        enc["labels"]=ex["label"]; enc["doc_id"]=torch.tensor(ex["doc_id"],dtype=torch.int64)
        return enc

    ds_tr=load_dataset("json",data_files=train_path)["train"]
    if exclude: ds_tr=ds_tr.filter(lambda ex: ex["text"] not in exclude)
    ds_tr=ds_tr.shuffle(seed=SEED).map(encode,batched=True)
    ds_va=load_dataset("json",data_files=valid_path)["train"].map(encode,batched=True)

    exp_path.mkdir(parents=True,exist_ok=True)
    targs=TrainingArguments(
        output_dir=str(exp_path),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        include_inputs_for_metrics=True,
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.EPOCH,
    )
    class_w=torch.tensor([1.0,cfg.get("pos_weight",3.0)],device=device)
    trainer=WeightedCETrainer(class_weight=class_w,model=model,args=targs,
        train_dataset=ds_tr,eval_dataset=ds_va,tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),compute_metrics=compute_metrics)

    res=trainer.train()
    best=exp_path/"best"; trainer.save_model(str(best)); tok.save_pretrained(str(best))
    metrics=trainer.evaluate(); metrics["train_runtime"]=res.metrics["train_runtime"]; return metrics,best

# ───────────── main ─────────────
def main(a):
    cfg=yaml.safe_load(open(a.cfg,encoding="utf-8"))
    if a.epochs: cfg["epochs"]=a.epochs
    ck_root=pathlib.Path("last_sentence_kobert/checkpoints")
    if not a.cv:                                # 단일
        m,b=train_once(cfg,a.train,a.valid,ck_root/a.exp)
        print("✅ Single",json.dumps(m,indent=2,ensure_ascii=False)); print("Best→",b); return
    folds=sorted(pathlib.Path(a.fold_dir).glob("fold_*.jsonl"))
    assert folds,f"{a.fold_dir} 에 fold_*.jsonl 없음"
    res=[]
    for k,vf in enumerate(folds):
        ex=set(load_dataset("json",data_files=str(vf))["train"]["text"])
        print(f"\n── Fold {k} ──")
        m,_=train_once(cfg,a.train,str(vf),ck_root/f"{a.exp}_fold{k}",ex); m["fold"]=k; res.append(m)
        print("metrics:",m)
    f1s=[m["f1"] for m in res]; print(f"\n📊 {len(res)}-Fold avg F1 = {np.mean(f1s):.4f}")
    (ck_root/f"{a.exp}_cv_metrics.json").write_text(json.dumps(res,indent=2,ensure_ascii=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True); p.add_argument("--valid"); p.add_argument("--cfg",required=True)
    p.add_argument("--exp",required=True); p.add_argument("--epochs",type=int)
    p.add_argument("--cv",action="store_true"); p.add_argument("--fold_dir",default="data/fold_last")
    args=p.parse_args()
    if args.cv: assert pathlib.Path(args.fold_dir).is_dir()
    else:       assert args.valid,"--valid 경로 필요"
    main(args)
