# tools/summarize_cv.py
import json, glob, pandas as pd, pathlib

rows = []
for f in glob.glob("first_sentence_kobert/checkpoints/*_cv_metrics.json"):
    exp = pathlib.Path(f).stem.replace("_cv_metrics", "")
    cv = json.load(open(f, encoding="utf-8"))
    f1s = [fold["eval_f1"] for fold in cv]
    rows.append({"model": exp,
                 "F1_mean": sum(f1s)/len(f1s),
                 "F1_std": pd.Series(f1s).std()})

print(pd.DataFrame(rows).sort_values("F1_mean", ascending=False)
        .to_markdown(index=False))
