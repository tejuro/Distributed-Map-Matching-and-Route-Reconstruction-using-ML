"""Bayesian optimisation: tuned vs untuned comparison on the student-test trips.

The tuned configuration comes from results of the Optuna search (recorded in
models/final_model_config.json). The untuned baseline is the hand-chosen
configuration from the initial five-model comparison. To make the comparison
fair, BOTH are trained identically - on the full reliable training partition -
and evaluated with identical metrics on the student-test partition.

The only thing that differs is the hyperparameters.

    python src/tuned_vs_untuned.py
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, log_loss,
                             roc_auc_score)

SEED = 42
COMMON = dict(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED)

UNTUNED = {  # initial hand-chosen config from src/02_train_compare.py
    "n_estimators": 300, "learning_rate": 0.1, "max_depth": 6,
    "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def evaluate(model, X, df):
    p = model.predict_proba(X)[:, 1]
    y = df.is_true_arc.to_numpy()
    thr = (p >= 0.5).astype(int)

    d = df[["trip_id", "obs_idx", "arc_id", "is_true_arc"]].copy()
    d["score"] = p
    top = d.loc[d.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
    truth = (d[d.is_true_arc == 1][["trip_id", "obs_idx", "arc_id"]]
             .rename(columns={"arc_id": "true_arc"}))
    merged = top.merge(truth, on=["trip_id", "obs_idx"], how="left")
    top1 = (merged["arc_id"] == merged["true_arc"]).mean()

    return {
        "accuracy": accuracy_score(y, thr),
        "macro_f1": f1_score(y, thr, average="macro"),
        "log_loss": log_loss(y, p),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, thr),
        "top1_accuracy": top1,
    }


def main():
    cfg = json.load(open("models/final_model_config.json"))
    feats = cfg["features"]
    tuned_params = dict(cfg["params"])

    tr = pd.read_parquet("data/features/model_table_train.parquet")
    tr = tr[tr.label_reliable == True].reset_index(drop=True)
    X = tr[feats].to_numpy(dtype=np.float32)
    y = tr.is_true_arc.to_numpy(dtype=np.int8)
    print(f"training partition: {len(X):,} reliable pairs, "
          f"positive rate {y.mean():.4f}")

    te = pd.read_parquet("data/features/model_table_student_test.parquet")
    Xte = te[feats].to_numpy(dtype=np.float32)
    print(f"student-test partition: {len(Xte):,} pairs "
          f"({te.groupby(['trip_id','obs_idx']).ngroups:,} observations)")

    rows = []
    for tag, params in (("untuned", UNTUNED), ("tuned", tuned_params)):
        t0 = time.perf_counter()
        model = xgb.XGBClassifier(**params, **COMMON)
        model.fit(X, y)
        fit_s = time.perf_counter() - t0
        rep = evaluate(model, Xte, te)
        rep.update({"variant": tag, "fit_seconds": round(fit_s, 1)})
        rows.append(rep)
        print(f"{tag:9s} fit={fit_s:6.1f}s  top1={rep['top1_accuracy']:.4f}  "
              f"f1={rep['macro_f1']:.4f}  logloss={rep['log_loss']:.4f}  "
              f"pr_auc={rep['pr_auc']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv("results/tuned_vs_untuned.csv", index=False)
    print("\nsaved results/tuned_vs_untuned.csv")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
