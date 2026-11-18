"""Compute the four metrics missing from model_comparison.csv:
Top-3 candidate recall, ROC-AUC, PR-AUC, balanced accuracy.

Uses the student-test probabilities saved by src/02_train_compare.py
(models/<Model>_student_test_proba.npy) plus the final rebuilt model, scored
against the student-test labels in data/features/model_table_student_test.parquet.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             roc_auc_score)


def _row(name, te, y, proba):
    d = te[["trip_id", "obs_idx", "arc_id", "is_true_arc"]].copy()
    d["score"] = proba
    d["rank"] = d.groupby(["trip_id", "obs_idx"])["score"].rank(
        method="first", ascending=False) - 1
    true = d[d.is_true_arc == 1]
    top3 = (true["rank"] <= 2).mean()
    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    bal = balanced_accuracy_score(y, (proba >= 0.5).astype(int))
    print(f"{name:28s} top3={top3:.4f} roc={roc:.4f} pr={pr:.4f} bal={bal:.4f}")
    return {"model": name, "test_top3_recall": top3, "test_roc_auc": roc,
            "test_pr_auc": pr, "test_balanced_accuracy": bal}


te = pd.read_parquet("data/features/model_table_student_test.parquet")
y = te.is_true_arc.to_numpy()
n_obs = te.groupby(["trip_id", "obs_idx"]).ngroups
print(f"student_test pairs: {len(te):,}  observations: {n_obs:,}  "
      f"positive rate: {y.mean():.4f}")

rows = []
for name in ["LogisticRegression", "KNN", "SVM", "RandomForest", "XGBoost"]:
    p = f"models/{name}_student_test_proba.npy"
    if not os.path.exists(p):
        print(f"{name}: missing proba file - skipped")
        continue
    proba = np.load(p)
    assert len(proba) == len(y), f"{name}: length mismatch"
    rows.append(_row(name, te, y, proba))

# final rebuilt model
blob = joblib.load("models/final_model.joblib")
cfg = json.load(open("models/final_model_config.json"))
X = te[cfg["features"]].to_numpy(dtype=np.float32)
proba = blob["model"].predict_proba(X)[:, 1]
rows.append(_row("final_model (tuned XGBoost)", te, y, proba))

out = pd.DataFrame(rows)
out.to_csv("results/extra_metrics.csv", index=False)
print("\n" + out.round(4).to_string(index=False))
