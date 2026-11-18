"""Rebuild final_model.joblib from the RECORDED tuned hyperparameters.

The original artifact fails to load ("input stream corrupted" - an xgboost
serialisation version mismatch). The tuned configuration is fully recorded in
models/final_model_config.json, so the deployed model is refitted from it on the
full reliable training partition. This script does NOT re-run the Bayesian
search; it reproduces only the missing artifact.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

SEED = 42

cfg = json.load(open("models/final_model_config.json"))
feats = cfg["features"]
params = dict(cfg["params"])
print(f"features: {len(feats)}")
print(f"params: {params}")

tr = pd.read_parquet("data/features/model_table_train.parquet")
tr = tr[tr.label_reliable == True].reset_index(drop=True)
X = tr[feats].to_numpy(dtype=np.float32)
y = tr.is_true_arc.to_numpy(dtype=np.int8)
print(f"train rows: {len(X):,}  positives: {(y==1).sum():,}")

t0 = time.perf_counter()
model = xgb.XGBClassifier(**params, tree_method="hist", eval_metric="logloss",
                          n_jobs=-1, random_state=SEED)
model.fit(X, y)
print(f"fit: {time.perf_counter()-t0:.1f}s")

joblib.dump({"model": model, "features": feats, "model_name": "XGBoost"},
            "models/final_model.joblib", compress=3)
print("saved models/final_model.joblib")

# ---- sanity: load back and evaluate on student-test ----
blob = joblib.load("models/final_model.joblib")
m = blob["model"]
te = pd.read_parquet("data/features/model_table_student_test.parquet")
Xte = te[feats].to_numpy(dtype=np.float32)
p = m.predict_proba(Xte)[:, 1]

from sklearn.metrics import f1_score, log_loss, accuracy_score
thr = (p >= 0.5).astype(int)
print("\n=== student-test sanity (final model) ===")
print(f"accuracy       : {accuracy_score(te.is_true_arc, thr):.4f}")
print(f"macro f1       : {f1_score(te.is_true_arc, thr, average='macro'):.4f}")
print(f"log loss       : {log_loss(te.is_true_arc, p):.4f}")

# observation-level top-1
d = te[["trip_id", "obs_idx", "arc_id", "is_true_arc"]].copy()
d["score"] = p
idx = d.groupby(["trip_id", "obs_idx"])["score"].idxmax()
top = d.loc[idx]
print(f"obs top1       : {(top.arc_id == top[top.is_true_arc==1].arc_id).mean() if False else np.nan}")
true = d[d.is_true_arc == 1].set_index(["trip_id", "obs_idx"])
top1 = (top.set_index(["trip_id", "obs_idx"]).arc_id == true.arc_id).mean()
print(f"obs top1       : {top1:.4f}")
print("\nreference (recorded): test_top1_classifier 0.8198, macro_f1 0.8929, log_loss 0.1266")
