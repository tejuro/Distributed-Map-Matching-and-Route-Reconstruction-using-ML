"""Member 2 inference API - load the final model and score candidate-road pairs.

This is the interface Member 3 (and the FastAPI backend) uses to obtain
candidate-road probabilities. Two ways to use it:

1. As a library:
       from predict import load_final_model, predict_candidate_proba
       model, features = load_final_model()
       proba = predict_candidate_proba(model, candidate_table, features)

2. As a CLI (useful for checking the artifacts without training anything):
       python src/predict.py <feature_table.parquet|csv> [out_proba.npy]

The candidate table must contain the 15 feature columns listed in
models/final_model_config.json, in that exact order, one row per
(observation, candidate-arc) pair.
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models")


def load_final_model():
    """Load the deployed model and its feature list.

    Returns (model, features) where ``model`` is the tuned XGBClassifier and
    ``features`` is the ordered list of the 15 input columns.
    """
    blob = joblib.load(os.path.join(MODEL_DIR, "final_model.joblib"))
    return blob["model"], blob["features"]


def predict_candidate_proba(model, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Score each candidate row: probability that the arc is the true road.

    ``X`` may carry extra columns (trip_id, obs_idx, arc_id, ...) - only the
    feature columns are read. Rows are returned in input order.
    """
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"candidate table is missing feature columns: {missing}")
    return model.predict_proba(X[features].to_numpy(dtype=np.float32))[:, 1]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"trip_id": str})

    model, features = load_final_model()
    print(f"loaded {type(model).__name__} with {len(features)} features")
    print(f"candidate rows: {len(df):,}")

    proba = predict_candidate_proba(model, df, features)
    print(f"predicted probabilities for {len(proba):,} rows "
          f"(mean {proba.mean():.4f}, positive rate at 0.5: {(proba >= 0.5).mean():.4f})")

    # if the table carries labels, report the point-level Top-1
    if "is_true_arc" in df.columns and {"trip_id", "obs_idx", "arc_id"} <= set(df.columns):
        d = df[["trip_id", "obs_idx", "arc_id", "is_true_arc"]].copy()
        d["score"] = proba
        top = d.loc[d.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
        truth = (d[d.is_true_arc == 1][["trip_id", "obs_idx", "arc_id"]]
                 .rename(columns={"arc_id": "true_arc"}))
        merged = top.merge(truth, on=["trip_id", "obs_idx"], how="left")
        top1 = (merged["arc_id"] == merged["true_arc"]).mean()
        print(f"observation-level Top-1 accuracy: {top1:.4f} "
              f"({len(merged):,} observations)")

    if out_path:
        np.save(out_path, proba)
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
