"""Part 2 - build the model table from the preprocessed files.
Usage: python src/01_build_features.py data/processed data/features
Output: model_table_<part>.parquet  (FEATURES + is_true_arc + label_reliable + HMM helper columns)
"""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features, FEATURES

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rd = lambda f: pd.read_csv(f"{D}/{f}", dtype={"trip_id": str})
for part in ["train", "student_test"]:
    pairs, pts, arcs = rd(f"pairs_{part}_features.csv"), rd(f"points_{part}_features.csv"), rd(f"arcs_{part}_features.csv")
    t = build_features(pairs, pts, arcs)
    # labels are joined only AFTER features are built (features never see labels)
    t = t.merge(rd(f"pairs_{part}_labels.csv"), on=["trip_id", "obs_idx", "arc_id"], how="left")
    t = t.merge(rd(f"points_{part}_labels.csv")[["trip_id", "obs_idx", "label_reliable", "label_arc_id"]],
                on=["trip_id", "obs_idx"], how="left")
    t["is_true_arc"] = t.is_true_arc.astype(int)
    assert t[FEATURES].notna().all().all(), "NaN in features"
    t.to_parquet(f"{OUT}/model_table_{part}.parquet", index=False)
    print(part, t.shape, "positives", round(t.is_true_arc.mean(), 3),
          "reliable rows", int(t.label_reliable.sum()))
