"""
Feature engineering (Part 2) + shared helpers.
Builds the model input table from the preprocessed pairs/points/arcs files.
The SAME build_features() is used for training, student test and the API.
"""
import numpy as np
import pandas as pd

# 8 features delivered by preprocessing
BASE = ["dist_m", "proj_frac", "rank", "arc_length_m", "heading_diff_deg",
        "heading_diff_undirected_deg", "speed_mps", "n_candidates"]
# 7 features engineered here
EXTRA = ["has_heading", "dist_minus_min", "dist_ratio", "d_prev", "d_next",
         "out_deg", "in_deg"]
FEATURES = BASE + EXTRA


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def build_features(pairs, points, arcs):
    """pairs: trip_id, obs_idx, arc_id + BASE.  points: x, y, segment_id.  arcs: geometry + nodes.
    Returns pairs with FEATURES filled (no NaN) plus proj_x/proj_y for the HMM."""
    pr = pairs.copy()
    key = ["trip_id", "obs_idx"]
    P = pr[key].merge(points[key + ["x", "y", "segment_id"]], on=key, how="left")
    A = pr[["trip_id", "arc_id"]].merge(arcs[["trip_id", "arc_id", "x1", "y1", "x2", "y2",
                                              "from_node", "to_node"]], on=["trip_id", "arc_id"], how="left")

    # 1) heading is undefined when the car is stopped -> flag it, fill with "neutral" values
    pr["has_heading"] = pr.heading_diff_deg.notna().astype(int)
    pr["heading_diff_deg"] = pr.heading_diff_deg.fillna(90.0)
    pr["heading_diff_undirected_deg"] = pr.heading_diff_undirected_deg.fillna(45.0)
    pr["speed_mps"] = pr.speed_mps.fillna(0.0)

    # 2) how much farther is this candidate than the closest one?
    mn = pr.groupby(key).dist_m.transform("min")
    pr["dist_minus_min"] = pr.dist_m - mn
    pr["dist_ratio"] = pr.dist_m / (mn + 1.0)

    # 3) temporal context: distance of the previous / next GPS point (same segment) to this arc
    pts = points.set_index(key)
    for col, sh in [("d_prev", -1), ("d_next", 1)]:
        nb = pd.MultiIndex.from_arrays([pr.trip_id.values, pr.obs_idx.values + sh])
        nx = pts.x.reindex(nb).values; ny = pts.y.reindex(nb).values
        nseg = pts.segment_id.reindex(nb).values
        d, _ = seg_dist(nx, ny, A.x1.values, A.y1.values, A.x2.values, A.y2.values)
        same = nseg == P.segment_id.values
        pr[col] = np.where(same & ~np.isnan(d), d, pr.dist_m.values)   # no neighbour -> own distance

    # 4) local road connectivity (within the candidate road network of the trip)
    out_cnt = arcs.groupby(["trip_id", "from_node"]).size()
    in_cnt = arcs.groupby(["trip_id", "to_node"]).size()
    pr["out_deg"] = out_cnt.reindex(pd.MultiIndex.from_arrays([A.trip_id, A.to_node])).fillna(0).values
    pr["in_deg"] = in_cnt.reindex(pd.MultiIndex.from_arrays([A.trip_id, A.from_node])).fillna(0).values

    # projected point on the arc (used by the HMM transitions)
    pr["proj_x"] = A.x1.values + pr.proj_frac.values * (A.x2.values - A.x1.values)
    pr["proj_y"] = A.y1.values + pr.proj_frac.values * (A.y2.values - A.y1.values)
    return pr
