"""
PathSnap - single, final preprocessing pipeline (replaces 01_preprocess.py + 02_clean.py).

Usage:  python preprocess.py <raw_root_with_train_and_student_test> <out_dir>

Order of operations (labels and candidates are ALWAYS built from the final cleaned points):
  1. load raw trip           5. kinematics (reset at time gaps)
  2. remove invalid/dupes    6. candidate roads: exact point-to-segment distance, no route info
  3. project to metres       7. labels: fixed monotone alignment to the reference route
  4. remove GPS outliers     8. trip-level reliability -> unreliable train trips excluded everywhere
Then: write leakage-safe feature / label files, GPS-candidate pair table, final trip summary,
and run consistency checks (the script fails loudly if any check fails).
"""
import sys, glob, os
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

OUTSIDE_NET_M = 1000   # m   GPS farther than this outside the road-network box is invalid
MAX_SPEED = 55.0       # m/s (~200 km/h) physically implausible for road vehicles
GAP_SPLIT = 30.0       # s   time gap that starts a new sub-trajectory segment
STATIONARY_M = 0.5     # m   step below this = not moving (heading undefined)
CAND_RADIUS = 300      # m   arc is a candidate if any GPS point is within this distance
PAIR_RADIUS = 100      # m   GPS-candidate pairs for the classifier / HMM emission
PAIR_TOPK = 8          #     at most this many nearest candidate arcs per GPS point
MAX_LABEL_DIST = 100   # m   per-point label unreliable beyond this
TRIP_LABEL_MEDIAN = 30 # m   whole trip unreliable if median label distance exceeds this

POINT_LABELS = ["route_pos", "label_arc_id", "label_dist_m", "label_frac", "label_reliable"]
ARC_LABELS = ["in_route"]
PAIR_LABELS = ["is_true_arc"]


def utm_for(lon, lat):
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return epsg, Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def load(path, ncol):
    a = np.loadtxt(path, ndmin=2)
    assert a.shape[1] == ncol, path
    return a


def seg_dist(px, py, ax, ay, bx, by):
    """Element-wise distance from point i to segment i, plus projection fraction t in [0,1]."""
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def monotone_align(D):
    """For GPS points i=0..N-1 and route positions j=0..R-1 with cost D[i,j], find a
    non-decreasing path j_0 <= j_1 <= ... minimising sum D[i, j_i].
    cost_i[j] = D[i,j] + min_{k<=j} cost_{i-1}[k];  argmin tracked explicitly (first minimum)."""
    N, R = D.shape
    cost = D[0].astype(np.float64)
    back = np.empty((N, R), dtype=np.int32)
    back[0] = np.arange(R)
    for i in range(1, N):
        best_val = np.empty(R); best_idx = np.empty(R, dtype=np.int32)
        # running prefix minimum with its index (vectorised: index of first prefix min)
        pm = np.minimum.accumulate(cost)
        is_new_min = np.r_[True, cost[1:] < pm[:-1]]          # strictly better than everything before
        best_idx[:] = np.maximum.accumulate(np.where(is_new_min, np.arange(R), 0))
        best_val[:] = pm
        back[i] = best_idx
        cost = D[i] + best_val
    path = np.empty(N, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for i in range(N - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def kinematics(df):
    dt = np.r_[np.nan, np.diff(df.t.values)]
    dx = np.r_[np.nan, np.diff(df.x.values)]; dy = np.r_[np.nan, np.diff(df.y.values)]
    df["dt"] = dt
    df["step_m"] = np.hypot(dx, dy)
    df["speed_mps"] = df.step_m / df.dt
    df["heading_deg"] = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    df["gap_before"] = df.dt.fillna(0) > GAP_SPLIT
    df["segment_id"] = df.gap_before.cumsum()
    first = df.dt.isna() | df.gap_before                       # no valid previous point
    df.loc[first, ["dt", "step_m", "speed_mps", "heading_deg"]] = np.nan
    df["is_stationary"] = (df.step_m < STATIONARY_M).fillna(False)
    df.loc[df.is_stationary, "heading_deg"] = np.nan          # direction of a ~0 m step is noise
    return df


def remove_speed_outliers(df):
    """1) two-sided spikes (impossible speed in AND out) -> isolated bad fix;
       2) remaining one-sided jumps -> drop the point arrived at with impossible speed.
       Speeds are only measured inside a segment (never across a time gap)."""
    spikes = jumps = 0
    while len(df) > 2:
        v = kinematics(df.copy()).speed_mps.values
        vin, vout = v, np.r_[v[1:], np.nan]
        spike = (vin > MAX_SPEED) & (vout > MAX_SPEED)
        if spike.any():
            spikes += int(spike.sum()); df = df[~spike].reset_index(drop=True); continue
        jump = vin > MAX_SPEED
        if jump.any():
            jumps += int(jump.sum()); df = df[~jump].reset_index(drop=True); continue
        break
    return df, spikes, jumps


def candidate_pairs(px, py, a):
    """Exact (arc, point) pairs with point-to-SEGMENT distance <= CAND_RADIUS.
    Uses only road geometry + GPS; the reference route is never consulted."""
    tree = cKDTree(np.c_[px, py])
    mx, my = (a.x1.values + a.x2.values) / 2, (a.y1.values + a.y2.values) / 2
    reach = a.length_m.values / 2 + CAND_RADIUS               # any point within CAND_RADIUS of the
    A, P = [], []                                              # segment lies within this of its midpoint
    for s in range(0, len(a), 20000):
        lists = tree.query_ball_point(np.c_[mx[s:s+20000], my[s:s+20000]], reach[s:s+20000])
        n = np.fromiter((len(l) for l in lists), int, len(lists))
        if n.sum():
            A.append(np.repeat(np.arange(s, s + len(lists)), n)); P.append(np.concatenate(lists).astype(int))
    if not A:
        return np.array([], int), np.array([], int), np.array([]), np.array([])
    A, P = np.concatenate(A), np.concatenate(P)
    d, t = seg_dist(px[P], py[P], a.x1.values[A], a.y1.values[A], a.x2.values[A], a.y2.values[A])
    k = d <= CAND_RADIUS
    return A[k], P[k], d[k], t[k]


def process_trip(tdir, partition):
    tid = os.path.basename(tdir)
    s = {"trip_id": tid, "partition": partition}
    trk = load(f"{tdir}/{tid}.track", 3); nodes = load(f"{tdir}/{tid}.nodes", 2)
    arcs = load(f"{tdir}/{tid}.arcs", 2).astype(int)
    rp = f"{tdir}/{tid}.route"
    route = load(rp, 1)[:, 0].astype(int) if os.path.exists(rp) else None

    df = pd.DataFrame(trk, columns=["lon", "lat", "t"]); s["raw_points"] = len(df)
    bad = df.isna().any(axis=1) | ~df.lon.between(-180, 180) | ~df.lat.between(-90, 90)
    s["removed_invalid"] = int(bad.sum()); df = df[~bad]
    n = len(df); df = df.drop_duplicates().drop_duplicates("t"); s["removed_duplicates"] = n - len(df)
    s["was_unsorted"] = bool((np.diff(df.t.values) < 0).any())
    df = df.sort_values("t").reset_index(drop=True)

    epsg, TO_M = utm_for(df.lon.median(), df.lat.median()); s["utm_epsg"] = epsg
    df["x"], df["y"] = TO_M.transform(df.lon.values, df.lat.values)
    nx, ny = TO_M.transform(nodes[:, 0], nodes[:, 1])
    m = OUTSIDE_NET_M
    out = ~(df.x.between(nx.min() - m, nx.max() + m) & df.y.between(ny.min() - m, ny.max() + m))
    s["removed_outside_network"] = int(out.sum()); df = df[~out].reset_index(drop=True)
    df, s["removed_speed_spikes"], s["removed_speed_jumps"] = remove_speed_outliers(df)
    df = kinematics(df)                                        # FINAL point set from here on
    df.insert(0, "trip_id", tid); df.insert(1, "partition", partition)
    df.insert(2, "obs_idx", np.arange(len(df)))

    # arcs (geometry only)
    a = pd.DataFrame({"trip_id": tid, "arc_id": np.arange(len(arcs)),
                      "from_node": arcs[:, 0], "to_node": arcs[:, 1]})
    a["x1"], a["y1"] = nx[arcs[:, 0]], ny[arcs[:, 0]]
    a["x2"], a["y2"] = nx[arcs[:, 1]], ny[arcs[:, 1]]
    a["length_m"] = np.hypot(a.x2 - a.x1, a.y2 - a.y1)
    a["bearing_deg"] = (np.degrees(np.arctan2(a.x2 - a.x1, a.y2 - a.y1)) + 360) % 360

    # candidates (no route information used)
    A, P, d, t = candidate_pairs(df.x.values, df.y.values, a)
    cand = np.unique(A)
    a["min_dist_to_track_m"] = np.nan
    a.loc[cand, "min_dist_to_track_m"] = pd.Series(d).groupby(A).min().loc[cand].values
    a["in_route"] = a.arc_id.isin(route) if route is not None else False
    s["n_nodes"], s["n_arcs"], s["candidate_arcs"] = len(nodes), len(arcs), len(cand)

    # GPS-candidate pairs: <= PAIR_RADIUS, top-K nearest per point
    pr = pd.DataFrame({"obs_idx": P, "arc_id": A, "dist_m": d, "proj_frac": t})
    pr = pr[pr.dist_m <= PAIR_RADIUS].sort_values(["obs_idx", "dist_m"])
    pr = pr.groupby("obs_idx").head(PAIR_TOPK).reset_index(drop=True)
    pr["rank"] = pr.groupby("obs_idx").cumcount()
    pr["arc_length_m"] = a.length_m.values[pr.arc_id]
    hd = df.heading_deg.values[pr.obs_idx]; br = a.bearing_deg.values[pr.arc_id]
    diff = np.abs((hd - br + 180) % 360 - 180)
    pr["heading_diff_deg"] = diff                              # vs. travelling A->B
    pr["heading_diff_undirected_deg"] = np.minimum(diff, 180 - diff)
    pr["speed_mps"] = df.speed_mps.values[pr.obs_idx]
    pr["n_candidates"] = pr.groupby("obs_idx").arc_id.transform("size")
    pr.insert(0, "trip_id", tid)
    s["points_without_candidate"] = int(len(df) - pr.obs_idx.nunique())

    # labels: monotone alignment of FINAL points to the reference route
    if route is not None:
        D, F = seg_dist(df.x.values[:, None].astype(np.float32), df.y.values[:, None].astype(np.float32),
                        a.x1.values[route].astype(np.float32), a.y1.values[route].astype(np.float32),
                        a.x2.values[route].astype(np.float32), a.y2.values[route].astype(np.float32))
        idx = monotone_align(D)
        r = np.arange(len(df))
        df["route_pos"] = idx; df["label_arc_id"] = route[idx]
        df["label_dist_m"] = D[r, idx].astype(float); df["label_frac"] = F[r, idx].astype(float)
        del D, F
        s["route_len"] = len(route)
        s["route_arcs_valid"] = bool((route < len(arcs)).all())
        s["route_connected_breaks"] = int((arcs[route[:-1], 1] != arcs[route[1:], 0]).sum())
        s["label_median_dist_m"] = float(df.label_dist_m.median())
        s["trip_label_ok"] = s["label_median_dist_m"] <= TRIP_LABEL_MEDIAN
        df["label_reliable"] = (df.label_dist_m <= MAX_LABEL_DIST) & s["trip_label_ok"]
        s["label_unreliable_points"] = int((~df.label_reliable).sum())
        key = set(zip(pr.obs_idx, pr.arc_id))
        s["label_in_candidate_arcs_pct"] = 100 * float(np.isin(df.label_arc_id, cand).mean())
        s["label_in_pairs_pct"] = 100 * float(np.mean([(o, l) in key for o, l in zip(df.obs_idx, df.label_arc_id)]))
        pr["is_true_arc"] = pr.arc_id.values == df.label_arc_id.values[pr.obs_idx]

    s["clean_points"] = len(df); s["n_segments"] = int(df.segment_id.nunique())
    s["max_gap_s"] = float(np.nanmax(np.diff(df.t))) if len(df) > 1 else 0.0
    s["duration_s"] = float(df.t.iloc[-1] - df.t.iloc[0])
    s["path_len_km"] = float(np.nansum(df.step_m) / 1000)
    s["max_speed_mps"] = float(np.nanmax(df.speed_mps))
    s["n_pairs"] = len(pr)
    return df, a.iloc[cand].reset_index(drop=True), pr, s


summaries = []
for part in ["train", "student_test"]:
    P, A, PR = [], [], []
    for tdir in sorted(glob.glob(f"{RAW}/{part}/*")):
        df, a, pr, s = process_trip(tdir, part)
        excluded = part == "train" and not s.get("trip_label_ok", True)
        s["excluded_from_outputs"] = excluded
        s["exclusion_reason"] = (f"median GPS-to-route distance {s['label_median_dist_m']:.0f} m "
                                 f"> {TRIP_LABEL_MEDIAN} m (reference route unreliable)") if excluded else ""
        summaries.append(s)
        print(part, s["trip_id"], s["raw_points"], "->", s["clean_points"], "EXCLUDED" if excluded else "")
        if not excluded:
            P.append(df); A.append(a); PR.append(pr)
    p, a, pr = pd.concat(P, ignore_index=True), pd.concat(A, ignore_index=True), pd.concat(PR, ignore_index=True)

    p.drop(columns=POINT_LABELS).to_csv(f"{OUT}/points_{part}_features.csv", index=False)
    p[["trip_id", "obs_idx"] + POINT_LABELS].to_csv(f"{OUT}/points_{part}_labels.csv", index=False)
    a.drop(columns=ARC_LABELS).to_csv(f"{OUT}/arcs_{part}_features.csv", index=False)
    a[["trip_id", "arc_id"] + ARC_LABELS].to_csv(f"{OUT}/arcs_{part}_labels.csv", index=False)
    pr.drop(columns=PAIR_LABELS).to_csv(f"{OUT}/pairs_{part}_features.csv", index=False)
    pr[["trip_id", "obs_idx", "arc_id"] + PAIR_LABELS].to_csv(f"{OUT}/pairs_{part}_labels.csv", index=False)

    # ---------------- consistency checks (fail loudly) ----------------
    tp, ta, tr = set(p.trip_id), set(a.trip_id), set(pr.trip_id)
    assert tp == ta == tr, f"{part}: trip IDs differ between points/arcs/pairs"
    assert not p.duplicated(["trip_id", "obs_idx"]).any()
    assert (p.groupby("trip_id").obs_idx.apply(lambda s: (s.values == np.arange(len(s))).all())).all()
    assert (p.speed_mps.dropna() <= MAX_SPEED).all()
    assert not p.loc[p.gap_before, "speed_mps"].notna().any()
    akeys = set(zip(a.trip_id, a.arc_id))
    assert all(k in akeys for k in zip(pr.trip_id, pr.arc_id)), "pair arc missing from arc table"
    rel = p[p.label_reliable]
    miss = sum(k not in akeys for k in zip(rel.trip_id, rel.label_arc_id))
    print(f"{part}: reliable labels whose arc is not a candidate: {miss} / {len(rel)}")
    leak = set(POINT_LABELS + ARC_LABELS + PAIR_LABELS)
    for f in ["points", "arcs", "pairs"]:
        cols = set(pd.read_csv(f"{OUT}/{f}_{part}_features.csv", nrows=1).columns)
        assert not cols & leak, f"leakage column in {f}_{part}_features"
    print(f"{part}: all checks passed  points={len(p)} arcs={len(a)} pairs={len(pr)} trips={len(tp)}")

pd.DataFrame(summaries).to_csv(f"{OUT}/trip_summary.csv", index=False)
print("done")
