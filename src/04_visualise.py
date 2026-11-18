"""EDA plots + Folium maps (run after 01-03). The partition used is written in each title.
Usage: python src/04_visualise.py <raw_root> data/processed data/features results models
"""
import sys, os, json
import numpy as np, pandas as pd, joblib, folium
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES
from hmm import viterbi, collapse

RAW, D, F, R, M = sys.argv[1:6]
P = f"{R}/plots"; os.makedirs(P, exist_ok=True); os.makedirs(f"{R}/maps", exist_ok=True)
rd = lambda f: pd.read_csv(f"{D}/{f}", dtype={"trip_id": str})
s = rd("trip_summary.csv")
pts = rd("points_train_features.csv"); lab = rd("points_train_labels.csv")

inv = s.groupby("partition").agg(trips=("trip_id", "count"), excluded=("excluded_from_outputs", "sum"),
                                 raw_points=("raw_points", "sum"), clean_points=("clean_points", "sum"),
                                 road_nodes=("n_nodes", "sum"), road_arcs=("n_arcs", "sum"),
                                 candidate_arcs=("candidate_arcs", "sum"))
inv.to_csv(f"{R}/dataset_inventory.csv")
s.groupby("partition")[["removed_invalid", "removed_duplicates", "removed_outside_network",
                        "removed_speed_spikes", "removed_speed_jumps", "points_without_candidate",
                        "label_unreliable_points"]].sum().to_csv(f"{R}/data_quality_summary.csv")
pf = pd.read_csv(f"{D}/pairs_train_features.csv", nrows=400000)
pf.isna().mean().mul(100).round(2).rename("missing_%").to_csv(f"{R}/missing_values_pairs_train.csv")

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].hist(s.raw_points, bins=30, color="#4C78A8"); ax[0, 0].set_title("GPS points per trip (train + student test)")
dt = pts.dt.dropna(); ax[0, 1].hist(dt.clip(upper=10), bins=20, color="#F58518")
ax[0, 1].set_title(f"Sampling interval, s (train). Exactly 1 s: {(dt == 1).mean():.1%}")
ax[1, 0].hist(pts.speed_mps.dropna() * 3.6, bins=50, color="#54A24B")
ax[1, 0].set_title("Speed between fixes, km/h (train, cleaned)")
rem = s[["removed_speed_spikes", "removed_speed_jumps", "points_without_candidate", "label_unreliable_points"]].sum()
ax[1, 1].bar(range(4), rem.values, color="#E45756"); ax[1, 1].set_xticks(range(4))
ax[1, 1].set_xticklabels(["spikes", "jumps", "no road\n<100 m", "unreliable\nlabel"])
ax[1, 1].set_title("Data-quality issues (all partitions)")
plt.tight_layout(); plt.savefig(f"{P}/eda_overview.png", dpi=110); plt.close()

plt.figure(figsize=(9, 4.5))
for part, col in [("train", "#4C78A8"), ("student_test", "#E45756")]:
    q = s[s.partition == part]
    lon = pts.groupby("trip_id").lon.mean() if part == "train" else rd("points_student_test_features.csv").groupby("trip_id").lon.mean()
    lat = pts.groupby("trip_id").lat.mean() if part == "train" else rd("points_student_test_features.csv").groupby("trip_id").lat.mean()
    plt.scatter(lon, lat, c=col, s=25, label=part)
plt.legend(); plt.grid(alpha=.3); plt.xlabel("longitude"); plt.ylabel("latitude")
plt.title("Where the trips are (mean position of each trip)"); plt.tight_layout()
plt.savefig(f"{P}/trip_locations.png", dpi=110); plt.close()

plt.figure(figsize=(7, 3.5))
plt.hist(lab.label_dist_m.clip(upper=60), bins=60, color="#72B7B2")
plt.title("GPS point -> reference road distance, m (train, clipped at 60)"); plt.tight_layout()
plt.savefig(f"{P}/gps_noise_distance.png", dpi=110); plt.close()

tb = pd.read_parquet(f"{F}/model_table_train.parquet").sample(200000, random_state=42)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for i, (col, rng_) in enumerate([("dist_m", (0, 100)), ("heading_diff_deg", (0, 180)), ("d_next", (0, 100))]):
    for v, c in [(0, "#999999"), (1, "#E45756")]:
        ax[i].hist(tb[tb.is_true_arc == v][col].clip(*rng_), bins=40, alpha=.6, density=True, color=c,
                   label=["wrong road", "true road"][v])
    ax[i].set_title(f"{col} (train)"); ax[i].legend()
plt.tight_layout(); plt.savefig(f"{P}/feature_separation.png", dpi=110); plt.close()

tid = sorted(pts.trip_id.unique())[0]
arcs = rd("arcs_train_features.csv"); al = rd("arcs_train_labels.csv")
a = arcs[arcs.trip_id == tid].merge(al, on=["trip_id", "arc_id"]); q = pts[pts.trip_id == tid]
plt.figure(figsize=(7, 7))
for r in a.itertuples():
    plt.plot([r.x1, r.x2], [r.y1, r.y2], color="#cccccc", lw=.5)
for r in a[a.in_route].itertuples():
    plt.plot([r.x1, r.x2], [r.y1, r.y2], color="green", lw=2.5)
plt.scatter(q.x, q.y, s=2, c="blue"); plt.axis("equal")
plt.title(f"Train trip {tid}: candidate roads (grey), reference (green), GPS (blue)")
plt.tight_layout(); plt.savefig(f"{P}/trajectory_train_{tid}.png", dpi=110); plt.close()

# ---- Folium: student-test trips, GPS + reference + prediction
model = joblib.load(f"{M}/final_model.joblib"); hp = json.load(open(f"{M}/hmm_params.json"))
te = pd.read_parquet(f"{F}/model_table_student_test.parquet")
pte, ate = rd("points_student_test_features.csv"), rd("arcs_student_test_features.csv")
for tid in sorted(pte.trip_id.unique())[:3]:
    nodes = np.loadtxt(f"{RAW}/student_test/{tid}/{tid}.nodes", ndmin=2)
    arcs_raw = np.loadtxt(f"{RAW}/student_test/{tid}/{tid}.arcs", ndmin=2).astype(int)
    route = np.loadtxt(f"{RAW}/student_test/{tid}/{tid}.route", ndmin=1).astype(int)
    p_ = pte[pte.trip_id == tid]; c = te[te.trip_id == tid]; a = ate[ate.trip_id == tid]
    pred = collapse(viterbi(c, p_, a, model.predict_proba(c[FEATURES].values)[:, 1], hp))
    seg = lambda ids: [[(nodes[arcs_raw[i, 0], 1], nodes[arcs_raw[i, 0], 0]),
                        (nodes[arcs_raw[i, 1], 1], nodes[arcs_raw[i, 1], 0])] for i in ids]
    mp = folium.Map(location=[p_.lat.mean(), p_.lon.mean()], zoom_start=14, tiles="cartodbpositron")
    g1 = folium.FeatureGroup("GPS points (blue)").add_to(mp)
    for la, lo in zip(p_.lat.values[::3], p_.lon.values[::3]):
        folium.CircleMarker((la, lo), radius=2, color="blue", fill=True).add_to(g1)
    folium.PolyLine(seg(route), color="green", weight=7, opacity=.5).add_to(
        folium.FeatureGroup("Reference route (green)").add_to(mp))
    folium.PolyLine(seg(pred), color="red", weight=3).add_to(folium.FeatureGroup("Predicted route (red)").add_to(mp))
    folium.LayerControl(collapsed=False).add_to(mp)
    mp.save(f"{R}/maps/student_test_{tid}.html"); print("map", tid)
