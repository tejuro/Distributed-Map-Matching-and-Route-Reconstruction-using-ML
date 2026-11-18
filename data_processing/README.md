# data_processing/ — Member 1: Dataset & Feature Engineering (Part 2)

Dataset exploration, preprocessing and the 15 ML features that every model in
Part 3 trains on and that the backend uses at inference time.

---

## Pipeline

```
data/raw/<train|student_test>/<trip_id>/*.{track,nodes,arcs,route}
        │
        ▼  data_processing/preprocess.py
data/processed/
   pairs_{partition}_features.csv     candidate-pair features
   pairs_{partition}_labels.csv       is_true_arc
   points_{partition}_features.csv    cleaned GPS points + kinematics
   points_{partition}_labels.csv      label_arc_id, label_reliable
   arcs_{partition}_features.csv      road-arc geometry
        │
        ▼  src/01_build_features.py
data/features/
   model_table_train.parquet
   model_table_student_test.parquet
        │
        ▼  used by src/02_train_compare.py and api/
```

## Run it

```bash
python data_processing/preprocess.py data/raw data/processed
python src/01_build_features.py data/processed data/features
```

`preprocess.py` runs consistency checks and **fails loudly** rather than
silently producing bad features.

---

## What preprocessing does

1. Load the raw trip (`.track`, `.nodes`, `.arcs`, `.route`).
2. Remove invalid coordinates, duplicates and GPS outliers.
3. Project longitude/latitude to metres. Degrees are angles: one degree of
   longitude is ~111 km × cos(latitude), so Euclidean distance on raw degrees
   would be badly skewed by latitude. Each trip is projected into its own
   metric frame.
4. Build candidate roads by exact point-to-segment distance — **no route
   information is used here**, so candidates cannot leak the answer.
5. Label the reference arc by monotone alignment to the reference route.
6. Mark each point's label as reliable or not; unreliable points are excluded
   from supervised training.

Trip-level reliability: trips whose reference route is systematically offset
from their own GPS track are excluded from training. Two training trips were
excluded this way.

---

## The 15 features

| Feature | Meaning |
|---|---|
| `dist_m` | perpendicular distance from the GPS point to the candidate arc, metres |
| `proj_frac` | where along the arc the projection lands, 0–1 |
| `rank` | this candidate's rank by distance for the observation (0 = closest) |
| `arc_length_m` | length of the candidate arc |
| `heading_diff_deg` | signed difference between travel bearing and arc bearing |
| `heading_diff_undirected_deg` | absolute value of the above |
| `speed_mps` | observed speed at the point |
| `n_candidates` | number of candidates generated for the point |
| `has_heading` | 0 when the GPS step has no usable bearing (e.g. stationary) |
| `dist_minus_min` | candidate distance minus the closest candidate's distance |
| `dist_ratio` | candidate distance divided by the closest candidate's distance |
| `d_prev` | distance of this arc at the **previous** observation |
| `d_next` | distance of this arc at the **next** observation |
| `out_deg` | out-degree of the arc's from-node (network topology) |
| `in_deg` | in-degree of the arc's to-node |

**Feature order is fixed** by `models/final_model_config.json`. The trained model
expects exactly this sequence — do not reorder.

The table also carries, for evaluation and decoding only (never as model input):
`is_true_arc`, `label_reliable`, `label_arc_id`, and the HMM helpers `proj_x`,
`proj_y`.

---

## Results in this folder

| File | Contents |
|---|---|
| `results/dataset_inventory.csv` | per-trip counts, extents, gaps |
| `results/data_quality_summary.csv` | missing values, invalid coordinates, anomalies |
| `results/missing_values_pairs_train.csv` | missingness audit on the pair table |
| `results/plots/eda_overview.png` | combined EDA figure |
| `results/plots/trip_locations.png` | where the 90 trips are (the dataset is global) |
| `results/plots/gps_noise_distance.png` | GPS noise relative to the road network |
| `results/plots/feature_separation.png` | how well features separate true/false candidates |
| `results/plots/trajectory_train_00000001.png` | one full trajectory |
| `results/maps/*.html` | interactive Folium maps (GPS points, road network, reference route) |

---

## Data-quality findings

- Coordinates are clean: no NaN, no (0,0) sentinels, no out-of-range values, no
  duplicate or non-monotonic timestamps.
- Sampling is 1 Hz nominal, but the largest gap is **153,064 s (42.5 h)** on trip
  `00000090`; 135 gaps exceed 30 s and 42 exceed 120 s.
- **Two training trips had reference routes systematically offset from their own
  GPS tracks** (median 122 m and 138 m; 66% and 88% of points more than 40 m
  off). All other trips sit at a 4.5 m median. Those two are excluded from
  training and their exclusion is documented in `trip_summary.csv`.
- The dataset is **global**, not one city — trip centroids span longitude
  −100.3 to 137.2 and latitude −34.6 to 61.7. That is why projection happens
  per trip rather than once globally.

---

## Note on authorship

Part 2 belongs to Member 1. Member 1 became unavailable (AI token limit) before
pushing, so this part was committed by Member 2 **from the team's existing
project package, unchanged**, to keep the pipeline complete and integration
unblocked. No content was invented or modified.
