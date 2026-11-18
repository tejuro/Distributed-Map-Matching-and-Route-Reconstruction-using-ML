# PathSnap – Final Data Preprocessing

**Task:** match each noisy GPS point (`.track`) to the road segment (`arc`) the vehicle was really on (`.route`).

There is now **one script**, `data_processing/preprocess.py`. It goes straight from raw data to the final files, so the labels, candidate roads, pairs and summary all come from the same final cleaned points. It replaces `01_preprocess.py` and `02_clean.py`.

```
python data_processing/preprocess.py "<raw Student folder>" data/processed     (~45 s)
```

## Pipeline (per trip)

```
raw .track .nodes .arcs .route
 1. remove invalid lon/lat, duplicate rows/timestamps, sort by time
 2. project lon/lat -> metres (local UTM zone)
 3. remove GPS points > 1 km outside the road-network box
 4. remove speed outliers (> 55 m/s): two-sided spikes, then one-sided jumps
 5. kinematics: dt, step, speed, heading; reset at time gaps > 30 s;
    heading = NaN when stationary (< 0.5 m)                    <- FINAL POINTS
 6. candidate roads: exact point-to-segment distance <= 300 m   (no route info)
 7. GPS-candidate pairs: <= 100 m, 8 nearest per point          (no route info)
 8. labels: monotone alignment of the FINAL points to the route
 9. trip check: median label distance > 30 m -> train trip excluded everywhere
10. consistency checks (the script stops with an error if any fail)
```

## Every review issue and how it is resolved

| # | Issue | Resolution |
|---|-------|-----------|
| R1 | `monotone_align()` reported as critical | Tested against a brute-force solver on 800 random cases, including tied distances. The **original was already optimal** in every case. It has been rewritten anyway with explicit first-minimum tracking and comments, so it is easier to verify; the new version passes the same tests. |
| R2 | Labels made before pass 2 removed points | Single pipeline: labels are computed **after** all cleaning (step 8). |
| R3 | Trips 33/54 had points but no arcs | Trips 33 (median 123 m) and 54 (140 m) are now excluded from **every** train file. They are still listed in `trip_summary.csv` with `excluded_from_outputs=True` and a reason. An automatic check confirms that points, arcs and pairs have identical trip IDs. |
| R4 | `label_arc_id` is trip-specific, not a global class | New **pairs** files: one row per (GPS point, candidate road) with general features. The target is `is_true_arc` (yes/no), which works across trips. |
| R5 | `trip_summary.csv` described first-pass data | Rebuilt from the final data: final counts, candidate counts, label coverage, max speed, exclusion flag. |
| T1 | `near \| a.in_route` leaked the route into candidates | Removed. Candidates use only GPS and road geometry. Coverage is still complete: **100 %** of reliable labels are in the candidate arcs, and **99.4 %** are in the 8-nearest pairs. |
| T2 | Inconsistent trip IDs between points and arcs | Fixed (see R3) and checked automatically. |
| T3 | Candidates checked only endpoints and midpoints | Now the exact minimum distance from each GPS point to the whole segment. The distance is stored as `min_dist_to_track_m`. |
| Earlier | Unreliable labels, leakage columns, speed across gaps, one-sided jumps, stationary heading | All kept from the previous round, now in one script. |

## Output files (`data/processed/`)

| File | Rows (train / test) | Use |
|------|------|-----|
| `points_<part>_features.csv` | 181,863 / 19,156 | Cleaned GPS points, inputs only |
| `points_<part>_labels.csv` | same | `label_arc_id`, `label_reliable`, … (join on `trip_id, obs_idx`) |
| `arcs_<part>_features.csv` | 431,708 / 42,177 | Candidate roads: geometry, length, bearing, `min_dist_to_track_m` |
| `arcs_<part>_labels.csv` | same | `in_route` (evaluation only) |
| `pairs_<part>_features.csv` | 1,230,673 / 130,912 | **For the classifier / HMM emissions:** `dist_m`, `proj_frac`, `rank`, `arc_length_m`, `heading_diff_deg`, `heading_diff_undirected_deg`, `speed_mps`, `n_candidates` |
| `pairs_<part>_labels.csv` | same | `is_true_arc` (target; about 14.6 % positive) |
| `trip_summary.csv` | 90 trips | Final per-trip statistics and exclusions |

**Columns never to use as inputs:** `route_pos`, `label_arc_id`, `label_dist_m`, `label_frac`, `label_reliable`, `in_route`, `is_true_arc`. The script checks that none of them appear in a features file.

## Notes for each member

- **Member 2 (classifier):** Train on the pairs. Join features to labels on `trip_id, obs_idx, arc_id`, then join `points_train_labels` and keep only rows where `label_reliable == True`; that keeps 180,757 of 181,863 points. Predict `is_true_arc`, then for each point choose the highest-scoring arc. Split validation **by trip**. Classes are imbalanced (about 1 true arc in 7), so use a class weight or a ranking metric.
- **Member 3 (HMM):** Pairs give the emission candidates. Use `arcs_*_features` (`from_node`, `to_node`) for the transition graph. Trips 33 and 54 do not appear anywhere in train. 311 train points (and 0 test points) have no road within 100 m. They have no pairs, so treat them as gaps or skip them.
- **Both:** `student_test` labels are for scoring only. NaN in `dt`, `step_m`, `speed_mps` and `heading_deg` at segment starts and stationary points is expected.
