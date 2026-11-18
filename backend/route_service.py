from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from backend.ml_hmm import decode_candidate_scores

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_data():
    df = pd.read_parquet(
        ROOT / "data/features/model_table_student_test.parquet"
    )

    probabilities = np.load(
        ROOT / "data/features/student_test_proba.npy"
    )

    if len(df) != len(probabilities):
        raise ValueError("Prediction count does not match candidate rows")

    df["probability"] = probabilities

    arcs = pd.read_csv(
        ROOT / "data/processed/arcs_student_test_features.csv",
        dtype={"trip_id": str}
    )

    points = pd.read_csv(
        ROOT / "data/processed/points_student_test_features.csv",
        dtype={"trip_id": str}
    )

    return df, arcs, points


def predict_route(trip_id):
    df, arcs, points = load_data()

    trip = df[df.trip_id == trip_id].copy()
    trip_arcs = arcs[arcs.trip_id == trip_id]
    trip_points = points[points.trip_id == trip_id]

    if trip.empty or trip_arcs.empty or trip_points.empty:
        raise ValueError("Trip not found")

    roads = {
        int(r.arc_id): (int(r.from_node), int(r.to_node))
        for r in trip_arcs.itertuples(index=False)
    }

    path, restarts = decode_candidate_scores(
        trip[["obs_idx", "arc_id", "probability"]].to_dict("records"),
        roads
    )

    lon = float(trip_points.lon.median())
    lat = float(trip_points.lat.median())

    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        f"EPSG:{epsg}",
        "EPSG:4326",
        always_xy=True
    )

    geometry = trip_arcs.set_index("arc_id")

    predicted_route = []

    for arc_id in path:
        road = geometry.loc[arc_id]

        start_lon, start_lat = transformer.transform(
            float(road.x1), float(road.y1)
        )

        end_lon, end_lat = transformer.transform(
            float(road.x2), float(road.y2)
        )

        predicted_route.append({
            "arc_id": int(arc_id),
            "start": [start_lat, start_lon],
            "end": [end_lat, end_lon]
        })

    gps_points = [
        [float(r.lat), float(r.lon)]
        for r in trip_points.sort_values("obs_idx").itertuples(index=False)
    ]

    return {
        "trip_id": trip_id,
        "gps_points": gps_points,
        "predicted_route": predicted_route,
        "metrics": {
            "gps_observations": len(path),
            "hmm_restarts": len(restarts),
            "road_segments": len(predicted_route),
            "utm_epsg": epsg
        }
    }
