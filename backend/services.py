from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_FILES = [
    PROJECT_ROOT / "Data" / "processed" / "pairs_student_test_features.csv",
    PROJECT_ROOT / "data" / "processed" / "pairs_student_test_features.csv",
]


def get_available_trips():
    for path in CANDIDATE_FILES:
        if path.is_file():
            df = pd.read_csv(path, usecols=["trip_id"], dtype={"trip_id": str})
            return sorted(df["trip_id"].dropna().unique().tolist())

    return []
