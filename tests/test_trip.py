
import pandas as pd

from hmm_viterbi.hmm import HiddenMarkovModel
from hmm_viterbi.viterbi import viterbi_decode_with_restarts


def test_real_trip():
    trip_id = 1
    limit = 2000
    gap_threshold = 2.0

    points = pd.read_csv(
        "Data/processed/points_train_features.csv"
    )
    pairs = pd.read_csv(
        "Data/processed/pairs_train_features.csv"
    )
    arcs = pd.read_csv(
        "Data/processed/arcs_train_features.csv"
    )

    # Select GPS observations
    points = points[
        (points["trip_id"] == trip_id)
        & (points["obs_idx"] < limit)
    ].sort_values("obs_idx").copy()

    pairs = pairs[
        (pairs["trip_id"] == trip_id)
        & (pairs["obs_idx"] < limit)
    ]

    arcs = arcs[
        arcs["trip_id"] == trip_id
    ].drop_duplicates("arc_id")

    # Build road network
    road_network = {
        int(row.arc_id): (
            int(row.from_node),
            int(row.to_node)
        )
        for row in arcs.itertuples()
    }

    # Detect GPS gaps and sequence boundaries
    points["time_diff"] = points["t"].diff()

    points["new_sequence"] = (
        (points["time_diff"] > gap_threshold)
        | points["gap_before"].fillna(False)
        | (points["segment_id"] != points["segment_id"].shift())
    )

    points["sequence_id"] = (
        points["new_sequence"].cumsum()
    )

    hmm = HiddenMarkovModel()

    all_predictions = []
    total_connected = 0
    total_disconnected = 0
    total_restarts = 0

    # Process each continuous GPS sequence separately
    for sequence_id, sequence_points in points.groupby("sequence_id"):

        emissions = []
        candidate_sets = []

        for obs_idx in sequence_points["obs_idx"]:

            rows = pairs[
                pairs["obs_idx"] == obs_idx
            ].to_dict("records")

            if not rows:
                continue

            emission = hmm.calculate_emission_from_features(rows)

            emissions.append(emission)
            candidate_sets.append(set(emission.keys()))

        if not emissions:
            continue

        def transition(previous_arc, current_arc, index):
            return hmm.calculate_transition_probability(
                previous_arc,
                current_arc,
                road_network,
                candidate_next_arcs=candidate_sets[index]
            )

        # Decode and restart when transitions are disconnected
        predicted_path, restart_indices = (
            viterbi_decode_with_restarts(
                emissions,
                transition
            )
        )

        all_predictions.extend(predicted_path)
        total_restarts += len(restart_indices)

        print(
            f"Sequence {sequence_id}: "
            f"{len(predicted_path)} observations, "
            f"{len(restart_indices)} disconnected-transition restarts"
        )

        # Check continuity, excluding restart boundaries
        restart_set = set(restart_indices)

        for index, (previous, current) in enumerate(
            zip(predicted_path, predicted_path[1:]),
            start=1
        ):

            if index in restart_set:
                continue

            if (
                previous == current
                or road_network[previous][1]
                == road_network[current][0]
            ):
                total_connected += 1
            else:
                total_disconnected += 1

    print("\n--- GPS and Disconnected Sequence Results ---")
    print("Trip:", trip_id)
    print("Total GPS observations:", len(points))
    print("Number of GPS sequences:", points["sequence_id"].nunique())
    print("Predicted road segments:", len(all_predictions))
    print("Disconnected-transition restarts:", total_restarts)
    print("Connected transitions:", total_connected)
    print("Disconnected transitions:", total_disconnected)


if __name__ == "__main__":
    test_real_trip()
