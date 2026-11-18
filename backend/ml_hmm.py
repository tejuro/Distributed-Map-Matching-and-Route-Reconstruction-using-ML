from collections import defaultdict

from hmm_viterbi.hmm import HiddenMarkovModel
from hmm_viterbi.viterbi import viterbi_decode_with_restarts


def decode_candidate_scores(candidate_rows, road_network):
    """
    Convert candidate-road classifier scores into a road sequence.

    Each row requires:
        obs_idx, arc_id, probability

    road_network:
        {arc_id: (from_node, to_node)}

    Returns:
        predicted_arcs, restart_indices
    """
    if not candidate_rows:
        raise ValueError("Candidate rows cannot be empty")

    hmm = HiddenMarkovModel()
    grouped = defaultdict(dict)

    for row in candidate_rows:
        obs_idx = int(row["obs_idx"])
        arc_id = int(row["arc_id"])
        probability = float(row["probability"])

        if arc_id in grouped[obs_idx]:
            raise ValueError("Duplicate candidate arc at observation")

        grouped[obs_idx][arc_id] = probability

    emissions = [
        hmm.calculate_emission_probability(grouped[idx])
        for idx in sorted(grouped)
    ]

    def transition(previous, current, index):
        return hmm.calculate_transition_probability(
            previous,
            current,
            road_network,
            emissions[index].keys()
        )

    return viterbi_decode_with_restarts(emissions, transition)
