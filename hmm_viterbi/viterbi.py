
"""
Viterbi Algorithm for PathSnap.

Finds the most likely sequence of road segments
using emission and transition probabilities.
"""

import math


def viterbi_decode(emissions, transition_probability):
    """
    Decode the most likely road sequence.

    Args:
        emissions:
            List of dictionaries:
            [{arc_id: emission_probability}, ...]

        transition_probability:
            Function accepting:
            (previous_arc, current_arc, observation_index)

    Returns:
        Most likely sequence of road arc IDs.
    """

    if not emissions:
        return []

    # Validate emissions
    for index, observation in enumerate(emissions):

        if not observation:
            raise ValueError(
                f"No candidate roads at observation {index}."
            )

        for arc_id, probability in observation.items():

            probability = float(probability)

            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(
                    f"Invalid emission probability at observation {index}."
                )

    # Initialize first observation
    scores = {
        arc_id: (
            math.log(probability)
            if probability > 0 else float("-inf")
        )
        for arc_id, probability in emissions[0].items()
    }

    backpointers = []

    # Process subsequent observations
    for index in range(1, len(emissions)):

        current_emissions = emissions[index]

        new_scores = {}
        pointers = {}

        for current_arc, emission in current_emissions.items():

            best_score = float("-inf")
            best_previous_arc = None

            for previous_arc, previous_score in scores.items():

                transition = transition_probability(
                    previous_arc,
                    current_arc,
                    index
                )

                transition = float(transition)

                if not math.isfinite(transition) or not 0 <= transition <= 1:
                    raise ValueError("Invalid transition probability.")

                if (
                    previous_score == float("-inf")
                    or transition == 0
                    or emission == 0
                ):
                    candidate_score = float("-inf")
                else:
                    candidate_score = (
                        previous_score
                        + math.log(transition)
                        + math.log(float(emission))
                    )

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous_arc = previous_arc

            new_scores[current_arc] = best_score
            pointers[current_arc] = best_previous_arc

        if all(score == float("-inf") for score in new_scores.values()):
            raise ValueError(
                f"No valid road transition at observation {index}."
            )

        scores = new_scores
        backpointers.append(pointers)

    # Select the best final road
    best_final_arc = max(scores, key=scores.get)

    # Backtrack
    best_path = [best_final_arc]

    for pointers in reversed(backpointers):

        previous_arc = pointers[best_path[-1]]

        if previous_arc is None:
            raise ValueError("Unable to reconstruct road sequence.")

        best_path.append(previous_arc)

    best_path.reverse()

    return best_path

def viterbi_decode_with_restarts(emissions, transition_probability):
    """
    Decode road sequences and restart when no valid
    transition exists.
    """

    if not emissions:
        return [], []

    predicted_path = []
    restart_indices = []
    start = 0

    while start < len(emissions):

        remaining = emissions[start:]

        def local_transition(previous, current, index):
            return transition_probability(
                previous,
                current,
                start + index
            )

        try:
            segment_path = viterbi_decode(
                remaining,
                local_transition
            )

            predicted_path.extend(segment_path)
            break

        except ValueError as error:

            prefix = "No valid road transition at observation "

            if not str(error).startswith(prefix):
                raise

            local_index = int(
                str(error)[len(prefix):].rstrip(".")
            )

            if local_index <= 0 or local_index >= len(remaining):
                raise

            valid_path = viterbi_decode(
                remaining[:local_index],
                local_transition
            )

            predicted_path.extend(valid_path)

            start += local_index
            restart_indices.append(start)

    return predicted_path, restart_indices
