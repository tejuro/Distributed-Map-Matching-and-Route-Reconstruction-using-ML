
"""
Hidden Markov Model (HMM) for PathSnap.

This module handles:
1. Emission probabilities
2. Transition probabilities
3. Directed road connectivity
4. GPS-to-road matching likelihood
5. Disconnected road transitions
"""

import math


class HiddenMarkovModel:

    def __init__(self, distance_sigma=20.0, heading_sigma=45.0):
        """
        Initialize the Hidden Markov Model.

        distance_sigma: GPS distance tolerance in metres.
        heading_sigma: Heading difference tolerance in degrees.
        """

        if distance_sigma <= 0 or heading_sigma <= 0:
            raise ValueError("Sigma values must be positive.")

        self.distance_sigma = distance_sigma
        self.heading_sigma = heading_sigma

    # --------------------------------------------------
    # 1. CLASSIFIER-BASED EMISSION PROBABILITY
    # --------------------------------------------------

    def calculate_emission_probability(self, candidate_probabilities):
        """
        Normalize classifier probabilities for candidate roads.

        Args:
            candidate_probabilities:
                {arc_id: probability}

        Returns:
            Normalized emission probabilities.
        """

        if not candidate_probabilities:
            raise ValueError("Candidate probabilities cannot be empty.")

        probabilities = {}

        for arc_id, probability in candidate_probabilities.items():

            probability = float(probability)

            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(
                    f"Invalid probability for road {arc_id}."
                )

            probabilities[int(arc_id)] = probability

        total = sum(probabilities.values())

        if total <= 0:
            raise ValueError("Total probability must be positive.")

        return {
            arc_id: probability / total
            for arc_id, probability in probabilities.items()
        }

    # --------------------------------------------------
    # 2. FEATURE-BASED EMISSION PROBABILITY
    # --------------------------------------------------

    def calculate_emission_from_features(self, candidate_rows):
        """
        Calculate emission probabilities using GPS-to-road
        distance and heading difference.

        Args:
            candidate_rows:
                List of dictionaries containing:
                arc_id, dist_m, heading_diff_deg

        Returns:
            {arc_id: normalized_emission_probability}
        """

        if not candidate_rows:
            raise ValueError("Candidate rows cannot be empty.")

        scores = {}

        for row in candidate_rows:

            arc_id = int(row["arc_id"])
            distance = float(row["dist_m"])

            if not math.isfinite(distance) or distance < 0:
                raise ValueError("Invalid GPS-to-road distance.")

            # Gaussian distance likelihood
            distance_score = math.exp(
                -0.5 * (distance / self.distance_sigma) ** 2
            )

            heading = row.get("heading_diff_deg")

            # Missing heading: use distance information only
            if heading is None:
                heading_score = 1.0
            else:
                heading = float(heading)

                if not math.isfinite(heading):
                    heading_score = 1.0
                else:
                    # Circular angular difference
                    heading = abs(heading) % 360
                    heading = min(heading, 360 - heading)

                    heading_score = math.exp(
                        -0.5 * (heading / self.heading_sigma) ** 2
                    )

            scores[arc_id] = distance_score * heading_score

        total = sum(scores.values())

        if total <= 0 or not math.isfinite(total):
            raise ValueError("Unable to calculate emission probabilities.")

        return {
            arc_id: score / total
            for arc_id, score in scores.items()
        }

    # --------------------------------------------------
    # 3. VALIDATE ROAD NETWORK
    # --------------------------------------------------

    @staticmethod
    def validate_road_network(road_network):
        """
        Validate directed road-network representation.

        Format:
            {
                arc_id: (from_node, to_node)
            }
        """

        if not road_network:
            raise ValueError("Road network cannot be empty.")

        for arc_id, nodes in road_network.items():

            if not isinstance(nodes, (tuple, list)) or len(nodes) != 2:
                raise ValueError(
                    f"Invalid road-network entry for arc {arc_id}."
                )

            if nodes[0] is None or nodes[1] is None:
                raise ValueError(
                    f"Missing node information for arc {arc_id}."
                )

    # --------------------------------------------------
    # 4. TRANSITION PROBABILITY
    # --------------------------------------------------

    def calculate_transition_probability(
        self,
        previous_arc_id,
        current_arc_id,
        road_network,
        candidate_next_arcs=None
    ):
        """
        Calculate normalized transition probability
        using directed road connectivity.

        A transition is allowed when:
        1. The vehicle stays on the same arc.
        2. The previous arc's destination node equals
           the current arc's starting node.

        Disconnected transitions receive probability zero.

        candidate_next_arcs:
            Candidate arc IDs available at the next GPS observation.

        Returns:
            Transition probability between 0 and 1.
        """

        self.validate_road_network(road_network)

        if previous_arc_id not in road_network:
            raise ValueError(
                f"Previous arc {previous_arc_id} not found."
            )

        if current_arc_id not in road_network:
            raise ValueError(
                f"Current arc {current_arc_id} not found."
            )

        previous_from, previous_to = road_network[previous_arc_id]

        # Candidate states for the next observation
        if candidate_next_arcs is None:
            next_candidates = set(road_network.keys())
        else:
            next_candidates = set(candidate_next_arcs)

        # Find legal next road segments
        valid_next_arcs = {
            arc_id
            for arc_id in next_candidates
            if arc_id in road_network
            and road_network[arc_id][0] == previous_to
        }

        # Staying on the same road is also allowed
        if previous_arc_id in next_candidates:
            valid_next_arcs.add(previous_arc_id)

        # No legal transition
        if not valid_next_arcs:
            return 0.0

        # Disconnected transition
        if current_arc_id not in valid_next_arcs:
            return 0.0

        # Uniform probability across legal candidate transitions
        return 1.0 / len(valid_next_arcs)
