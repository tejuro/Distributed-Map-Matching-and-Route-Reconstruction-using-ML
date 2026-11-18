import pytest

from hmm_viterbi.hmm import HiddenMarkovModel
from hmm_viterbi.viterbi import viterbi_decode


def test_connected_road_sequence():
    hmm = HiddenMarkovModel()

    roads = {
        1: (10, 20),
        2: (20, 30),
        3: (30, 40),
        4: (50, 60),
    }

    emissions = [
        {1: 0.9, 4: 0.1},
        {2: 0.8, 4: 0.2},
        {3: 0.9, 4: 0.1},
    ]

    def transition(previous, current, index):
        return hmm.calculate_transition_probability(
            previous, current, roads, emissions[index].keys()
        )

    assert viterbi_decode(emissions, transition) == [1, 2, 3]


def test_emission_normalization():
    hmm = HiddenMarkovModel()
    probabilities = hmm.calculate_emission_probability({
        1: 0.8,
        2: 0.2,
    })

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[1] == pytest.approx(0.8)


def test_disconnected_transition():
    hmm = HiddenMarkovModel()

    roads = {
        1: (10, 20),
        2: (50, 60),
    }

    probability = hmm.calculate_transition_probability(
        1, 2, roads, [2]
    )

    assert probability == 0.0


def test_empty_emissions():
    assert viterbi_decode([], lambda previous, current, index: 1.0) == []


def test_invalid_probability():
    hmm = HiddenMarkovModel()

    with pytest.raises(ValueError):
        hmm.calculate_emission_probability({1: -0.5})
