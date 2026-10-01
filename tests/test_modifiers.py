import pytest

from math_model import RANKS, standard_deck
from modifiers import (
    evaluate_death,
    evaluate_hanged_man,
    evaluate_strength,
    next_rank,
)


def test_initial_best_modifier_actions_and_benchmarks():
    deck = standard_deck()
    hanged = evaluate_hanged_man(deck)[0]
    death = evaluate_death(deck)[0]
    strength = evaluate_strength(deck)[0]
    assert hanged.selection == (2, 2)
    assert hanged.predicted_change == pytest.approx(0.205878462, abs=1e-9)
    assert death.selection == (2, 10)
    assert death.predicted_change == pytest.approx(0.171994952, abs=1e-9)
    assert strength.selection == (2, 2)
    assert strength.predicted_change == pytest.approx(0.122230431, abs=1e-9)


@pytest.mark.parametrize(
    "evaluator", [evaluate_hanged_man, evaluate_death, evaluate_strength]
)
def test_displayed_prediction_equals_sum_of_atomic_predictions(evaluator):
    candidate = evaluator(standard_deck())[0]
    atomic_sum = sum(
        step.prediction.trapezoidal_prediction for step in candidate.steps
    )
    assert candidate.predicted_change == pytest.approx(atomic_sum, abs=1e-15)


def test_hanged_man_sequential_state_and_recalculation():
    candidate = evaluate_hanged_man(standard_deck())[0]
    assert len(candidate.steps) == 2
    assert candidate.steps[0].deck_after[2] == 3
    assert candidate.steps[1].deck_before == candidate.steps[0].deck_after
    assert candidate.steps[1].deck_after[2] == 2
    assert sum(candidate.proposed_deck.values()) == 50
    assert (
        candidate.steps[0].prediction.end_sensitivities
        == candidate.steps[1].prediction.start_sensitivities
    )


def test_strength_sequential_state_and_ace_wrap():
    candidate = evaluate_strength(standard_deck())[0]
    assert candidate.steps[0].deck_after[2] == 3
    assert candidate.steps[0].deck_after[3] == 5
    assert candidate.steps[1].deck_before == candidate.steps[0].deck_after
    assert candidate.proposed_deck[2] == 2
    assert candidate.proposed_deck[3] == 6
    assert next_rank(14) == 2


def test_death_is_atomic_preserves_n_and_exposes_contributions():
    candidate = evaluate_death(standard_deck())[0]
    assert len(candidate.steps) == 1
    assert candidate.after_metrics.deck_size == candidate.before_metrics.deck_size
    assert candidate.removal_contribution + candidate.addition_contribution == pytest.approx(
        candidate.predicted_change
    )


def test_legal_action_generation_and_same_rank_copy_requirement():
    deck = {rank: 0 for rank in RANKS}
    deck[2] = 1
    deck[3] = 5
    deck[10] = 1
    hanged = evaluate_hanged_man(deck)
    assert all(candidate.selection != (2, 2) for candidate in hanged)
    strength = evaluate_strength(deck)
    assert all(candidate.selection != (2, 2) for candidate in strength)
    assert any(candidate.selection == (2, 3) for candidate in strength)


def test_candidate_order_is_deterministic():
    first = [candidate.selection for candidate in evaluate_strength(standard_deck())]
    second = [candidate.selection for candidate in evaluate_strength(standard_deck())]
    assert first == second
