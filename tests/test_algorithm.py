from dataclasses import replace

import pytest

from algorithm import (
    apply_recommended,
    apply_selected_modifier,
    evaluate_state,
    new_state,
    recommend_candidate,
    reset_state,
    run_automatic_rounds,
    undo_last_modifier,
)
from math_model import calculate_metrics, standard_deck
from modifiers import evaluate_death, evaluate_hanged_man


def test_first_recommendation_and_apply_pause_progression():
    state = evaluate_state(new_state())
    assert state.round_number == 0
    assert state.deck == standard_deck()
    assert state.pending.recommended.modifier == "Hanged Man"
    assert state.pending.recommended.selection == (2, 2)
    applied = apply_recommended(state)
    assert applied.round_number == 1
    assert applied.pending is None
    assert applied.deck[2] == 2
    assert calculate_metrics(applied.deck).deck_size == 50
    assert calculate_metrics(applied.deck).total_hands == 2_118_760
    atomic_sum = sum(
        step.prediction.trapezoidal_prediction
        for step in state.pending.recommended.steps
    )
    assert state.pending.recommended.predicted_change == pytest.approx(atomic_sum)
    assert applied.history[-1].predicted_change == pytest.approx(atomic_sum)


def test_override_undo_and_reset():
    staged = evaluate_state(new_state())
    death_state = apply_selected_modifier(staged, "Death")
    assert death_state.history[-1].modifier == "Death"
    restored = undo_last_modifier(death_state)
    assert restored.deck == standard_deck()
    assert restored.round_number == 0
    assert restored.history == ()
    reset = reset_state(death_state)
    assert reset.deck == standard_deck()
    assert reset.snapshots == ()


def test_automatic_rounds_are_bounded_and_deterministic():
    first = run_automatic_rounds(new_state(), 2)
    second = run_automatic_rounds(new_state(), 2)
    assert first.round_number == 2
    assert first.deck == second.deck
    assert first.history == second.history


def test_recommendation_uses_prediction_not_actual_change():
    deck = standard_deck()
    hanged = evaluate_hanged_man(deck)[0]
    death = evaluate_death(deck)[0]
    manipulated_hanged = replace(hanged, predicted_change=10.0, actual_change=-999.0)
    manipulated_death = replace(death, predicted_change=9.0, actual_change=999.0)
    recommended, _, _ = recommend_candidate(
        [manipulated_hanged, manipulated_death]
    )
    assert recommended is manipulated_hanged


def test_cannot_apply_before_evaluation():
    with pytest.raises(ValueError):
        apply_recommended(new_state())
