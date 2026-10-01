"""Round evaluation, recommendation, and reversible application state."""

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from math_model import RANKS, Deck, calculate_metrics, standard_deck, validate_deck
from modifiers import (
    MODIFIER_ORDER,
    ModifierCandidate,
    add_best_margin,
    evaluate_death,
    evaluate_hanged_man,
    evaluate_strength,
)


@dataclass(frozen=True)
class RoundEvaluation:
    """All action rankings and the three best modifier candidates."""

    rankings: Dict[str, Tuple[ModifierCandidate, ...]]
    best_by_modifier: Dict[str, Optional[ModifierCandidate]]
    recommended: Optional[ModifierCandidate]
    recommendation_margin: Optional[float]
    tie_break_explanation: str


@dataclass(frozen=True)
class HistoryEntry:
    """One committed modifier-round record."""

    round_number: int
    modifier: str
    card_changes: str
    predicted_change: float
    actual_change: float
    deck_size: int
    total_hands: int
    expected_full_house: float
    expected_straight: float
    expected_total: float
    deck: Deck


@dataclass(frozen=True)
class StateSnapshot:
    """Minimal reversible state captured before a modifier is applied."""

    deck: Deck
    round_number: int
    history: Tuple[HistoryEntry, ...]


@dataclass(frozen=True)
class AppState:
    """Framework-independent application state used by Streamlit and tests."""

    deck: Deck
    initial_deck: Deck
    round_number: int
    history: Tuple[HistoryEntry, ...]
    snapshots: Tuple[StateSnapshot, ...]
    pending: Optional[RoundEvaluation]


def recommend_candidate(
    candidates: Iterable[ModifierCandidate],
) -> Tuple[Optional[ModifierCandidate], Optional[float], str]:
    """Choose solely by predicted sensitivity gain with deterministic ties.

    Exact ``actual_change`` values are intentionally absent from the sort key.
    """

    modifier_priority = {name: index for index, name in enumerate(MODIFIER_ORDER)}
    candidate_list = list(candidates)
    if not candidate_list:
        return None, None, "No legal modifier candidate exists."
    ranked = sorted(
        candidate_list,
        key=lambda candidate: (
            -candidate.predicted_change,
            modifier_priority.get(candidate.modifier, len(MODIFIER_ORDER)),
            candidate.selection,
        ),
    )
    if len(ranked) == 1:
        return ranked[0], None, "Only one modifier has a legal candidate."
    margin = ranked[0].predicted_change - ranked[1].predicted_change
    note = ""
    if abs(margin) <= 1e-12:
        note = (
            "The best modifier predictions tie numerically. The fixed order "
            "Hanged Man, Death, Strength, then lower rank tuples breaks the tie."
        )
    return ranked[0], margin, note


def evaluate_round(deck: Mapping[int, int]) -> RoundEvaluation:
    """Evaluate all three modifiers from the same committed deck."""

    counts = validate_deck(deck)
    rankings: Dict[str, Tuple[ModifierCandidate, ...]] = {
        "Hanged Man": tuple(evaluate_hanged_man(counts)),
        "Death": tuple(evaluate_death(counts)),
        "Strength": tuple(evaluate_strength(counts)),
    }
    best = {
        modifier: add_best_margin(rankings[modifier])
        for modifier in MODIFIER_ORDER
    }
    legal_best = [candidate for candidate in best.values() if candidate is not None]
    recommended, margin, tie_note = recommend_candidate(legal_best)
    return RoundEvaluation(
        rankings=rankings,
        best_by_modifier=best,
        recommended=recommended,
        recommendation_margin=margin,
        tie_break_explanation=tie_note,
    )


def new_state(deck: Optional[Mapping[int, int]] = None) -> AppState:
    """Create a clean state from a standard or custom starting deck."""

    counts = validate_deck(standard_deck() if deck is None else deck)
    return AppState(
        deck=dict(counts),
        initial_deck=dict(counts),
        round_number=0,
        history=(),
        snapshots=(),
        pending=None,
    )


def evaluate_state(state: AppState) -> AppState:
    """Stage a round evaluation without mutating the committed deck."""

    return replace(state, pending=evaluate_round(state.deck))


def _candidate_is_pending(state: AppState, candidate: ModifierCandidate) -> bool:
    if state.pending is None:
        return False
    return any(
        best == candidate
        for best in state.pending.best_by_modifier.values()
        if best is not None
    )


def apply_candidate(state: AppState, candidate: ModifierCandidate) -> AppState:
    """Commit one staged best-per-modifier candidate and record history."""

    if not _candidate_is_pending(state, candidate):
        raise ValueError("Candidate is not part of the current staged evaluation")
    if candidate.steps[0].deck_before != state.deck:
        raise ValueError("Candidate was evaluated from a different committed deck")

    snapshot = StateSnapshot(
        deck=dict(state.deck),
        round_number=state.round_number,
        history=state.history,
    )
    metrics = candidate.after_metrics
    new_round = state.round_number + 1
    entry = HistoryEntry(
        round_number=new_round,
        modifier=candidate.modifier,
        card_changes=candidate.card_changes,
        predicted_change=candidate.predicted_change,
        actual_change=candidate.actual_change,
        deck_size=metrics.deck_size,
        total_hands=metrics.total_hands,
        expected_full_house=metrics.expected_full_house,
        expected_straight=metrics.expected_straight,
        expected_total=metrics.expected_total,
        deck=dict(candidate.proposed_deck),
    )
    return AppState(
        deck=dict(candidate.proposed_deck),
        initial_deck=dict(state.initial_deck),
        round_number=new_round,
        history=state.history + (entry,),
        snapshots=state.snapshots + (snapshot,),
        pending=None,
    )


def apply_recommended(state: AppState) -> AppState:
    """Commit the current recommendation."""

    if state.pending is None or state.pending.recommended is None:
        raise ValueError("Evaluate a modifier round before applying a recommendation")
    return apply_candidate(state, state.pending.recommended)


def apply_selected_modifier(state: AppState, modifier: str) -> AppState:
    """Commit the staged best candidate for a user-selected modifier."""

    if state.pending is None:
        raise ValueError("Evaluate a modifier round before applying a candidate")
    candidate = state.pending.best_by_modifier.get(modifier)
    if candidate is None:
        raise ValueError("No legal candidate exists for {}".format(modifier))
    return apply_candidate(state, candidate)


def undo_last_modifier(state: AppState) -> AppState:
    """Restore the exact state that preceded the last committed modifier."""

    if not state.snapshots:
        return state
    snapshot = state.snapshots[-1]
    return AppState(
        deck=dict(snapshot.deck),
        initial_deck=dict(state.initial_deck),
        round_number=snapshot.round_number,
        history=snapshot.history,
        snapshots=state.snapshots[:-1],
        pending=None,
    )


def reset_state(
    state: AppState, deck: Optional[Mapping[int, int]] = None
) -> AppState:
    """Reset to the standard deck, or explicitly begin from a custom deck."""

    del state
    return new_state(standard_deck() if deck is None else deck)


def run_automatic_rounds(state: AppState, rounds: int) -> AppState:
    """Evaluate and commit recommendations for a bounded number of rounds."""

    if rounds < 1:
        raise ValueError("Automatic round count must be at least 1")
    current = state
    for _ in range(rounds):
        current = evaluate_state(current)
        if current.pending is None or current.pending.recommended is None:
            return replace(current, pending=None)
        current = apply_recommended(current)
    return current


def state_metrics(state: AppState):
    """Convenience wrapper used by the UI."""

    return calculate_metrics(state.deck)
