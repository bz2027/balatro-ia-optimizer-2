"""Legal modifier generation and sensitivity-only candidate ranking."""

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from math_model import RANKS, RANK_LABELS, Deck, DeckMetrics, calculate_metrics, validate_deck
from sensitivity import SensitivityPrediction, trapezoidal_sensitivity_prediction


MODIFIER_ORDER: Tuple[str, ...] = ("Hanged Man", "Death", "Strength")


@dataclass(frozen=True)
class ActionStep:
    """One sequential atomic change and its complete sensitivity working."""

    step_number: int
    description: str
    deck_before: Deck
    deck_after: Deck
    prediction: SensitivityPrediction


@dataclass(frozen=True)
class ModifierCandidate:
    """One complete legal action evaluated from a committed deck."""

    modifier: str
    selection: Tuple[int, ...]
    card_changes: str
    proposed_deck: Deck
    predicted_change: float
    actual_change: float
    before_metrics: DeckMetrics
    after_metrics: DeckMetrics
    steps: Tuple[ActionStep, ...]
    removal_contribution: Optional[float] = None
    addition_contribution: Optional[float] = None
    margin_to_next: Optional[float] = None
    confidence: str = "Not calculated"
    tie_break_explanation: str = ""


def next_rank(rank: int) -> int:
    """Return the isolated Strength progression, including Ace -> 2."""

    if rank not in RANKS:
        raise ValueError("Rank must be between 2 and 14")
    return 2 if rank == 14 else rank + 1


def _single_delta(source: int, target: Optional[int] = None) -> Dict[int, int]:
    delta = {rank: 0 for rank in RANKS}
    delta[source] -= 1
    if target is not None:
        delta[target] += 1
    return delta


def _candidate_sort_key(candidate: ModifierCandidate) -> Tuple[object, ...]:
    """Sort by prediction only, then by a deterministic lower-rank tuple."""

    return (-candidate.predicted_change, candidate.selection)


def rank_candidates(
    candidates: Sequence[ModifierCandidate],
) -> List[ModifierCandidate]:
    """Return candidates ranked without consulting actual expected changes."""

    return sorted(candidates, key=_candidate_sort_key)


def _confidence(best: float, margin: float) -> str:
    scale = max(abs(best), 1e-12)
    if margin <= max(1e-12, 0.01 * scale):
        return "Low - the top sensitivity predictions are very close"
    if margin <= 0.05 * scale:
        return "Medium - the leading margin is modest"
    return "High - the sensitivity prediction has a clear margin"


def add_best_margin(
    ranked: Sequence[ModifierCandidate],
) -> Optional[ModifierCandidate]:
    """Attach the best-versus-runner-up margin to the leading action."""

    if not ranked:
        return None
    if len(ranked) == 1:
        return replace(
            ranked[0],
            margin_to_next=None,
            confidence="Only legal action",
            tie_break_explanation="No runner-up action exists.",
        )
    margin = ranked[0].predicted_change - ranked[1].predicted_change
    tie_note = ""
    if abs(margin) <= 1e-12:
        tie_note = (
            "The predictions tie numerically; the lexicographically lower "
            "ordered rank tuple is selected deterministically."
        )
    return replace(
        ranked[0],
        margin_to_next=margin,
        confidence=_confidence(ranked[0].predicted_change, margin),
        tie_break_explanation=tie_note,
    )


def evaluate_hanged_man(deck: Mapping[int, int]) -> List[ModifierCandidate]:
    """Evaluate every legal ordered two-card deletion sequence."""

    counts = validate_deck(deck)
    if sum(counts.values()) < 7:
        return []
    before_metrics = calculate_metrics(counts)
    candidates: List[ModifierCandidate] = []
    for first_rank in RANKS:
        if counts[first_rank] < 1:
            continue
        first_prediction = trapezoidal_sensitivity_prediction(
            counts, _single_delta(first_rank)
        )
        temporary = first_prediction.end_deck
        for second_rank in RANKS:
            if temporary[second_rank] < 1:
                continue
            second_prediction = trapezoidal_sensitivity_prediction(
                temporary, _single_delta(second_rank)
            )
            final_deck = second_prediction.end_deck
            after_metrics = calculate_metrics(final_deck)
            steps = (
                ActionStep(
                    step_number=1,
                    description="Destroy one {}".format(RANK_LABELS[first_rank]),
                    deck_before=dict(counts),
                    deck_after=dict(temporary),
                    prediction=first_prediction,
                ),
                ActionStep(
                    step_number=2,
                    description="Destroy one {}".format(RANK_LABELS[second_rank]),
                    deck_before=dict(temporary),
                    deck_after=dict(final_deck),
                    prediction=second_prediction,
                ),
            )
            candidates.append(
                ModifierCandidate(
                    modifier="Hanged Man",
                    selection=(first_rank, second_rank),
                    card_changes="Destroy {} then {}".format(
                        RANK_LABELS[first_rank], RANK_LABELS[second_rank]
                    ),
                    proposed_deck=dict(final_deck),
                    predicted_change=(
                        first_prediction.trapezoidal_prediction
                        + second_prediction.trapezoidal_prediction
                    ),
                    actual_change=(
                        after_metrics.expected_total
                        - before_metrics.expected_total
                    ),
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    steps=steps,
                )
            )
    return rank_candidates(candidates)


def evaluate_death(deck: Mapping[int, int]) -> List[ModifierCandidate]:
    """Evaluate every legal ordered source -> existing-target conversion."""

    counts = validate_deck(deck)
    before_metrics = calculate_metrics(counts)
    candidates: List[ModifierCandidate] = []
    for source in RANKS:
        if counts[source] < 1:
            continue
        for target in RANKS:
            if target == source or counts[target] < 1:
                continue
            prediction = trapezoidal_sensitivity_prediction(
                counts, _single_delta(source, target)
            )
            final_deck = prediction.end_deck
            after_metrics = calculate_metrics(final_deck)
            removal = prediction.trapezoidal_contributions[source]
            addition = prediction.trapezoidal_contributions[target]
            step = ActionStep(
                step_number=1,
                description="Convert {} into {}".format(
                    RANK_LABELS[source], RANK_LABELS[target]
                ),
                deck_before=dict(counts),
                deck_after=dict(final_deck),
                prediction=prediction,
            )
            candidates.append(
                ModifierCandidate(
                    modifier="Death",
                    selection=(source, target),
                    card_changes="Convert {} -> {}".format(
                        RANK_LABELS[source], RANK_LABELS[target]
                    ),
                    proposed_deck=dict(final_deck),
                    predicted_change=prediction.trapezoidal_prediction,
                    actual_change=(
                        after_metrics.expected_total
                        - before_metrics.expected_total
                    ),
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    steps=(step,),
                    removal_contribution=removal,
                    addition_contribution=addition,
                )
            )
    return rank_candidates(candidates)


def _legal_two_selected_sources(counts: Mapping[int, int], first: int, second: int) -> bool:
    if first == second:
        return counts[first] >= 2
    return counts[first] >= 1 and counts[second] >= 1


def evaluate_strength(deck: Mapping[int, int]) -> List[ModifierCandidate]:
    """Evaluate every legal ordered pair of selected Strength source cards."""

    counts = validate_deck(deck)
    before_metrics = calculate_metrics(counts)
    candidates: List[ModifierCandidate] = []
    for first_rank in RANKS:
        for second_rank in RANKS:
            if not _legal_two_selected_sources(
                counts, first_rank, second_rank
            ):
                continue
            first_target = next_rank(first_rank)
            first_prediction = trapezoidal_sensitivity_prediction(
                counts, _single_delta(first_rank, first_target)
            )
            temporary = first_prediction.end_deck
            second_target = next_rank(second_rank)
            second_prediction = trapezoidal_sensitivity_prediction(
                temporary, _single_delta(second_rank, second_target)
            )
            final_deck = second_prediction.end_deck
            after_metrics = calculate_metrics(final_deck)
            steps = (
                ActionStep(
                    step_number=1,
                    description="Strengthen {} -> {}".format(
                        RANK_LABELS[first_rank], RANK_LABELS[first_target]
                    ),
                    deck_before=dict(counts),
                    deck_after=dict(temporary),
                    prediction=first_prediction,
                ),
                ActionStep(
                    step_number=2,
                    description="Strengthen {} -> {}".format(
                        RANK_LABELS[second_rank], RANK_LABELS[second_target]
                    ),
                    deck_before=dict(temporary),
                    deck_after=dict(final_deck),
                    prediction=second_prediction,
                ),
            )
            candidates.append(
                ModifierCandidate(
                    modifier="Strength",
                    selection=(first_rank, second_rank),
                    card_changes="Strengthen {} -> {}, then {} -> {}".format(
                        RANK_LABELS[first_rank],
                        RANK_LABELS[first_target],
                        RANK_LABELS[second_rank],
                        RANK_LABELS[second_target],
                    ),
                    proposed_deck=dict(final_deck),
                    predicted_change=(
                        first_prediction.trapezoidal_prediction
                        + second_prediction.trapezoidal_prediction
                    ),
                    actual_change=(
                        after_metrics.expected_total
                        - before_metrics.expected_total
                    ),
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    steps=steps,
                )
            )
    return rank_candidates(candidates)
