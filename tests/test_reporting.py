import json

from algorithm import evaluate_state, new_state
from reporting import actions_as_csv, detailed_html_report, history_as_csv, state_as_json


def test_exports_are_readable_and_separate_validation_values():
    state = evaluate_state(new_state())
    payload = json.loads(state_as_json(state))
    assert payload["modifier_round"] == 0
    assert payload["recommended"]["modifier"] == "Hanged Man"
    actions = actions_as_csv(state.pending)
    assert "predicted_sensitivity_change" in actions
    assert "actual_change_validation_only" in actions
    assert "ordered_steps_json" in actions
    report = detailed_html_report(state)
    assert "validation only" in report
    assert "Current sensitivity substitutions" in report
    assert history_as_csv(state.history).startswith("Round,Modifier")
