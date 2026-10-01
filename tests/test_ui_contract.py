import base64
from pathlib import Path

import numpy as np
import plotly.io as pio
import pytest
from streamlit.testing.v1 import AppTest

from math_model import RANK_LABELS, RANKS, calculate_metrics, standard_deck


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _luminance(hex_colour):
    channels = [int(hex_colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first, second):
    light, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _figure(element):
    return pio.from_json(element.proto.spec)


def _values(value):
    if isinstance(value, dict) and "bdata" in value:
        raw = base64.b64decode(value["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).tolist()
    return list(value)


def test_dark_cards_rank_order_and_baseline_history_contract():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()
    assert not app.exception

    css = app.markdown[0].value
    assert "--panel: #151c2b" in css
    assert "--text: #f8fafc" in css
    assert "background: #f6f8fb" not in css
    assert _contrast("#f8fafc", "#151c2b") >= 4.5
    assert _contrast("#4ade80", "#101725") >= 4.5
    assert _contrast("#60a5fa", "#101725") >= 4.5

    plots = app.get("plotly_chart")
    assert len(plots) == 4
    deck_figure = _figure(plots[0])
    expected_order = [RANK_LABELS[rank] for rank in RANKS]
    assert list(deck_figure.data[0].x) == expected_order
    assert list(deck_figure.layout.xaxis.categoryarray) == expected_order
    assert deck_figure.data[0].text is not None
    assert deck_figure.data[0].textposition == "outside"

    baseline = calculate_metrics(standard_deck())
    expected_values = [
        baseline.expected_total,
        baseline.expected_straight,
        baseline.expected_full_house,
    ]
    for element, expected in zip(plots[1:], expected_values):
        figure = _figure(element)
        assert _values(figure.data[0].x) == [0]
        assert _values(figure.data[0].y) == pytest.approx([expected])
        assert "markers" in figure.data[0].mode

    assert len(app.dataframe) == 1
    baseline_row = app.dataframe[0].value.iloc[0]
    assert baseline_row["Round"] == 0
    assert baseline_row["New E"] == "1.404"


def test_evaluated_page_uses_one_three_decimal_sequential_value_everywhere():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()
    app.button(key="evaluate_round").click().run(timeout=45)
    assert not app.exception

    candidate = app.session_state["app_state"].pending.recommended
    atomic_sum = sum(
        step.prediction.trapezoidal_prediction for step in candidate.steps
    )
    assert candidate.predicted_change == pytest.approx(atomic_sum)

    recommendation = next(
        item.value for item in app.success if "Recommended: Hanged Man" in item.value
    )
    assert "+0.206" in recommendation

    candidate_markup = "\n".join(
        item.value
        for item in app.markdown
        if item.value.strip().startswith('<div class="candidate-card">')
    )
    assert "Sensitivity-predicted change" in candidate_markup
    assert "+0.206" in candidate_markup
    assert "margin" not in candidate_markup.lower()
    assert "confidence" not in candidate_markup.lower()

    hanged_total = next(
        item.value
        for item in app.latex
        if "Delta E" in item.value and "Hanged" in item.value
    )
    assert r"\left(+0.088\right) + \left(+0.117\right)=+0.206" in hanged_total

    death_step = next(
        item.value
        for item in app.latex
        if "M_{10}" in item.value and "Delta E" in item.value
    )
    assert r"\\&\quad +" in death_step

    labels = [expander.label for expander in app.expander]
    assert labels[:3] == [
        "Hanged Man — Show sensitivity working",
        "Death — Show sensitivity working",
        "Strength — Show sensitivity working",
    ]
