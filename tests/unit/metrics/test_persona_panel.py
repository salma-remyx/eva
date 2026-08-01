"""Tests for PersonaPanelJudgeMetric (ESPP Mode 2 adapted port)."""

import json
from unittest.mock import AsyncMock

import pytest

import eva.metrics.experience  # noqa: F401  -- exercises the registry wiring (non-new module)
from eva.metrics.experience.persona_panel import (
    DEFAULT_PERSONAS,
    PersonaPanelJudgeMetric,
    PersonaProfile,
    _Panelist,
)
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Hi, I need help rebooking my flight."},
    {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation number?"},
    {"turn_id": 2, "role": "user", "content": "ABC123"},
    {"turn_id": 2, "type": "tool_call", "tool_name": "get_reservation", "parameters": {"code": "ABC123"}},
    {"turn_id": 2, "type": "tool_response", "tool_name": "get_reservation", "tool_response": {"status": "ok"}},
    {"turn_id": 2, "role": "assistant", "content": "Found it. You're rebooked for March 25th."},
    {"turn_id": 3, "role": "user", "content": "Great, thanks!"},
    {"turn_id": 3, "role": "assistant", "content": "Anything else I can help with?"},
]


def _resp(rating, explanation, evidence):
    return json.dumps({"rating": rating, "explanation": explanation, "evidence": evidence})


@pytest.fixture
def metric():
    return make_judge_metric(PersonaPanelJudgeMetric)


def test_registered_via_experience_package():
    """Importing eva.metrics.experience wires the metric into the global registry.

    The metric is opt-in (heavier than a single judge), so it is resolvable by
    name for explicit ``--metrics persona_panel`` selection but kept out of the
    default metrics list.
    """
    registry = get_global_registry()
    assert registry.get("persona_panel") is PersonaPanelJudgeMetric
    assert "persona_panel" not in registry.list_metrics()
    assert len(DEFAULT_PERSONAS) == 3


@pytest.mark.asyncio
async def test_compute_aggregates_panel_and_surfaces_baseline(metric):
    """Three personas rate independently; composite + single-judge baseline are produced."""
    metric.llm_client.generate_text = AsyncMock(
        side_effect=[
            (_resp(5, "The agent helped rebook the flight and confirmed the date.", "rebooked for March 25"), None),
            (
                _resp(
                    2, "The agent failed to verify the confirmation and assumed the date.", "never verified reservation"
                ),
                None,
            ),
            (_resp(4, "Resolved the rebooking with concise direct answers.", "quick flight rebooking"), None),
        ]
    )
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    # Baseline is the mean of the *initial* ratings: (5 + 2 + 4) / 3 = 3.667 -> normalized 0.6667
    assert result.details["single_judge_baseline_normalized"] == pytest.approx((11 / 3 - 1) / 4, abs=0.001)
    # Composite stays inside the convex hull of the initial ratings [2, 5].
    assert 2.0 <= result.score <= 5.0
    assert 0.25 <= result.normalized_score <= 1.0
    assert result.details["num_panelists"] == 3
    assert result.details["panel_dispersion"] >= 0.0
    # One sub-metric per persona, carrying each panelist's individual view.
    assert set(result.sub_metrics) == {
        "persona_panel.helpful_score",
        "persona_panel.critical_score",
        "persona_panel.efficient_score",
    }
    initials = sorted(sm.details["initial_rating"] for sm in result.sub_metrics.values())
    assert initials == [2.0, 4.0, 5.0]


@pytest.mark.asyncio
async def test_compute_drops_unparseable_panelist(metric):
    """A panelist returning no rating is dropped; the rest still produce a composite."""
    metric.llm_client.generate_text = AsyncMock(
        side_effect=[
            (_resp(5, "helpful rebooking of the flight", "rebooked flight"), None),
            ("not json at all", None),
            (_resp(4, "concise direct rebooking answers", "quick rebooking"), None),
        ]
    )
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["num_panelists"] == 2


@pytest.mark.asyncio
async def test_compute_all_unparseable_returns_error(metric):
    metric.llm_client.generate_text = AsyncMock(side_effect=[("garbage", None), ("garbage", None), ("garbage", None)])
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error == "No panelist produced a valid rating"
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_no_transcript_returns_error(metric):
    context = make_metric_context(conversation_trace=[])
    result = await metric.compute(context)
    assert result.error == "No transcript available"


def test_bounded_confidence_converges_close_and_gates_out_far(metric):
    """Semantically-close panelists converge; a disjoint panelist is gated out and holds its ground."""
    shared = {"agent", "helpful", "rebooked", "flight", "date"}
    far_tokens = {"unrelated", "vocabulary", "disjoint"}
    a = _Panelist(PersonaProfile("a", "", 1.0, 1.0), 5.0, "", "", 5.0, tokens=shared)
    b = _Panelist(PersonaProfile("b", "", 1.0, 1.0), 2.0, "", "", 2.0, tokens=shared)
    c = _Panelist(PersonaProfile("c", "", 1.0, 1.0), 4.0, "", "", 4.0, tokens=far_tokens)

    metric._exchange([a, b, c])  # default delphi_rounds converges a<->b toward their mean 3.5

    assert a.rating == pytest.approx(3.5, abs=0.05)
    assert b.rating == pytest.approx(3.5, abs=0.05)
    # c shares no vocabulary with a/b -> gated out -> never revised.
    assert c.rating == 4.0


def test_low_openness_moves_less(metric):
    """A panelist with lower openness revises less toward the neighborhood."""
    single = PersonaPanelJudgeMetric(config={"delphi_rounds": 1})
    shared = {"agent", "helpful", "rebooked", "flight"}
    high = _Panelist(PersonaProfile("high", "", 1.0, 1.0), 5.0, "", "", 5.0, tokens=shared)
    low = _Panelist(PersonaProfile("low", "", 0.5, 1.0), 1.0, "", "", 1.0, tokens=shared)

    single._exchange([high, low])

    # One step: high -> 3.0 (moved 2.0), low -> 2.0 (moved 1.0).
    assert abs(high.rating - high.initial_rating) > abs(low.rating - low.initial_rating)


def test_aggregate_is_weighted_mean_within_range(metric):
    """Delphi composite is a weighted mean of final ratings -> inside [min, max]."""
    panelists = [
        _Panelist(PersonaProfile("a", "", 0.5, 1.0), 2.0, "", "", 2.0),
        _Panelist(PersonaProfile("b", "", 0.5, 2.0), 4.0, "", "", 4.0),
        _Panelist(PersonaProfile("c", "", 0.5, 1.0), 5.0, "", "", 5.0),
    ]
    composite, weights = metric._aggregate(panelists)

    assert 2.0 <= composite <= 5.0
    assert sum(weights.values()) == pytest.approx(1.0)
    # Higher-expertise panelist (b) carries more weight than equal-expertise peers.
    assert weights["b"] > weights["a"]
