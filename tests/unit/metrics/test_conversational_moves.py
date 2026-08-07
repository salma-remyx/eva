"""Tests for ConversationalMovesJudgeMetric.

Covers both the per-turn judge behavior and the integration contract: that the
metric is discoverable through the existing global registry path (the wiring
that ``eva.metrics.experience`` performs at import time), not just callable in
isolation.
"""

import json
from unittest.mock import AsyncMock

import pytest

from eva.metrics.experience.conversational_moves import CONVERSATIONAL_MOVES, ConversationalMovesJudgeMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Hi, I need help rebooking my flight."},
    {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation number?"},
    {"turn_id": 2, "role": "user", "content": "ABC123."},
    {"turn_id": 2, "role": "assistant", "content": "Found your reservation. What date works?"},
    {"turn_id": 3, "role": "user", "content": "March 25th please."},
    {"turn_id": 3, "role": "assistant", "content": "Done, you're rebooked for March 25th."},
]


def test_metric_registered_in_global_registry():
    """The metric is wired into the existing registry path and resolvable by name.

    This exercises the integration surface (the experience package import + the
    @register_metric hook) rather than the class in isolation: the global
    registry is the path MetricsRunner uses to instantiate metrics by name.
    """
    registry = get_global_registry()
    assert registry.get("conversational_moves") is ConversationalMovesJudgeMetric
    assert "conversational_moves" in registry.list_metrics()
    instance = registry.create("conversational_moves")
    assert isinstance(instance, ConversationalMovesJudgeMetric)


@pytest.fixture
def metric():
    return make_judge_metric(ConversationalMovesJudgeMetric)


@pytest.mark.asyncio
async def test_all_turns_rated_with_moves(metric):
    """Each assistant turn gets a rating plus a classified move."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "move": "querying", "rating": 3, "explanation": "asks for needed detail"},
            {"turn_id": 2, "move": "querying", "rating": 2, "explanation": "asks for date"},
            {"turn_id": 3, "move": "replying", "rating": 3, "explanation": "delivers result"},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 3, 2: 2, 3: 3}
    assert result.details["per_turn_move"] == {1: "querying", 2: "querying", 3: "replying"}
    assert result.details["num_turns"] == 3
    assert result.details["num_evaluated"] == 3
    # mean of [3, 2, 3] = 2.667; normalized [1.0, 0.5, 1.0] -> 0.833
    assert result.score == pytest.approx(2.667, abs=0.001)
    assert result.normalized_score == pytest.approx(0.833, abs=0.001)


@pytest.mark.asyncio
async def test_move_distribution_sub_metrics(metric):
    """One sub-metric per move type, each giving the share of rated turns."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "move": "querying", "rating": 3, "explanation": "asks"},
            {"turn_id": 2, "move": "querying", "rating": 3, "explanation": "asks"},
            {"turn_id": 3, "move": "replying", "rating": 3, "explanation": "answers"},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.sub_metrics is not None
    expected_keys = {f"{move}_rate" for move in CONVERSATIONAL_MOVES}
    assert set(result.sub_metrics.keys()) == expected_keys
    # 2 of 3 rated turns are querying
    querying = result.sub_metrics["querying_rate"]
    assert querying.name == "conversational_moves.querying_rate"
    assert querying.score == pytest.approx(2 / 3, abs=0.002)
    assert querying.details["count"] == 2
    assert querying.details["num_rated"] == 3
    assert set(querying.details["turn_ids"]) == {1, 2}
    # moves never used still emitted at rate 0
    assert result.sub_metrics["acknowledging_rate"].score == 0.0


@pytest.mark.asyncio
async def test_unknown_move_coerced_to_other(metric):
    """A move label outside the taxonomy is coerced to the catch-all 'other'."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "move": "querying", "rating": 2, "explanation": "ok"},
            {"turn_id": 2, "move": "chitchat", "rating": 1, "explanation": "off-task"},
            {"turn_id": 3, "move": "replying", "rating": 3, "explanation": "answers"},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_move"][2] == "other"
    assert result.sub_metrics["other_rate"].score == pytest.approx(1 / 3, abs=0.002)


@pytest.mark.asyncio
async def test_null_rating_excluded_from_aggregation(metric):
    """User-only turns (rating null) are stored but excluded from the score and distribution."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "move": "other", "rating": None, "explanation": "user-only turn"},
            {"turn_id": 2, "move": "querying", "rating": 3, "explanation": "asks"},
            {"turn_id": 3, "move": "replying", "rating": 2, "explanation": "answers"},
        ]
    )

    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: None, 2: 3, 3: 2}
    assert result.details["num_evaluated"] == 2
    # distribution denominator is rated turns only (2), so querying_rate = 1/2
    assert result.sub_metrics["querying_rate"].details["num_rated"] == 2
    assert result.sub_metrics["querying_rate"].score == pytest.approx(0.5, abs=0.002)


@pytest.mark.asyncio
async def test_no_response_from_judge(metric):
    """None response from LLM returns a parse error."""
    metric.llm_client.generate_text = AsyncMock(return_value=(None, None))
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error == "Failed to parse judge response"
    assert result.score == 0.0
