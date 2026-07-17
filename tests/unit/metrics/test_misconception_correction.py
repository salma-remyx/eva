"""Tests for MisconceptionCorrectionMetric.

Integration is exercised through the existing per-turn judge framework:
``PerTurnConversationJudgeMetric.compute`` (from eva.metrics.base, a non-new
module) drives the metric end-to-end with a mocked judge, and the metric
self-registers with the global metric registry on import — the same registry
``MetricsRunner`` resolves metric names from at run time.
"""

import json
from unittest.mock import AsyncMock

import pytest

from eva.metrics.base import PerTurnConversationJudgeMetric
from eva.metrics.experience.misconception_correction import MisconceptionCorrectionMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context

# A multi-turn conversation where a misconception (antibiotics cure the flu) is
# introduced in turn 1, corrected, then re-raised in turn 3 where the assistant
# goes along with it (error propagation), and recovered in turn 4.
SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Can you prescribe antibiotics for my flu?"},
    {
        "turn_id": 1,
        "role": "assistant",
        "content": "Flu is viral, so antibiotics won't help — rest and fluids are recommended.",
    },
    {"turn_id": 2, "role": "user", "content": "But my friend got antibiotics for the flu and felt better."},
    {
        "turn_id": 2,
        "role": "assistant",
        "content": "They likely had a bacterial infection on top of it; antibiotics don't treat viral flu.",
    },
    {"turn_id": 3, "role": "user", "content": "So I should take my leftover antibiotics just in case."},
    {
        "turn_id": 3,
        "role": "assistant",
        "content": "Sure, taking leftover antibiotics just in case is reasonable.",
    },
    {"turn_id": 4, "role": "user", "content": "Really? Even though it's viral?"},
    {
        "turn_id": 4,
        "role": "assistant",
        "content": "You're right to double-check — no, don't take antibiotics for viral flu.",
    },
]


@pytest.fixture
def metric():
    return make_judge_metric(MisconceptionCorrectionMetric, mock_llm=True)


def test_metric_attributes(metric):
    assert metric.name == "misconception_correction"
    assert metric.category == "experience"
    assert metric.rating_scale == (1, 3)
    assert isinstance(metric, PerTurnConversationJudgeMetric)


def test_registers_with_global_registry():
    """Importing the module registers it in the framework's global metric registry."""
    assert get_global_registry().get("misconception_correction") is MisconceptionCorrectionMetric


@pytest.mark.asyncio
async def test_compute_scores_per_turn_and_aggregates(metric):
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "corrected": True, "misconception_present": True, "explanation": "corrected"},
            {"turn_id": 2, "rating": 3, "corrected": True, "misconception_present": True, "explanation": "corrected"},
            {"turn_id": 3, "rating": 1, "corrected": False, "misconception_present": True, "explanation": "propagated"},
            {"turn_id": 4, "rating": 3, "corrected": True, "misconception_present": False, "explanation": "recovered"},
        ]
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))

    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error is None
    assert result.details["per_turn_ratings"] == {1: 3, 2: 3, 3: 1, 4: 3}
    # normalized [1.0, 1.0, 0.0, 1.0] -> mean 0.75
    assert result.normalized_score == pytest.approx(0.75, abs=0.001)
    assert result.details["per_turn_corrected"][3] is False
    assert result.details["per_turn_misconception_present"][1] is True


@pytest.mark.asyncio
async def test_surfaces_across_turn_degradation(metric):
    """First turn holds; later turns drop -> degradation + propagation surfaced."""
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "corrected": True, "misconception_present": True, "explanation": "strong"},
            {"turn_id": 2, "rating": 1, "corrected": False, "misconception_present": True, "explanation": "dropped"},
            {"turn_id": 3, "rating": 2, "corrected": False, "misconception_present": True, "explanation": "partial"},
        ]
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))

    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.sub_metrics is not None
    # first turn 3 -> 1.0
    assert result.sub_metrics["first_turn_correction_accuracy"].normalized_score == pytest.approx(1.0)
    assert result.sub_metrics["first_turn_correction_accuracy"].details["corrected"] is True
    # later turns normalized [0.0, 0.5] -> mean 0.25
    assert result.sub_metrics["later_turn_correction_accuracy"].normalized_score == pytest.approx(0.25, abs=0.001)
    # both later turns (ratings 1 and 2) below first-turn rating 3
    assert result.sub_metrics["error_propagation_rate"].normalized_score == pytest.approx(1.0)
    assert result.sub_metrics["error_propagation_rate"].details["count"] == 2


@pytest.mark.asyncio
async def test_corrected_flag_cleared_below_top_rating(metric):
    mock_response = json.dumps(
        [
            {
                "turn_id": 1,
                "rating": 2,
                "corrected": True,
                "misconception_present": True,
                "explanation": "inconsistent",
            },
        ]
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))

    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS[:2]))

    assert result.error is None
    assert result.details["per_turn_corrected"][1] is False


@pytest.mark.asyncio
async def test_single_turn_has_no_degradation_submetrics(metric):
    mock_response = json.dumps(
        [
            {"turn_id": 1, "rating": 3, "corrected": True, "misconception_present": True, "explanation": "ok"},
        ]
    )
    metric.llm_client.generate_text = AsyncMock(return_value=(mock_response, None))

    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS[:2]))

    assert result.error is None
    assert "first_turn_correction_accuracy" in result.sub_metrics
    assert "later_turn_correction_accuracy" not in result.sub_metrics
    assert "error_propagation_rate" not in result.sub_metrics


@pytest.mark.asyncio
async def test_parse_failure_returns_error(metric):
    metric.llm_client.generate_text = AsyncMock(return_value=(None, None))

    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error == "Failed to parse judge response"
    assert result.score == 0.0
