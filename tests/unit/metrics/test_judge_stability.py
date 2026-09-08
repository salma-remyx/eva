"""Tests for JudgeStabilityMetric."""

import json
from unittest.mock import AsyncMock

import pytest

import eva.metrics.diagnostic  # noqa: F401  (import registers all diagnostic metrics)
from eva.metrics.diagnostic.judge_stability import JudgeStabilityMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Hi, I need help rebooking my flight."},
    {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation number?"},
    {"turn_id": 2, "role": "user", "content": "ABC123"},
    {"turn_id": 2, "role": "assistant", "content": "Found your reservation. When would you like to fly?"},
    {"turn_id": 3, "role": "user", "content": "March 25th please."},
    {"turn_id": 3, "role": "assistant", "content": "Done, you're rebooked for March 25th."},
]


def _judge_response(rating: int) -> str:
    return json.dumps({"rating": rating, "explanation": "ok"})


@pytest.fixture
def metric():
    return make_judge_metric(JudgeStabilityMetric)


def test_registered_in_global_registry():
    """The diagnostic package registers the metric; it stays opt-in (pilot) by default."""
    registry = get_global_registry()
    assert registry.get("judge_stability") is JudgeStabilityMetric
    assert "judge_stability" not in registry.list_metrics()


def test_repeats_config():
    """`repeats` is configurable and clamped to the minimum needed to measure agreement."""
    assert JudgeStabilityMetric(config={"repeats": 3}).repeats == 3
    assert JudgeStabilityMetric(config={"repeats": 1}).repeats == 2
    assert JudgeStabilityMetric(config={"repeats": "many"}).repeats == JudgeStabilityMetric.default_repeats


@pytest.mark.asyncio
async def test_stable_judge_scores_full_agreement(metric):
    """A judge returning the same rating on every identical call scores 1.0."""
    metric.llm_client.generate_text = AsyncMock(return_value=(_judge_response(3), None))
    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error is None
    assert result.score == 1.0
    assert result.normalized_score == 1.0
    assert result.details["per_repeat_ratings"] == [3, 3, 3, 3, 3]
    assert result.details["rating_histogram"] == {3: 5}
    assert result.details["distinct_ratings"] == 1
    assert result.details["noise_floor_normalized"] == 0.0
    assert result.details["num_parse_failures"] == 0

    # Every repeat sent byte-identical content through the shared call path.
    calls = metric.llm_client.generate_text.await_args_list
    assert len(calls) == metric.repeats
    assert all(call.args[0] == calls[0].args[0] for call in calls)


@pytest.mark.asyncio
async def test_unstable_judge_reports_partial_agreement(metric):
    """A wobbling judge yields partial agreement and a nonzero noise floor."""
    metric.llm_client.generate_text = AsyncMock(side_effect=[(_judge_response(r), None) for r in (2, 3, 2, 3, 2)])
    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error is None
    # Modal rating 2 appears 3/5 times.
    assert result.score == 0.6
    assert result.details["modal_rating"] == 2
    assert result.details["distinct_ratings"] == 2
    assert result.details["rating_histogram"] == {2: 3, 3: 2}
    # Spread of one step on the 1-3 scale = 0.5 in normalized units.
    assert result.details["noise_floor_normalized"] == 0.5


@pytest.mark.asyncio
async def test_unparseable_and_invalid_ratings_are_counted_not_scored(metric):
    """Parse failures and out-of-schema ratings are surfaced as counts instead of ratings."""
    metric.llm_client.generate_text = AsyncMock(
        side_effect=[
            ("not json", None),
            (_judge_response(2), None),
            (json.dumps({"rating": "excellent", "explanation": "label drift"}), None),
        ]
    )
    metric.repeats = 3
    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error is None
    assert result.details["num_valid"] == 1
    assert result.details["num_parse_failures"] == 1
    assert result.details["num_invalid_ratings"] == 1
    # A single usable rating trivially agrees with itself.
    assert result.score == 1.0
    assert result.details["parse_stability_rate"] == round(1 / 3, 3)


@pytest.mark.asyncio
async def test_all_failures_surface_as_error(metric):
    """When no repeat returns a usable rating, the metric reports an error."""
    metric.llm_client.generate_text = AsyncMock(return_value=("not json", None))
    metric.repeats = 2
    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error == "No judge call returned a usable rating"
    assert result.score == 0.0
    assert result.details["num_parse_failures"] == 2


@pytest.mark.asyncio
async def test_no_transcript_short_circuits(metric):
    """Records without a transcript return the standard no-transcript error."""
    result = await metric.compute(make_metric_context(conversation_trace=[]))

    assert result.error == "No transcript available"
    metric.llm_client.generate_text.assert_not_called()
