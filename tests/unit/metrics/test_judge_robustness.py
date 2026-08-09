"""Tests for JudgeRobustnessDiagnosticMetric.

Exercises the registry wiring (the diagnostic ``__init__`` import) and the
score-shift robustness computation against the existing ``conciseness`` judge.
"""

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from eva.metrics.diagnostic import judge_robustness
from eva.metrics.diagnostic.judge_robustness import JudgeRobustnessDiagnosticMetric
from eva.metrics.experience.conciseness import ConcisenessJudgeMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Hi, I need help rebooking my flight."},
    {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation number?"},
    {"turn_id": 2, "role": "user", "content": "ABC123"},
    {"turn_id": 2, "role": "assistant", "content": "Found it. When would you like to fly?"},
    {"turn_id": 3, "role": "user", "content": "March 25th please."},
    {"turn_id": 3, "role": "assistant", "content": "Done, you're rebooked for March 25th."},
]


def _judge_response(rating: int) -> tuple[str, None]:
    """Build a conciseness judge response (text, usage) for a single turn."""
    payload = [{"turn_id": 1, "rating": rating, "explanation": "x", "failure_modes": []}]
    return json.dumps(payload), None


class _PinnedJudgeMetric(JudgeRobustnessDiagnosticMetric):
    """Wrapper variant that returns a pre-built inner judge, for deterministic tests."""

    def __init__(self, inner: ConcisenessJudgeMetric, **config: Any) -> None:
        super().__init__(config=dict(config))
        self._inner = inner

    def _build_inner_judge(self) -> ConcisenessJudgeMetric:
        return self._inner


def _make_metric(responses: list[tuple[str, None]]) -> _PinnedJudgeMetric:
    inner = make_judge_metric(ConcisenessJudgeMetric)
    inner.llm_client.generate_text = AsyncMock(side_effect=list(responses))
    return _PinnedJudgeMetric(inner)


def test_registered_in_global_registry() -> None:
    """Importing the diagnostic package wires the metric into the registry."""
    assert get_global_registry().get("judge_robustness") is JudgeRobustnessDiagnosticMetric


def test_excluded_from_default_metrics() -> None:
    """Robustness is opt-in (expensive), so not in the default metric list."""
    assert "judge_robustness" not in get_global_registry().list_metrics()


def test_perturbation_helpers_apply_each_attack() -> None:
    """Each attack produces a distinct, deterministic perturbation."""
    trace = [{"turn_id": 1, "role": "assistant", "content": "The flight is booked."}]

    prefixed = judge_robustness.perturb_conversation_trace(trace, judge_robustness.SPEAKER_TAG_PREFIX)
    assert prefixed[0]["content"] == "Agent: The flight is booked."

    static = judge_robustness.perturb_conversation_trace(trace, judge_robustness.STATIC_RESPONSE)
    assert static[0]["content"] == judge_robustness._STATIC_RESPONSE_TEXT

    ungram = judge_robustness.perturb_conversation_trace(trace, judge_robustness.UNGRAMMATICAL_RESPONSE)
    assert ungram[0]["content"] == "flight flight is booked."

    repeated = judge_robustness.perturb_conversation_trace(trace, judge_robustness.REPEATED_CONTEXT)
    assert len(repeated) == 2
    assert repeated[0]["turn_id"] == 1
    assert repeated[1]["turn_id"] == 2  # re-numbered duplicate


@pytest.mark.asyncio
async def test_zero_shift_when_judge_is_stable() -> None:
    """A judge returning the same rating everywhere yields zero shift."""
    metric = _make_metric([_judge_response(3)] * 5)
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.score == 0.0
    assert result.details["judge_metric"] == "conciseness"
    assert result.details["baseline_score"] == pytest.approx(1.0, abs=0.001)
    assert set(result.sub_metrics.keys()) == {
        "speaker_tag_prefix_shift",
        "static_response_shift",
        "ungrammatical_response_shift",
        "repeated_context_shift",
    }
    assert all(sub.score == 0.0 for sub in result.sub_metrics.values())


@pytest.mark.asyncio
async def test_reports_nonzero_shift_when_judge_moves() -> None:
    """Baseline rating 3 (norm 1.0) vs perturbed rating 1 (norm 0.0) -> shift 1.0 each."""
    # Order: baseline, then the four attacks in DEFAULT_ATTACKS order.
    metric = _make_metric(
        [_judge_response(3), _judge_response(1), _judge_response(1), _judge_response(1), _judge_response(1)]
    )
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is None
    assert result.score == pytest.approx(1.0, abs=0.001)
    assert result.sub_metrics["static_response_shift"].score == pytest.approx(1.0, abs=0.001)
    assert result.details["per_attack"]["static_response"]["shift"] == pytest.approx(1.0, abs=0.001)


@pytest.mark.asyncio
async def test_unknown_judge_returns_error() -> None:
    """An unresolvable wrapped judge surfaces a clear error rather than crashing."""
    metric = JudgeRobustnessDiagnosticMetric(config={"judge_metric": "does_not_exist"})
    context = make_metric_context(conversation_trace=SAMPLE_TURNS)
    result = await metric.compute(context)

    assert result.error is not None
    assert "does_not_exist" in result.error


@pytest.mark.asyncio
async def test_empty_transcript_returns_error() -> None:
    """No transcript means there is nothing to perturb or score."""
    metric = _make_metric([_judge_response(3)])
    context = make_metric_context(conversation_trace=[])
    result = await metric.compute(context)

    assert result.error is not None
