"""Integration tests for FaithfulnessVerifierMetric.

These exercise the wiring through existing modules: importing ``eva.metrics``
registers the metric in the global registry, and ``compute()`` drives the real
judge-metric machinery (transcript formatting, usage logging, MetricScore
building) with the LiteLLM Router mocked so no network call is made.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eva.metrics.accuracy.faithfulness_verifier import FaithfulnessVerifierMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Rebook me on a later flight."},
    {"turn_id": 1, "role": "assistant", "content": "Sure, what's your confirmation code?"},
    {"turn_id": 2, "role": "user", "content": "ABC123"},
    {"turn_id": 2, "role": "assistant", "content": "Rebooked you to flight 808."},
]


def _entry(token: str, logprob: float, top_logprobs: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob, top_logprobs=top_logprobs or [])


def _top(token: str, logprob: float) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob)


def _completion_response(entries: list, text: str) -> SimpleNamespace:
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        logprobs=SimpleNamespace(content=entries),
    )
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model="test-model",
    )


def _patch_router(monkeypatch, acompletion):
    monkeypatch.setattr("eva.utils.router.get", lambda: SimpleNamespace(acompletion=acompletion))


def test_metric_registered_via_package_import():
    """Importing the existing eva.metrics package wires the new metric into the registry."""
    import eva.metrics  # noqa: F401  -- side effect: registers all metrics

    assert get_global_registry().get("faithfulness_verifier") is FaithfulnessVerifierMetric


@pytest.mark.asyncio
async def test_compute_returns_continuous_score_from_logprobs(monkeypatch):
    metric = FaithfulnessVerifierMetric()
    # A rating-3-favoring distribution -> continuous expectation near 3.
    entries = [_entry("3", -0.1, [_top("1", -6.0), _top("2", -3.0), _top("3", -0.1)])]
    _patch_router(monkeypatch, AsyncMock(return_value=_completion_response(entries, "3\n{}")))

    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error is None
    assert result.details["scoring_method"] == "llm_as_a_verifier_logprob_expectation"
    assert 2.9 < result.score <= 3.0
    assert result.details["probability_distribution"]["3"] > result.details["probability_distribution"]["1"]
    assert result.normalized_score == pytest.approx((result.details["expectation"] - 1) / 2, abs=1e-3)


@pytest.mark.asyncio
async def test_compute_falls_back_to_discrete_rating_without_logprobs(monkeypatch):
    metric = FaithfulnessVerifierMetric()
    _patch_router(monkeypatch, AsyncMock(return_value=_completion_response([], '{"rating": 2}')))

    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error is None
    # Discrete rating 2 on scale (1, 3) -> expectation 2.0, normalized 0.5.
    assert result.score == 2.0
    assert result.normalized_score == 0.5


@pytest.mark.asyncio
async def test_n_samples_averages_independent_draws(monkeypatch):
    metric = FaithfulnessVerifierMetric()
    metric.n_samples = 3
    responses = [
        _completion_response([_entry("3", -0.1, [_top("3", -0.1)])], "3\n{}"),
        _completion_response([_entry("1", -0.1, [_top("1", -0.1)])], "1\n{}"),
        _completion_response([_entry("2", -0.1, [_top("2", -0.1)])], "2\n{}"),
    ]
    _patch_router(monkeypatch, AsyncMock(side_effect=responses))

    result = await metric.compute(make_metric_context(conversation_trace=SAMPLE_TURNS))

    assert result.error is None
    assert result.details["n_samples"] == 3
    # mean of per-draw expectations [3.0, 1.0, 2.0] == 2.0
    assert result.score == pytest.approx(2.0, abs=1e-4)
