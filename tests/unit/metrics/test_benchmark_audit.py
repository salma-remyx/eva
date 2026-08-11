"""Tests for BenchmarkAuditMetric (reference-free benchmark-quality auditor)."""

import json
from unittest.mock import AsyncMock

import pytest

import eva.metrics.diagnostic  # noqa: F401  -- triggers registry wiring via __init__
from eva.metrics.diagnostic.benchmark_audit import BenchmarkAuditMetric
from eva.metrics.registry import get_global_registry

from .conftest import make_judge_metric, make_metric_context


@pytest.fixture
def metric():
    return make_judge_metric(BenchmarkAuditMetric)


def _scenario_context(**overrides):
    """A well-formed scenario definition for the auditor to read."""
    base = {
        "user_goal": "Book a one-way flight from JFK to LHR for under $500 and confirm the booking.",
        "agent_instructions": (
            "You are an airline booking agent. Always verify the passenger's full name and date "
            "of birth before booking. Never book a flight that exceeds the user's stated budget. "
            "Only offer flights within the next 12 months."
        ),
        "expected_scenario_db": {
            "booking": {"status": "confirmed", "route": "JFK-LHR", "price": 420},
            "passenger_verified": True,
        },
    }
    base.update(overrides)
    return make_metric_context(**base)


class TestRegistryWiring:
    """The diagnostic __init__ import must register the metric for MetricsRunner."""

    def test_resolvable_by_name(self):
        registry = get_global_registry()
        assert registry.get("benchmark_audit") is BenchmarkAuditMetric
        assert "benchmark_audit" in registry.get_all()

    def test_create_returns_instance(self):
        instance = get_global_registry().create("benchmark_audit")
        assert isinstance(instance, BenchmarkAuditMetric)

    def test_opt_in_not_in_default_list(self):
        # Audits the benchmark, not the conversation: excluded from default metrics.
        assert "benchmark_audit" not in get_global_registry().list_metrics()


@pytest.mark.asyncio
async def test_strong_scenario_scores_full(metric):
    """All-three-strong scenario yields 1.0 normalized and matching sub-metrics."""
    context = _scenario_context()
    metric.llm_client.generate_text = AsyncMock(
        return_value=(
            json.dumps(
                {
                    "rating": 3,
                    "consistency": {"rating": 3, "explanation": "coherent"},
                    "complexity": {"rating": 3, "explanation": "multi-step"},
                    "policy_coverage": {"rating": 3, "explanation": "stresses policy"},
                    "explanation": "A strong scenario.",
                    "diagnostics": [],
                }
            ),
            None,
        )
    )

    result = await metric.compute(context)

    assert result.error is None
    assert result.normalized_score == 1.0
    assert result.score == 3.0  # mean of raw 1-3 dimension ratings
    assert set(result.sub_metrics) == {"consistency", "complexity", "policy_coverage"}
    assert result.sub_metrics["consistency"].normalized_score == 1.0
    assert result.details["overall_rating"] == 3
    assert result.details["diagnostics"] == []
    assert result.prompt_hash is not None  # inline prompt still hashed for reproducibility


@pytest.mark.asyncio
async def test_mixed_ratings_average(metric):
    """Mixed per-dimension ratings average across both raw and normalized scales."""
    context = _scenario_context()
    metric.llm_client.generate_text = AsyncMock(
        return_value=(
            json.dumps(
                {
                    "rating": 2,
                    "consistency": {"rating": 3, "explanation": "coherent"},
                    "complexity": {"rating": 1, "explanation": "too trivial"},
                    "policy_coverage": {"rating": 2, "explanation": "partial"},
                    "explanation": "A simple scenario.",
                    "diagnostics": ["too easy", "policy only partly exercised"],
                }
            ),
            None,
        )
    )

    result = await metric.compute(context)

    assert result.error is None
    # raw: (3 + 1 + 2) / 3 = 2.0 ; normalized: (1.0 + 0.0 + 0.5) / 3 = 0.5
    assert result.score == pytest.approx(2.0, abs=1e-3)
    assert result.normalized_score == pytest.approx(0.5, abs=1e-3)
    assert result.sub_metrics["complexity"].normalized_score == 0.0
    assert result.sub_metrics["policy_coverage"].normalized_score == 0.5
    assert result.details["dimension_ratings"] == {
        "consistency": 3,
        "complexity": 1,
        "policy_coverage": 2,
    }
    assert len(result.details["diagnostics"]) == 2


@pytest.mark.asyncio
async def test_missing_user_goal_is_skipped(metric):
    """No task description means there is no scenario to audit."""
    context = _scenario_context(user_goal="   ")
    result = await metric.compute(context)

    assert result.skipped is True
    assert result.error == "No scenario description (user_goal) to audit"


@pytest.mark.asyncio
async def test_unparseable_judge_response(metric):
    """A response the judge cannot parse surfaces a parse error."""
    context = _scenario_context()
    metric.llm_client.generate_text = AsyncMock(return_value=("I cannot produce JSON.", None))

    result = await metric.compute(context)

    assert result.error == "Failed to parse judge response"
    assert result.normalized_score == 0.0


@pytest.mark.asyncio
async def test_all_dimensions_missing(metric):
    """Parseable response with no dimension ratings reports an evaluation failure."""
    context = _scenario_context()
    metric.llm_client.generate_text = AsyncMock(
        return_value=(json.dumps({"rating": 2, "explanation": "no axes", "diagnostics": []}), None)
    )

    result = await metric.compute(context)

    assert result.error == "All dimensions failed to evaluate"
    assert result.normalized_score == 0.0
