"""Tests for the measurement-stability layer of JudgeSwapAuditMetric.

Two levels of coverage:

* Pure statistics in ``eva.metrics.diagnostic.measurement_stability`` (variance,
  bootstrap CI, significance flag).
* Integration through the existing ``JudgeSwapAuditMetric.compute`` call site:
  configuring ``stability_samples`` drives the extra per-judge sampling and
  attaches the ``measurement_stability`` block to ``details``.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.metrics.diagnostic.judge_swap_audit import JudgeSwapAuditMetric
from eva.metrics.diagnostic.measurement_stability import (
    assess_stability,
    bootstrap_shift_ci,
    measurement_variance,
)
from tests.unit.metrics.conftest import make_metric_context


def _mock_client(model: str) -> MagicMock:
    client = MagicMock()
    client.model = model
    client.params = {}
    client.generate_text = AsyncMock()
    return client


class TestMeasurementVariance:
    def test_zero_variance_for_constant_samples(self):
        assert measurement_variance([0.5, 0.5, 0.5]) == 0.0

    def test_zero_variance_for_singleton_or_empty(self):
        assert measurement_variance([0.5]) == 0.0
        assert measurement_variance([]) == 0.0

    def test_positive_variance_for_spread(self):
        # population variance of {0.0, 1.0} around mean 0.5 is 0.25
        assert measurement_variance([0.0, 1.0]) == pytest.approx(0.25)


class TestBootstrapShiftCI:
    def test_identical_distributions_ci_hugs_zero(self):
        point, low, high = bootstrap_shift_ci([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert point == 0.0
        assert low == 0.0
        assert high == 0.0

    def test_separated_distributions_ci_excludes_zero(self):
        # judge A always 1.0, judge B always 0.0 -> every resample yields shift 1.0
        point, low, high = bootstrap_shift_ci([1.0, 1.0], [0.0, 0.0])
        assert point == 1.0
        assert low == 1.0 and high == 1.0

    def test_ci_is_reproducible_with_fixed_seed(self):
        a, b = [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]
        assert bootstrap_shift_ci(a, b, seed=7) == bootstrap_shift_ci(a, b, seed=7)


class TestAssessStability:
    def test_bundle_keys_and_significance_flag(self):
        result = assess_stability([1.0, 1.0], [0.0, 0.0], iterations=200)
        assert result["samples_per_judge"] == {"a": 2, "b": 2}
        assert result["shift_point"] == 1.0
        assert result["shift_exceeds_noise"] is True
        assert result["bootstrap_iterations"] == 200

    def test_no_significance_when_judges_agree(self):
        result = assess_stability([0.5, 0.5], [0.5, 0.5])
        assert result["shift_exceeds_noise"] is False


class TestJudgeSwapAuditStabilityWiring:
    @pytest.mark.asyncio
    async def test_stability_block_attached_when_sampled(self):
        ctx = make_metric_context(transcribed_assistant_turns={0: "I booked your flight."})
        metric = JudgeSwapAuditMetric(config={"stability_samples": 3, "bootstrap_iterations": 100})
        metric.llm_client = _mock_client("mock-judge-a")
        metric.llm_client_b = _mock_client("mock-judge-b")
        # Judge A varies its rating across the 3 samples; judge B is steady.
        metric.llm_client.generate_text.side_effect = [
            (json.dumps({"rating": 3}), None),
            (json.dumps({"rating": 1}), None),
            (json.dumps({"rating": 3}), None),
        ]
        metric.llm_client_b.generate_text.return_value = (json.dumps({"rating": 1}), None)

        score = await metric.compute(ctx)

        stability = score.details["measurement_stability"]
        assert stability is not None
        assert stability["samples_per_judge"] == {"a": 3, "b": 3}
        # Judge A's ratings spread (norm 1.0/0.0/1.0) -> positive variance; B constant -> 0.
        assert stability["judge_a_variance"] > 0.0
        assert stability["judge_b_variance"] == 0.0
        assert 0.0 <= stability["shift_ci_low"] <= stability["shift_ci_high"] <= 1.0

    @pytest.mark.asyncio
    async def test_no_stability_block_by_default(self):
        ctx = make_metric_context(transcribed_assistant_turns={0: "Done."})
        metric = JudgeSwapAuditMetric()
        metric.llm_client = _mock_client("mock-judge-a")
        metric.llm_client_b = _mock_client("mock-judge-b")
        metric.llm_client.generate_text.return_value = (json.dumps({"rating": 2}), None)
        metric.llm_client_b.generate_text.return_value = (json.dumps({"rating": 2}), None)

        score = await metric.compute(ctx)

        # Single-shot default: stability is not computed, one call per judge.
        assert score.details["measurement_stability"] is None
        assert metric.llm_client.generate_text.await_count == 1
