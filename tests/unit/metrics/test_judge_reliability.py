"""Tests for the judge reliability statistics (eva.metrics.judge_reliability).

Covers the pure per-item and population statistics; the wiring into
``MetricsRunner`` lives in ``test_runner.py``, the mirror of the module that
changed.
"""

import pytest

from eva.metrics.judge_reliability import (
    REVERSED,
    VerdictSample,
    audit_metric_verdict,
    reverse_context_order,
    summarize_verdicts,
    trustworthy_verdict_rate,
    verdict_of,
)
from eva.models.results import MetricScore
from tests.unit.metrics.conftest import make_metric_context


class TestReverseContextOrder:
    def test_reverses_transcript_without_touching_the_original(self):
        ctx = make_metric_context(conversation_trace=[{"turn_id": 0}, {"turn_id": 1}, {"turn_id": 2}])

        swapped = reverse_context_order(ctx)

        assert [t["turn_id"] for t in swapped.conversation_trace] == [2, 1, 0]
        assert [t["turn_id"] for t in ctx.conversation_trace] == [0, 1, 2]
        assert swapped is not ctx
        assert swapped.record_id == ctx.record_id


class TestVerdictOf:
    def test_prefers_the_reported_rating(self):
        score = MetricScore(name="m", score=3.0, normalized_score=1.0, details={"rating": 2})
        assert verdict_of(score) == 2.0

    def test_falls_back_to_the_aggregate_for_per_turn_judges(self):
        score = MetricScore(name="m", score=2.5, normalized_score=0.75, details={"per_turn_ratings": {}})
        assert verdict_of(score) == 2.5

    def test_no_verdict_at_all(self):
        assert verdict_of(MetricScore(name="m", score=None, details={})) is None


class TestSummarizeVerdicts:
    def test_reproducible_and_order_invariant(self):
        samples = [VerdictSample("forward", 3)] * 3 + [VerdictSample(REVERSED, 3)] * 2
        summary = summarize_verdicts(samples, ground_truth=3)

        assert summary["reproducible"] is True
        assert summary["order_invariant"] is True
        assert summary["accurate"] is True
        assert summary["trustworthy"] is True
        assert summary["verdict_agreement_rate"] == 1.0

    def test_order_flip_is_not_trustworthy(self):
        samples = [VerdictSample("forward", 3)] * 3 + [VerdictSample(REVERSED, 1)] * 3
        summary = summarize_verdicts(samples)

        assert summary["order_invariant"] is False
        assert summary["trustworthy"] is False
        # Accuracy is unknown, so it must not gate the verdict either way.
        assert summary["accurate"] is None

    def test_unreproducible_verdicts(self):
        summary = summarize_verdicts([VerdictSample("forward", r) for r in (3, 3, 1)])

        assert summary["modal_verdict"] == 3
        assert summary["reproducible"] is False
        assert summary["verdict_agreement_rate"] == pytest.approx(2 / 3)

    def test_components_not_estimable_are_none(self):
        summary = summarize_verdicts([VerdictSample("forward", 2)])

        assert summary["reproducible"] is None
        assert summary["order_invariant"] is None
        assert summary["trustworthy"] is None

    def test_unparsed_verdicts_are_counted_not_scored(self):
        samples = [VerdictSample("forward", 3), VerdictSample("forward", None), VerdictSample("forward", 3)]
        summary = summarize_verdicts(samples)

        assert summary["unparsed_verdicts"] == 1
        assert summary["verdicts"] == {"forward": [3.0, 3.0], REVERSED: []}


class TestTrustworthyVerdictRate:
    def test_population_rates(self):
        stable = summarize_verdicts([VerdictSample("forward", 3)] * 3 + [VerdictSample(REVERSED, 3)] * 2)
        flipped = summarize_verdicts([VerdictSample("forward", 3)] * 3 + [VerdictSample(REVERSED, 1)] * 2)

        rates = trustworthy_verdict_rate([stable, flipped])

        assert rates["num_items"] == 2
        assert rates["reproducibility_rate"] == 1.0
        assert rates["order_invariance_rate"] == 0.5
        assert rates["flip_rate"] == 0.5
        assert rates["trustworthy_verdict_rate"] == 0.5
        # Half the items flip under a presentation-order swap, so no judge can be
        # more than 75% accurate under an arbitrary order.
        assert rates["position_bias_accuracy_bound"] == 0.75

    def test_empty_population(self):
        assert trustworthy_verdict_rate([])["num_items"] == 0
        assert trustworthy_verdict_rate([])["trustworthy_verdict_rate"] is None

    def test_accuracy_rate_needs_ground_truth(self):
        audited = trustworthy_verdict_rate([summarize_verdicts([VerdictSample("forward", 3)], ground_truth=3)])
        unaudited = trustworthy_verdict_rate([summarize_verdicts([VerdictSample("forward", 3)])])
        assert audited["accuracy_rate"] == 1.0
        assert unaudited["accuracy_rate"] is None


class TestAuditMetricVerdict:
    @pytest.mark.asyncio
    async def test_disabled_audit_makes_no_calls(self):
        calls = []

        async def compute(_ctx):
            calls.append(_ctx)
            return MetricScore(name="m", score=3, details={"rating": 3})

        result = await audit_metric_verdict(compute, make_metric_context(), primary_verdict=3)

        assert result == {}
        assert calls == []

    @pytest.mark.asyncio
    async def test_repetitions_and_order_swap(self):
        calls = []

        async def compute(ctx):
            calls.append(ctx)
            rating = 1 if ctx.conversation_trace[0]["turn_id"] > ctx.conversation_trace[-1]["turn_id"] else 3
            return MetricScore(name="m", score=rating, details={"rating": rating})

        ctx = make_metric_context(
            conversation_trace=[{"turn_id": 0, "role": "user"}, {"turn_id": 1, "role": "assistant"}]
        )
        summary = await audit_metric_verdict(compute, ctx, primary_verdict=3, reps=2, order_swap=True)

        # 2 extra forward runs + 3 reversed runs (reps + a primary one).
        assert len(calls) == 5
        assert summary["verdicts"] == {"forward": [3.0, 3.0, 3.0], REVERSED: [1.0, 1.0, 1.0]}
        assert summary["reproducible"] is True
        assert summary["order_invariant"] is False
        assert summary["trustworthy"] is False
