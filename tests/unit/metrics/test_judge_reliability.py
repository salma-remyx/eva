"""Tests for the judge reliability audit (eva.metrics.judge_reliability).

Covers the pure reliability statistics and the wiring into MetricsRunner's
per-metric compute path, which is where the audit becomes part of a real run.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.accuracy.task_completion import TaskCompletion
from eva.metrics.base import MetricContext
from eva.metrics.judge_reliability import (
    REVERSED,
    VerdictSample,
    audit_metric_verdict,
    reverse_context_order,
    summarize_verdicts,
    trustworthy_verdict_rate,
    verdict_of,
)
from eva.metrics.runner import MetricsRunner
from eva.models.results import MetricScore
from tests.unit.metrics.conftest import make_metric_context


def make_judge(*, flip_on_reverse: bool = True) -> FaithfulnessJudgeMetric:
    """A faithfulness judge whose verdict flips when the transcript is reversed."""
    metric = FaithfulnessJudgeMetric()
    metric.llm_client = MagicMock()
    metric.llm_client.params = {}

    async def generate_text(messages):
        prompt = messages[0]["content"]
        rating = 1 if flip_on_reverse and prompt.index("Turn 1") < prompt.index("Turn 0") else 3
        return json.dumps({"rating": rating}), None

    metric.llm_client.generate_text = AsyncMock(side_effect=generate_text)
    return metric


def make_partial_runner(metric_configs: dict | None = None) -> MetricsRunner:
    """A MetricsRunner carrying only what _run_record needs, so the real path runs."""
    runner = MetricsRunner.__new__(MetricsRunner)
    runner.metrics = []
    runner.record_metric_filter = None
    runner.metric_configs = metric_configs or {}
    return runner


def make_audit_context() -> MetricContext:
    """A context with two turns in presentation order and non-empty turn counts."""
    return make_metric_context(
        record_id="1.1.1",
        num_assistant_turns=1,
        num_user_turns=1,
        conversation_trace=[
            {"turn_id": 0, "role": "user", "content": "book me a flight"},
            {"turn_id": 1, "role": "assistant", "content": "done"},
        ],
    )


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


class TestRunnerSettings:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("JUDGE_RELIABILITY_REPS", raising=False)
        monkeypatch.delenv("JUDGE_RELIABILITY_ORDER_SWAP", raising=False)
        runner = make_partial_runner()

        assert runner._judge_reliability_settings(FaithfulnessJudgeMetric()) == (0, False)

    def test_code_metrics_are_never_audited(self, monkeypatch):
        monkeypatch.setenv("JUDGE_RELIABILITY_REPS", "5")
        runner = make_partial_runner()

        assert runner._judge_reliability_settings(TaskCompletion()) == (0, False)

    def test_per_metric_config_wins(self, monkeypatch):
        monkeypatch.setenv("JUDGE_RELIABILITY_REPS", "5")
        runner = make_partial_runner({"faithfulness": {"reliability_reps": 2}})

        assert runner._judge_reliability_settings(FaithfulnessJudgeMetric()) == (2, False)

    def test_env_vars_enable_it(self, monkeypatch):
        monkeypatch.setenv("JUDGE_RELIABILITY_REPS", "3")
        monkeypatch.setenv("JUDGE_RELIABILITY_ORDER_SWAP", "true")
        runner = make_partial_runner()

        assert runner._judge_reliability_settings(FaithfulnessJudgeMetric()) == (3, True)


class TestRunRecordWiring:
    """Drive the audit hook through the real MetricsRunner._run_record path."""

    async def _run_record(self, judge, metric_configs):
        runner = make_partial_runner(metric_configs)
        runner.metrics = [judge]
        runner._load_context = AsyncMock(return_value=make_audit_context())
        return await runner._run_record("1.1.1", Path("records/1.1.1"))

    @pytest.mark.asyncio
    async def test_audit_off_by_default(self):
        judge = make_judge()

        result = await self._run_record(judge, None)

        assert "judge_reliability" not in result.metrics["faithfulness"].details
        assert judge.llm_client.generate_text.await_count == 1

    @pytest.mark.asyncio
    async def test_audit_attached_to_the_score(self):
        judge = make_judge()

        result = await self._run_record(
            judge, {"faithfulness": {"reliability_reps": 1, "reliability_order_swap": True}}
        )

        # Primary call + 1 forward repetition + 2 reversed runs (reps + a primary one).
        assert judge.llm_client.generate_text.await_count == 4
        reliability = result.metrics["faithfulness"].details["judge_reliability"]
        assert reliability["verdicts"] == {"forward": [3.0, 3.0], REVERSED: [1.0, 1.0]}
        assert reliability["reproducible"] is True
        assert reliability["order_invariant"] is False
        assert reliability["trustworthy"] is False
        # The reported verdict itself is unchanged by the audit.
        assert result.metrics["faithfulness"].score == 3.0

    @pytest.mark.asyncio
    async def test_stable_judge_is_trustworthy(self):
        judge = make_judge(flip_on_reverse=False)

        result = await self._run_record(
            judge, {"faithfulness": {"reliability_reps": 1, "reliability_order_swap": True}}
        )

        reliability = result.metrics["faithfulness"].details["judge_reliability"]
        assert reliability["reproducible"] is True
        assert reliability["order_invariant"] is True
        assert reliability["trustworthy"] is True
