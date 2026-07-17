"""Tests for JudgeSwapAuditMetric -- LLM-as-judge reliability audit.

Integration coverage: importing ``eva.metrics.diagnostic`` registers the metric
in the global registry (the ``diagnostic/__init__.py`` wiring edit), and
``compute()`` drives the existing ``TextJudgeMetric`` plumbing
(``parse_judge_response`` + ``validate_and_normalize_rating``) to produce the
evaluator-replacement shift, the position-bias probe, and the per-judge audit
trail.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from eva.metrics import diagnostic
from eva.metrics.diagnostic.judge_swap_audit import JudgeSwapAuditMetric
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_metric_context


def _mock_client(model: str) -> MagicMock:
    """Build a judge client mock matching the LLMClient surface compute() touches."""
    client = MagicMock()
    client.model = model
    client.params = {}
    client.generate_text = AsyncMock()
    return client


def _make_metric(*, config: dict | None = None) -> JudgeSwapAuditMetric:
    """Instantiate the metric with two distinct mocked judges for independent control."""
    metric = JudgeSwapAuditMetric(config=config)
    metric.llm_client = _mock_client("mock-judge-a")
    metric.llm_client_b = _mock_client("mock-judge-b")
    return metric


class TestJudgeSwapAuditRegistration:
    def test_registered_via_diagnostic_package(self):
        # Proves the diagnostic/__init__.py wiring: the opt-in metric is resolvable by name.
        registry = get_global_registry()
        assert registry.get("judge_swap_audit") is JudgeSwapAuditMetric
        assert "judge_swap_audit" in diagnostic.__all__

    def test_attributes(self):
        metric = _make_metric()
        assert metric.name == "judge_swap_audit"
        assert metric.category == "diagnostic"
        assert metric.version == "v0.2"
        assert metric.rating_scale == (1, 3)
        assert metric.higher_is_better is False
        assert metric.exclude_from_default_metrics is True
        assert metric.exclude_from_pass_at_k is True

    def test_identical_judges_by_default(self):
        assert _make_metric().judges_identical is True

    def test_distinct_judges_when_judge_model_b_configured(self):
        metric = _make_metric(config={"judge_model_b": "gpt-4o"})
        assert metric.judges_identical is False


class TestJudgeSwapAuditReplacementShift:
    @pytest.mark.asyncio
    async def test_shift_when_judges_disagree(self):
        # Single assistant turn -> position-bias probe skipped (needs >= 2 turns).
        ctx = make_metric_context(transcribed_assistant_turns={0: "I booked your flight."})
        metric = _make_metric()
        metric.llm_client.generate_text.return_value = (json.dumps({"rating": 3, "explanation": "good"}), None)
        metric.llm_client_b.generate_text.return_value = (json.dumps({"rating": 1, "explanation": "bad"}), None)

        score = await metric.compute(ctx)

        assert score.error is None
        # rating 3 -> norm 1.0, rating 1 -> norm 0.0, shift == 1.0
        assert score.normalized_score == 1.0
        assert score.details["evaluator_replacement_shift"] == 1.0
        audit = score.details["audit_trail"]
        assert audit["judge_a"]["normalized"] == 1.0
        assert audit["judge_b"]["normalized"] == 0.0
        assert audit["judge_a"]["parsed_ok"] is True
        assert audit["judges_identical"] is True
        assert score.sub_metrics is None
        assert score.details["position_bias_probe"] is None

    @pytest.mark.asyncio
    async def test_zero_shift_when_judges_agree(self):
        ctx = make_metric_context(transcribed_assistant_turns={0: "Done."})
        metric = _make_metric()
        rating = json.dumps({"rating": 2})
        metric.llm_client.generate_text.return_value = (rating, None)
        metric.llm_client_b.generate_text.return_value = (rating, None)

        score = await metric.compute(ctx)

        assert score.normalized_score == 0.0
        assert score.skipped is False

    @pytest.mark.asyncio
    async def test_parse_failure_is_skipped_with_audit_trail(self):
        ctx = make_metric_context(transcribed_assistant_turns={0: "Hi"})
        metric = _make_metric()
        metric.llm_client.generate_text.return_value = ("not json at all", None)
        metric.llm_client_b.generate_text.return_value = (json.dumps({"rating": 2}), None)

        score = await metric.compute(ctx)

        assert score.skipped is True
        assert score.normalized_score is None
        assert score.details["audit_trail"]["judge_a"]["parsed_ok"] is False
        assert score.details["audit_trail"]["judge_b"]["parsed_ok"] is True

    @pytest.mark.asyncio
    async def test_no_transcript_returns_error(self):
        ctx = make_metric_context(transcribed_assistant_turns={})
        metric = _make_metric()
        score = await metric.compute(ctx)
        assert score.error == "No assistant transcript available to judge"


class TestJudgeSwapAuditPositionBias:
    @pytest.mark.asyncio
    async def test_position_bias_detected_on_order_flip(self):
        # Two assistant turns -> probe runs (identical judges -> judge "a" only).
        ctx = make_metric_context(
            transcribed_assistant_turns={0: "first half response", 1: "second half response"},
        )
        metric = _make_metric()
        # Judge "a": rating call, then probe picks "A" in both orderings. A raw
        # choice of "A" in both orders means the fixed winner flips (first -> second),
        # which is exactly position bias.
        metric.llm_client.generate_text.side_effect = [
            (json.dumps({"rating": 3}), None),
            (json.dumps({"choice": "A"}), None),
            (json.dumps({"choice": "A"}), None),
        ]
        metric.llm_client_b.generate_text.return_value = (json.dumps({"rating": 3}), None)

        score = await metric.compute(ctx)

        assert "position_bias_rate" in score.sub_metrics
        assert score.sub_metrics["position_bias_rate"].normalized_score == 1.0
        probe = score.details["position_bias_probe"]
        assert probe["per_judge"]["a"]["winners"] == ["first_half", "second_half"]
        assert probe["per_judge"]["a"]["flipped"] is True

    @pytest.mark.asyncio
    async def test_no_position_bias_when_preference_stable(self):
        ctx = make_metric_context(
            transcribed_assistant_turns={0: "first", 1: "second"},
        )
        metric = _make_metric()
        # order 0 picks "A" (first_half); order 1 picks "B" (still first_half) -> stable.
        metric.llm_client.generate_text.side_effect = [
            (json.dumps({"rating": 3}), None),
            (json.dumps({"choice": "A"}), None),
            (json.dumps({"choice": "B"}), None),
        ]
        metric.llm_client_b.generate_text.return_value = (json.dumps({"rating": 3}), None)

        score = await metric.compute(ctx)

        assert score.sub_metrics["position_bias_rate"].normalized_score == 0.0
        assert score.details["position_bias_probe"]["per_judge"]["a"]["flipped"] is False
