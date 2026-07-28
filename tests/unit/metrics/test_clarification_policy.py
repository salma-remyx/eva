"""Tests for ClarificationPolicyJudgeMetric (clarification-as-policy, adapted from RegretBench)."""

import json

import pytest

from eva.metrics.experience.clarification_policy import ClarificationPolicyJudgeMetric
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestClarificationPolicy:
    def setup_method(self):
        self.metric = make_judge_metric(ClarificationPolicyJudgeMetric, mock_llm=True)

    def test_registered_by_name(self):
        """Integration: the metric is discoverable via the global registry under its name.

        This is the select-by-name surface the rest of the repo uses
        (``registry.create("clarification_policy")``); it only resolves because
        ``experience/__init__`` imports the module and ``@register_metric`` fires.
        """
        registry = get_global_registry()
        assert registry.get("clarification_policy") is ClarificationPolicyJudgeMetric
        assert registry.create("clarification_policy").name == "clarification_policy"

    def test_metric_attributes(self):
        assert self.metric.name == "clarification_policy"
        assert self.metric.category == "experience"
        assert self.metric.rating_scale == (1, 3)

    def test_get_prompt_variables(self):
        ctx = make_metric_context(user_goal="Book the cheapest flight to Paris")
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")
        assert variables["conversation_trace"] == "User: hi\nBot: hello"
        assert variables["user_goal"] == "Book the cheapest flight to Paris"

    def test_build_metric_score_surfaces_dimension_sub_metrics(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "rating": 1,
            "dimensions": {
                "missed_clarification": {"rating": 1, "flagged": True, "evidence": "acted without confirming"},
                "ineffective_clarification": {"rating": 3, "flagged": False, "evidence": "clean"},
                "poor_stopping_decision": {"rating": 2, "flagged": True, "evidence": "one extra turn"},
            },
        }

        score = self.metric.build_metric_score(
            rating=1,
            normalized=0.0,
            response=response,
            prompt="test prompt",
            context=ctx,
            raw_response="{...}",
        )

        assert score.name == "clarification_policy"
        assert score.score == 1.0
        assert score.normalized_score == 0.0
        assert score.details["num_turns"] == 2
        assert set(score.sub_metrics.keys()) == {
            "missed_clarification_rate",
            "ineffective_clarification_rate",
            "poor_stopping_decision_rate",
        }
        # Binary issue-flag: 1.0 when flagged, 0.0 when clean; lower is better.
        missed = score.sub_metrics["missed_clarification_rate"]
        assert missed.name == "clarification_policy.missed_clarification_rate"
        assert missed.score == 1.0  # flagged
        assert missed.details["flagged"] is True
        assert missed.details["rating"] == 1

        clean = score.sub_metrics["ineffective_clarification_rate"]
        assert clean.score == 0.0
        assert clean.details["flagged"] is False

    @pytest.mark.asyncio
    async def test_compute_low_regret(self):
        """A well-timed clarification that confirms the ambiguous value earns the top rating."""
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 3, "dimensions": {}}),
            None,
        )
        ctx = make_metric_context(
            user_goal="Cancel booking BA123",
            conversation_trace=[
                {"role": "user", "content": "cancel my booking"},
                {"role": "assistant", "content": "Sure — is that BA123?"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Done, BA123 is cancelled."},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        assert score.error is None

    @pytest.mark.asyncio
    async def test_compute_high_regret(self):
        """Acting on an unconfirmed, ambiguous value is a high-regret policy failure."""
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 1, "dimensions": {}}),
            None,
        )
        ctx = make_metric_context(
            user_goal="Cancel booking BA123",
            conversation_trace=[
                {"role": "user", "content": "cancel my booking"},
                {"role": "assistant", "content": "Done, cancelled BA124."},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 1.0
        assert score.normalized_score == 0.0
