"""Tests for MultiTurnUnderstandingJudgeMetric."""

import json

import pytest

# Importing the package triggers the ``@register_metric`` wiring in
# ``eva.metrics.accuracy.__init__``; resolving through the global registry
# below proves the metric is reachable via the production discovery path.
import eva.metrics  # noqa: F401
from eva.metrics.accuracy.multiturn_understanding import MultiTurnUnderstandingJudgeMetric
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestMultiTurnUnderstanding:
    def setup_method(self):
        self.metric = make_judge_metric(MultiTurnUnderstandingJudgeMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "multiturn_understanding"
        assert self.metric.category == "accuracy"
        assert self.metric.rating_scale == (1, 3)
        assert self.metric.version == "v0.1"

    def test_registered_in_global_registry(self):
        """The metric is discoverable by name via the production registry path."""
        registry = get_global_registry()
        assert registry.get("multiturn_understanding") is MultiTurnUnderstandingJudgeMetric
        assert "multiturn_understanding" in registry.list_metrics()
        created = registry.create("multiturn_understanding")
        assert created is not None
        assert created.name == "multiturn_understanding"

    def test_get_prompt_variables(self):
        ctx = make_metric_context()
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")
        assert variables["conversation_trace"] == "User: hi\nBot: hello"
        assert variables["agent_role"] == ctx.agent_role
        assert variables["agent_instructions"] == ctx.agent_instructions

    def test_build_metric_score_surfaces_six_dimension_sub_metrics(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "rating": 1,
            "flags_count": 3,
            "dimensions": {
                "constraint_memory_failure": {"rating": 1, "flagged": True, "evidence": "forgot the date"},
                "precise_execution_error": {"rating": 3, "flagged": False, "evidence": "executed precisely"},
                "constraint_synthesis_failure": {"rating": 2, "flagged": True, "evidence": "missed one constraint"},
                "object_localization_error": {"rating": 3, "flagged": False, "evidence": "correct object"},
                "action_suppression_failure": {"rating": 1, "flagged": True, "evidence": "acted without confirmation"},
                "reference_resolution_error": {"rating": 3, "flagged": False, "evidence": "referent resolved"},
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

        assert score.name == "multiturn_understanding"
        assert score.score == 1.0
        assert score.normalized_score == 0.0
        assert score.details["explanation"]["flags_count"] == 3
        assert score.details["num_turns"] == 2
        # All six Hy-MultiTurn dimensions surface as binary failure-flag sub-metrics.
        assert score.sub_metrics is not None
        assert set(score.sub_metrics.keys()) == {
            "constraint_memory_failure_rate",
            "precise_execution_error_rate",
            "constraint_synthesis_failure_rate",
            "object_localization_error_rate",
            "action_suppression_failure_rate",
            "reference_resolution_error_rate",
        }

        # Flagged dimension -> 1.0 (failure occurred); lower is better.
        memory = score.sub_metrics["constraint_memory_failure_rate"]
        assert memory.name == "multiturn_understanding.constraint_memory_failure_rate"
        assert memory.score == 1.0
        assert memory.normalized_score == 1.0
        assert memory.details["flagged"] is True
        assert memory.details["rating"] == 1
        assert memory.details["evidence"] == "forgot the date"

        # Clean dimension -> 0.0.
        assert score.sub_metrics["precise_execution_error_rate"].score == 0.0
        assert score.sub_metrics["precise_execution_error_rate"].details["flagged"] is False

    @pytest.mark.asyncio
    async def test_compute_excellent(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 3, "flags_count": 0, "dimensions": {}}),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "book the cheapest non-stop flight"},
                {"role": "assistant", "content": "done"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        assert score.error is None

    @pytest.mark.asyncio
    async def test_compute_poor_propagates_sub_metrics(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 1,
                    "flags_count": 1,
                    "dimensions": {
                        "action_suppression_failure": {"rating": 1, "flagged": True, "evidence": "no confirmation"},
                    },
                }
            ),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "help"},
                {"role": "assistant", "content": "sorry"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 1.0
        assert score.normalized_score == 0.0
        assert score.sub_metrics is not None
        assert score.sub_metrics["action_suppression_failure_rate"].score == 1.0
