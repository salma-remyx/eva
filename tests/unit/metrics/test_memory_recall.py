"""Tests for MemoryRecallMetric (long-term conversational memory).

Adapted from RUMBA (arXiv:2607.21447): a fine-grained taxonomy of memory-failure
dimensions scored by an LLM judge over the conversation trace.
"""

import json

import pytest

import eva.metrics  # noqa: F401  (importing the package triggers @register_metric)
from eva.metrics.accuracy.memory_recall import MemoryRecallMetric
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestMemoryRecall:
    def setup_method(self):
        self.metric = make_judge_metric(MemoryRecallMetric, mock_llm=True)

    def test_registered_in_global_registry(self):
        # Exercises the accuracy package wiring: the metric must be registered so
        # the runner can select it by name via --metrics memory_recall.
        assert get_global_registry().get("memory_recall") is MemoryRecallMetric

    def test_metric_attributes(self):
        assert self.metric.name == "memory_recall"
        assert self.metric.category == "accuracy"
        assert self.metric.rating_scale == (1, 3)
        assert self.metric.version == "v0.1"

    def test_get_prompt_variables(self):
        ctx = make_metric_context(
            agent_instructions="Be helpful",
            agent_role="Assistant",
            current_date_time="2026-01-01T00:00:00Z",
            pipeline_type="cascade",
        )
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")
        assert variables["agent_instructions"] == "Be helpful"
        assert variables["agent_role"] == "Assistant"
        assert variables["current_date_time"] == "2026-01-01T00:00:00Z"
        assert variables["conversation_trace"] == "User: hi\nBot: hello"
        assert "STT" in variables["user_turns_disclaimer"]  # cascade disclaimer

    def test_build_metric_score_surfaces_dimension_sub_metrics(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "rating": 1,
            "dimensions": {
                "forgetting_established_facts": {"rating": 1, "flagged": True, "evidence": "re-asked the ID"},
                "inconsistent_cross_turn_reasoning": {"rating": 3, "flagged": False, "evidence": "none"},
                "temporal_reasoning_error": {"rating": 3, "flagged": False, "evidence": "none"},
                "ignoring_standing_constraints": {"rating": 2, "flagged": True, "evidence": "broke the budget"},
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

        assert score.name == "memory_recall"
        assert score.score == 1.0
        assert score.normalized_score == 0.0
        assert score.details["num_turns"] == 2
        assert score.details["explanation"]["dimensions"] == response["dimensions"]
        assert set(score.sub_metrics.keys()) == {
            "forgetting_established_facts_rate",
            "inconsistent_cross_turn_reasoning_rate",
            "temporal_reasoning_error_rate",
            "ignoring_standing_constraints_rate",
        }
        # Binary issue-flag semantics: 1.0 when flagged, 0.0 when clean.
        forget = score.sub_metrics["forgetting_established_facts_rate"]
        assert forget.score == 1.0
        assert forget.details["flagged"] is True
        assert forget.details["rating"] == 1
        consistent = score.sub_metrics["inconsistent_cross_turn_reasoning_rate"]
        assert consistent.score == 0.0
        assert consistent.details["flagged"] is False

    def test_build_metric_score_skips_dimensions_without_flag(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}])
        response = {
            "rating": 3,
            "dimensions": {
                "forgetting_established_facts": {"rating": 3, "flagged": False, "evidence": "none"},
                "temporal_reasoning_error": {"rating": 3},  # no flagged field -> skipped
            },
        }

        score = self.metric.build_metric_score(
            rating=3,
            normalized=1.0,
            response=response,
            prompt="p",
            context=ctx,
            raw_response="{}",
        )

        assert score.sub_metrics is not None
        assert set(score.sub_metrics.keys()) == {"forgetting_established_facts_rate"}

    @pytest.mark.asyncio
    async def test_compute_success(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 3,
                    "dimensions": {
                        "forgetting_established_facts": {"rating": 3, "flagged": False, "evidence": "none"},
                        "inconsistent_cross_turn_reasoning": {"rating": 3, "flagged": False, "evidence": "none"},
                        "temporal_reasoning_error": {"rating": 3, "flagged": False, "evidence": "none"},
                        "ignoring_standing_constraints": {"rating": 3, "flagged": False, "evidence": "none"},
                    },
                }
            ),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "My confirmation number is ABC123."},
                {"role": "assistant", "content": "Got it, ABC123."},
                {"role": "user", "content": "What was my number again?"},
                {"role": "assistant", "content": "Your number is ABC123."},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        assert score.error is None

    @pytest.mark.asyncio
    async def test_compute_empty_transcript(self):
        ctx = make_metric_context(conversation_trace=[])
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert "No transcript" in score.error
