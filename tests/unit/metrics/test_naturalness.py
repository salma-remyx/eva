"""Tests for NaturalnessJudgeMetric.

Covers both the metric behaviour and its integration with the metric registry:
importing the ``eva.metrics`` package (via the non-new registry module) must
register ``naturalness`` so MetricsRunner can select it by name with ``--metrics``.
"""

import json

import pytest

from eva.metrics.experience.naturalness import NaturalnessJudgeMetric
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

_DIMENSIONS = (
    "emotional_naturalness",
    "interpersonal_stance",
    "prosody_appropriateness",
    "dialect_language_consistency",
    "relationship_appropriateness",
)


class TestNaturalnessIntegration:
    def test_metric_is_registered_by_name(self):
        # Proves the experience package wires the metric in: importing
        # eva.metrics.registry runs the subpackage __init__ that registers it,
        # so MetricsRunner.registry.create("naturalness") resolves.
        cls = get_global_registry().get("naturalness")
        assert cls is NaturalnessJudgeMetric

    def test_metric_listed_by_default(self):
        assert "naturalness" in get_global_registry().list_metrics()


class TestNaturalness:
    def setup_method(self):
        self.metric = make_judge_metric(NaturalnessJudgeMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "naturalness"
        assert self.metric.category == "experience"
        assert self.metric.rating_scale == (1, 3)
        assert self.metric.version == "v0.1"

    def test_get_prompt_variables(self):
        ctx = make_metric_context()
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")
        assert variables["conversation_trace"] == "User: hi\nBot: hello"
        # Disclaimers are populated for both pipeline flavours.
        assert isinstance(variables["user_turns_disclaimer"], str)
        assert isinstance(variables["assistant_turns_disclaimer"], str)

    def test_build_metric_score_surfaces_one_sub_metric_per_dimension(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        dimensions = {
            "emotional_naturalness": {"rating": 2, "flagged": True, "evidence": "flat tone"},
            "interpersonal_stance": {"rating": 3, "flagged": False, "evidence": ""},
            "prosody_appropriateness": {"rating": 3, "flagged": False, "evidence": ""},
            "dialect_language_consistency": {"rating": 2, "flagged": True, "evidence": "register drift"},
            "relationship_appropriateness": {"rating": 3, "flagged": False, "evidence": ""},
        }

        score = self.metric.build_metric_score(
            rating=2,
            normalized=0.5,
            response={"rating": 2, "explanation": "minor issues", "dimensions": dimensions},
            prompt="test prompt",
            context=ctx,
            raw_response="{...}",
        )

        assert score.name == "naturalness"
        assert score.score == 2.0
        assert score.normalized_score == 0.5
        assert score.details["explanation"] == "minor issues"
        assert set(score.sub_metrics.keys()) == {f"{d}_rate" for d in _DIMENSIONS}

        flagged = score.sub_metrics["emotional_naturalness_rate"]
        assert flagged.name == "naturalness.emotional_naturalness_rate"
        assert flagged.score == 1.0  # issue present
        assert flagged.details["flagged"] is True
        assert flagged.details["evidence"] == "flat tone"

        clean = score.sub_metrics["interpersonal_stance_rate"]
        assert clean.score == 0.0
        assert clean.details["flagged"] is False

    @pytest.mark.asyncio
    async def test_compute_natural(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 3,
                    "explanation": "fluent and appropriate",
                    "dimensions": {d: {"rating": 3, "flagged": False, "evidence": ""} for d in _DIMENSIONS},
                }
            ),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "thanks"},
                {"role": "assistant", "content": "you're welcome"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        # No flagged dimensions -> all sub-metrics report 0.0 (no issue).
        assert all(sm.score == 0.0 for sm in score.sub_metrics.values())

    @pytest.mark.asyncio
    async def test_compute_unnatural(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 1,
                    "explanation": "robotic and dismissive",
                    "dimensions": {
                        "emotional_naturalness": {"rating": 1, "flagged": True, "evidence": "flat"},
                        "interpersonal_stance": {"rating": 1, "flagged": True, "evidence": "curt"},
                        "prosody_appropriateness": {"rating": 3, "flagged": False, "evidence": ""},
                        "dialect_language_consistency": {"rating": 3, "flagged": False, "evidence": ""},
                        "relationship_appropriateness": {"rating": 3, "flagged": False, "evidence": ""},
                    },
                }
            ),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "i need help"},
                {"role": "assistant", "content": "whatever"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 1.0
        assert score.normalized_score == 0.0
        assert score.sub_metrics["emotional_naturalness_rate"].score == 1.0
        assert score.sub_metrics["prosody_appropriateness_rate"].score == 0.0
