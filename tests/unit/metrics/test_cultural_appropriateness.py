"""Tests for CulturalAppropriatenessMetric.

Integration-level: goes through the global metric registry (the same one every
dispatch uses), the shared judge-prompt machinery in configs/prompts/judge.yaml,
and the cultural grounding profiles wired into eva.utils.culture.
"""

import json
from unittest.mock import AsyncMock

import pytest

import eva.metrics  # noqa: F401  (imports register every metric)
from eva.metrics.base import MetricContext
from eva.metrics.diagnostic.cultural_appropriateness import CulturalAppropriatenessMetric
from eva.metrics.registry import get_global_registry
from eva.utils.cultural_grounding import cultural_aspect_keys

from .conftest import make_judge_metric, make_metric_context

SAMPLE_TURNS = [
    {"turn_id": 1, "role": "user", "content": "Bonjour, je voudrais changer mon vol."},
    {"turn_id": 1, "role": "assistant", "content": "Salut ! Tu peux me donner ton code ?"},
    {"turn_id": 2, "role": "user", "content": "Certainement. Le code est ABC123, pour le 14/06."},
    {"turn_id": 2, "role": "assistant", "content": "Done! You're rebooked for 06/14 at 6:30 PM."},
]


@pytest.fixture
def metric():
    return make_judge_metric(CulturalAppropriatenessMetric)


@pytest.fixture
def fr_context() -> MetricContext:
    return make_metric_context(language="fr", conversation_trace=SAMPLE_TURNS)


def _judge_response(language: str, rating: int, violated: dict[str, bool]) -> str:
    # Every aspect gets an entry so each one produces a sub-metric; aspects the
    # judge omits are skipped entirely (same semantics as user_behavioral_fidelity).
    aspects = cultural_aspect_keys(language)
    analysis = {
        aspect: {"analysis": f"analysis for {aspect}", "violated": violated.get(aspect, False)} for aspect in aspects
    }
    return json.dumps({"aspect_analysis": analysis, "rating": rating})


class TestRegistryIntegration:
    def test_registered_on_global_registry(self):
        registry = get_global_registry()
        assert registry.get("cultural_appropriateness") is CulturalAppropriatenessMetric

    def test_opt_in_only(self):
        """The judge is excluded from default runs, like tts_fidelity."""
        assert "cultural_appropriateness" not in get_global_registry().list_metrics()
        assert CulturalAppropriatenessMetric.exclude_from_default_metrics is True


class TestPromptVariables:
    def test_prompt_carries_cultural_brief(self, metric, fr_context):
        variables = metric.get_prompt_variables(fr_context, "TRANSCRIPT")
        assert variables["language_display_name"] == "European French"
        for aspect in cultural_aspect_keys("fr"):
            assert aspect in variables["cultural_brief"]
        assert "TRANSCRIPT" in variables["conversation"]

    def test_judge_prompt_renders_from_judge_yaml(self, metric, fr_context):
        prompt = metric.get_judge_prompt(**metric.get_prompt_variables(fr_context, "TRANSCRIPT"))
        assert "Cultural grounding brief" in prompt
        assert "formality_register" in prompt
        assert "TRANSCRIPT" in prompt


class TestCompute:
    @pytest.mark.asyncio
    async def test_scores_rating_and_aspect_sub_metrics(self, metric, fr_context):
        """A tutoiement + AM/PM answer should rate low and flag the violated aspects."""
        metric.llm_client.generate_text = AsyncMock(
            return_value=(
                _judge_response("fr", 1, {"formality_register": True, "date_time_format": True}),
                None,
            )
        )
        result = await metric.compute(fr_context)

        assert result.error is None
        assert result.score == 1.0
        assert result.normalized_score == 0.0
        assert result.details["language"] == "fr"
        sub = result.sub_metrics or {}
        assert sorted(sub) == [
            "date_time_format_rate",
            "formality_register_rate",
            "greeting_politeness_rate",
            "number_format_rate",
        ]
        assert sub["formality_register_rate"].score == 1.0
        assert sub["date_time_format_rate"].score == 1.0
        assert sub["number_format_rate"].score == 0.0

    @pytest.mark.asyncio
    async def test_competent_agent_scores_full(self, metric, fr_context):
        metric.llm_client.generate_text = AsyncMock(return_value=(_judge_response("fr", 3, {}), None))
        result = await metric.compute(fr_context)

        assert result.error is None
        assert result.score == 3.0
        assert result.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_ungrounded_language_returns_error_score(self, metric):
        context = make_metric_context(language="en", conversation_trace=SAMPLE_TURNS)
        result = await metric.compute(context)

        assert result.error is not None
        assert "No cultural grounding profile" in result.error
        assert result.score == 0.0
