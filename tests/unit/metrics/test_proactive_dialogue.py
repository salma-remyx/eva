"""Tests for ProactiveDialogueJudgeMetric."""

import json

import pytest

from eva.metrics.experience.proactive_dialogue import ProactiveDialogueJudgeMetric
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestProactiveDialogue:
    def setup_method(self):
        self.metric = make_judge_metric(ProactiveDialogueJudgeMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "proactive_dialogue"
        assert self.metric.category == "experience"
        assert self.metric.rating_scale == (1, 3)

    def test_get_prompt_variables(self):
        ctx = make_metric_context()
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nBot: hello")
        assert variables["conversation_trace"] == "User: hi\nBot: hello"
        assert variables["agent_role"] == ctx.agent_role
        assert variables["user_goal"] == ctx.user_goal
        # Pipeline disclaimers are always provided so the rubric reads cleanly.
        assert variables["user_turns_disclaimer"]
        assert variables["assistant_turns_disclaimer"]

    def test_build_metric_score(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}, {"role": "user"}])
        response = {
            "dimensions": {"target_planning": {"rating": 3, "flagged": False, "evidence": "drove the goal"}},
            "explanation": "strong",
        }

        score = self.metric.build_metric_score(
            rating=3,
            normalized=1.0,
            response=response,
            prompt="test prompt",
            context=ctx,
            raw_response='{"rating": 3}',
        )

        assert score.name == "proactive_dialogue"
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        assert score.details["explanation"]["dimensions"]["target_planning"]["evidence"] == "drove the goal"
        assert score.details["explanation"]["explanation"] == "strong"
        assert score.details["num_turns"] == 3

    def test_build_metric_score_surfaces_dimension_sub_metrics(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "rating": 2,
            "dimensions": {
                "target_planning": {"rating": 3, "flagged": False, "evidence": "drove toward goal"},
                "dialogue_guidance": {"rating": 1, "flagged": True, "evidence": "passive"},
            },
            "explanation": "mixed",
        }

        score = self.metric.build_metric_score(
            rating=2,
            normalized=0.5,
            response=response,
            prompt="test prompt",
            context=ctx,
            raw_response="{...}",
        )

        assert score.sub_metrics is not None
        assert set(score.sub_metrics.keys()) == {"target_planning_rate", "dialogue_guidance_rate"}
        # Binary deficiency flag: 1.0 when the dimension fell short, 0.0 otherwise.
        guidance = score.sub_metrics["dialogue_guidance_rate"]
        assert guidance.name == "proactive_dialogue.dialogue_guidance_rate"
        assert guidance.score == 1.0  # flagged
        assert guidance.normalized_score == 1.0
        assert guidance.details["flagged"] is True
        assert guidance.details["rating"] == 1
        assert guidance.details["evidence"] == "passive"

        planning = score.sub_metrics["target_planning_rate"]
        assert planning.score == 0.0
        assert planning.details["flagged"] is False

    @pytest.mark.asyncio
    async def test_compute_excellent(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 3, "dimensions": {}, "explanation": "proactive"}),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "book flight"},
                {"role": "assistant", "content": "done"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_compute_poor(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 1, "dimensions": {}, "explanation": "passive"}),
            None,
        )
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "help"},
                {"role": "assistant", "content": "ok"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 1.0
        assert score.normalized_score == 0.0


def test_registered_via_experience_package():
    """Importing the existing experience package wires the metric into the global registry.

    This exercises the registry-add call site (the import line in
    ``eva.metrics.experience``) end to end: the metric must be resolvable by
    name and included in the default metric list, exactly like its siblings.
    """
    import eva.metrics.experience  # noqa: F401  -- existing, non-new call-site package
    from eva.metrics.registry import get_global_registry

    registry = get_global_registry()
    assert registry.get("proactive_dialogue") is ProactiveDialogueJudgeMetric
    assert "proactive_dialogue" in registry.list_metrics()
