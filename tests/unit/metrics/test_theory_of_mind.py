"""Tests for TheoryOfMindJudgeMetric."""

import json

import pytest

import eva.metrics.experience  # noqa: F401  (ensures @register_metric has run)
from eva.metrics.experience.theory_of_mind import TheoryOfMindJudgeMetric
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context


class TestTheoryOfMind:
    def setup_method(self):
        self.metric = make_judge_metric(TheoryOfMindJudgeMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "theory_of_mind"
        assert self.metric.category == "experience"
        assert self.metric.rating_scale == (1, 3)
        assert self.metric.version == "v0.1"

    def test_registered_in_global_registry(self):
        """The package import wires the metric into the registry MetricsRunner uses."""
        registry = get_global_registry()
        assert registry.get("theory_of_mind") is TheoryOfMindJudgeMetric
        assert "theory_of_mind" in registry.list_metrics()
        assert isinstance(registry.create("theory_of_mind"), TheoryOfMindJudgeMetric)

    def test_get_prompt_variables_exposes_latent_state_oracle(self):
        ctx = make_metric_context(user_goal="rebook but avoid fees", user_persona="conflict-averse")
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nAssistant: hello")
        # The ground-truth latent state (goal/persona) is passed to the judge.
        assert variables["user_goal"] == "rebook but avoid fees"
        assert variables["user_persona"] == "conflict-averse"
        assert variables["conversation_trace"] == "User: hi\nAssistant: hello"

    def test_build_metric_score_surfaces_dimension_sub_metrics(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "rating": 1,
            "flags_count": 2,
            "dimensions": {
                "missed_latent_intent": {"rating": 1, "flagged": True, "evidence": "acted on wrong goal"},
                "pseudo_consensus_unrecognized": {"rating": 2, "flagged": True, "evidence": "took 'I guess' as yes"},
                "knowledge_state_misread": {"rating": 3, "flagged": False, "evidence": "matched user knowledge"},
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

        assert score.name == "theory_of_mind"
        assert score.score == 1.0
        assert score.normalized_score == 0.0
        assert score.details["explanation"]["flags_count"] == 2
        assert score.details["num_turns"] == 2

        assert score.sub_metrics is not None
        assert set(score.sub_metrics.keys()) == {
            "missed_latent_intent_rate",
            "pseudo_consensus_unrecognized_rate",
            "knowledge_state_misread_rate",
        }
        # Binary issue-flag: 1.0 when the ToM failure occurred, 0.0 when clean; lower is better.
        pseudo = score.sub_metrics["pseudo_consensus_unrecognized_rate"]
        assert pseudo.name == "theory_of_mind.pseudo_consensus_unrecognized_rate"
        assert pseudo.score == 1.0  # flagged
        assert pseudo.details["flagged"] is True
        assert pseudo.details["evidence"] == "took 'I guess' as yes"

        clean = score.sub_metrics["knowledge_state_misread_rate"]
        assert clean.score == 0.0
        assert clean.details["flagged"] is False

    @pytest.mark.asyncio
    async def test_compute_strong_tom(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 3, "flags_count": 0, "dimensions": {}}),
            None,
        )
        ctx = make_metric_context(
            user_goal="cancel a booking",
            user_persona="direct",
            conversation_trace=[
                {"role": "user", "content": "I need to cancel"},
                {"role": "assistant", "content": "done"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        assert score.error is None

    @pytest.mark.asyncio
    async def test_compute_poor_tom_normalizes_to_zero(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rating": 1,
                    "flags_count": 2,
                    "dimensions": {
                        "missed_latent_intent": {"flagged": True, "rating": 1, "evidence": "wrong goal"},
                        "pseudo_consensus_unrecognized": {"flagged": True, "rating": 2, "evidence": "lukewarm yes"},
                    },
                }
            ),
            None,
        )
        ctx = make_metric_context(
            user_goal="change a flight",
            conversation_trace=[
                {"role": "user", "content": "help"},
                {"role": "assistant", "content": "okay"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 1.0
        assert score.normalized_score == 0.0
        assert score.sub_metrics is not None
        assert "pseudo_consensus_unrecognized_rate" in score.sub_metrics

    @pytest.mark.asyncio
    async def test_compute_empty_transcript_returns_error(self):
        ctx = make_metric_context(conversation_trace=[])
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.error is not None
