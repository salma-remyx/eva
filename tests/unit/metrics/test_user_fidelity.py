"""Tests for UserFidelityMetric."""

import json

import pytest

from eva.metrics.registry import get_global_registry
from eva.metrics.validation.user_behavioral_fidelity import UserBehavioralFidelityMetric
from eva.metrics.validation.user_fidelity import UserFidelityMetric
from eva.utils.prompt_manager import get_prompt_manager
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

_USER_GOAL = {
    "high_level_user_goal": "Book a flight",
    "starting_utterance": "Hi, I need to book a flight",
    "information_required": "confirmation number ABC123",
    "decision_tree": {
        "must_have_criteria": "Same destination",
        "nice_to_have_criteria": "Window seat",
        "negotiation_behavior": "Accept first offer",
        "resolution_condition": "Flight booked",
        "failure_condition": "No availability",
        "escalation_behavior": "Ask for supervisor",
        "edge_cases": "None",
    },
}

_CRITERIA_RESPONSE = {
    "criteria_analysis": {
        "premature_disclosure": {"analysis": "none", "violated": False},
        "withheld_information": {"analysis": "none", "violated": False},
        "persona_inconsistency": {"analysis": "none", "violated": False},
        "protocol_violation": {"analysis": "none", "violated": False},
    }
}


def _make_ctx(**overrides):
    """Build a MetricContext with the structured user goal the rubric needs."""
    defaults = {
        "user_goal": _USER_GOAL,
        "user_persona": "Friendly traveler",
        "agent_id": "agent_airline",
        "intended_user_turns": {0: "Hi, I need to book a flight"},
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


class TestUserFidelity:
    def setup_method(self):
        self.metric = make_judge_metric(UserFidelityMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "user_fidelity"
        assert self.metric.category == "validation"
        assert self.metric.rating_scale == (0, 1)
        # Opt-in diagnostic: resolvable by name but not part of default runs.
        assert self.metric.exclude_from_default_metrics is True

    def test_opt_in_registration_in_global_registry(self):
        registry = get_global_registry()
        assert registry.get("user_fidelity") is UserFidelityMetric
        assert isinstance(registry.create("user_fidelity"), UserFidelityMetric)
        assert "user_fidelity" not in registry.list_metrics()

    def test_judge_prompt_loaded_from_prompts_directory(self):
        # The template ships in configs/prompts/user_fidelity.yaml under its
        # own top-level key (the PromptManager merges prompt files shallowly
        # by top-level namespace, so a new file cannot extend the "judge"
        # namespace owned by judge.yaml without clobbering it).
        template = get_prompt_manager().get_template("user_fidelity.user_prompt")
        assert "premature disclosure" in template.lower()

        prompt = self.metric.get_judge_prompt(**self.metric.get_prompt_variables(_make_ctx(), "transcript text"))
        assert "User Simulator Instructions" in prompt
        assert "transcript text" in prompt
        assert "Friendly traveler" in prompt

    def test_rubric_grounded_in_same_instructions_as_corruption_metric(self):
        # Both user-evaluation judges must ground their rubric in the exact
        # user-simulator system prompt the simulator was given.
        ctx = _make_ctx()
        corruption = make_judge_metric(UserBehavioralFidelityMetric)
        mine = self.metric.get_prompt_variables(ctx, "transcript")["user_simulator_instructions"]
        theirs = corruption.get_prompt_variables(ctx, "transcript")["user_simulator_instructions"]
        assert mine == theirs

    def test_get_prompt_variables_includes_user_side_text(self):
        variables = self.metric.get_prompt_variables(_make_ctx(), "transcript text")
        assert "Hi, I need to book a flight" in variables["conversation_evidence"]
        assert "transcript text" in variables["conversation_evidence"]
        assert variables["language_display_name"] == "English"

    def test_get_prompt_variables_requires_structured_user_goal(self):
        with pytest.raises(ValueError, match="user_goal"):
            self.metric.get_prompt_variables(_make_ctx(user_goal="Book a flight"), "transcript text")

    def test_build_metric_score_clean(self):
        score = self.metric.build_metric_score(
            rating=1,
            normalized=1.0,
            response=_CRITERIA_RESPONSE,
            prompt="test",
            context=_make_ctx(),
        )

        assert score.score == 1.0
        assert score.normalized_score == 1.0
        assert score.details["user_fidelity_score"] == 1.0
        assert score.details["num_criteria_violated"] == 0
        assert set(score.sub_metrics.keys()) == {
            "premature_disclosure_rate",
            "withheld_information_rate",
            "persona_inconsistency_rate",
            "protocol_violation_rate",
        }
        assert all(sm.score == 0.0 for sm in score.sub_metrics.values())

    def test_build_metric_score_premature_disclosure(self):
        response = json.loads(json.dumps(_CRITERIA_RESPONSE))
        response["criteria_analysis"]["premature_disclosure"] = {
            "analysis": "User volunteered the confirmation number in the opening turn.",
            "violated": True,
        }

        score = self.metric.build_metric_score(
            rating=0,
            normalized=0.0,
            response=response,
            prompt="test",
            context=_make_ctx(),
        )

        # UFS is rubric-driven: 3 of 4 criteria satisfied.
        assert score.score == 0.75
        assert score.details["num_criteria_violated"] == 1
        premature = score.sub_metrics["premature_disclosure_rate"]
        assert premature.score == 1.0
        assert premature.details["violated"] is True
        assert "confirmation number" in premature.details["analysis"]

    def test_build_metric_score_ignores_malformed_and_unknown_entries(self):
        response = {
            "criteria_analysis": {
                "premature_disclosure": {"analysis": "none", "violated": False},
                "withheld_information": "not-a-dict",  # malformed
                "persona_inconsistency": {},  # no violated key
                "protocol_violation": {"analysis": "none", "violated": False},
                "unknown_criterion": {"analysis": "n/a", "violated": True},  # not in the rubric
            }
        }

        score = self.metric.build_metric_score(
            rating=1,
            normalized=1.0,
            response=response,
            prompt="test",
            context=_make_ctx(),
        )

        assert score.score == 1.0
        assert set(score.sub_metrics.keys()) == {"premature_disclosure_rate", "protocol_violation_rate"}

    def test_build_metric_score_falls_back_to_rating_without_usable_criteria(self):
        score = self.metric.build_metric_score(
            rating=1,
            normalized=1.0,
            response={"criteria_analysis": {}},
            prompt="test",
            context=_make_ctx(),
        )

        assert score.score == 1.0
        assert score.sub_metrics is None
        assert score.details["num_criteria_violated"] == 0

    @pytest.mark.asyncio
    async def test_compute_clean(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 1, **_CRITERIA_RESPONSE}),
            None,
        )
        ctx = _make_ctx(
            conversation_trace=[
                {"role": "user", "content": "Hi, I need to book a flight"},
                {"role": "assistant", "content": "Of course, where to?"},
            ]
        )
        score = await self.metric.compute(ctx)
        assert score.error is None
        assert score.score == 1.0
        assert score.normalized_score == 1.0

    @pytest.mark.asyncio
    async def test_compute_reports_violations_from_rubric_verdicts(self):
        # Judge flags every criterion as violated.
        response = json.loads(json.dumps(_CRITERIA_RESPONSE))
        for entry in response["criteria_analysis"].values():
            entry["violated"] = True
        self.metric.llm_client.generate_text.return_value = (json.dumps({"rating": 0, **response}), None)

        ctx = _make_ctx(
            conversation_trace=[
                {"role": "user", "content": "Hi, confirmation ABC123, book me a flight"},
                {"role": "assistant", "content": "Sure, booked."},
            ]
        )
        score = await self.metric.compute(ctx)
        assert score.error is None
        assert score.score == 0.0
        assert score.details["num_criteria_violated"] == 4
        assert score.sub_metrics["premature_disclosure_rate"].score == 1.0
