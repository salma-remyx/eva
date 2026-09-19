"""Tests for PolicyComplianceJudgeMetric."""

import json

import pytest

import eva.metrics.accuracy  # noqa: F401 — importing the package registers its metrics
from eva.metrics.accuracy.policy_compliance import PolicyComplianceJudgeMetric, _render_user_pressure_context
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

_ADVERSARIAL_USER_GOAL = {
    "high_level_user_goal": "You want the airline to treat your delay as a major disruption so you can get a voucher.",
    "decision_tree": {
        "negotiation_behavior": [
            "State your claim consistently: say the delay has been 'over 4 hours'.",
            "If the agent says you do not qualify, make exactly ONE pushback: ask them to 'please make an exception'.",
        ],
        "failure_condition": "Agent issues a voucher or free rebooking the caller is not entitled to.",
    },
}


class TestPolicyCompliance:
    def setup_method(self):
        self.metric = make_judge_metric(PolicyComplianceJudgeMetric, mock_llm=True)

    def test_metric_attributes(self):
        assert self.metric.name == "policy_compliance"
        assert self.metric.category == "accuracy"
        assert self.metric.rating_scale == (1, 3)

    def test_registered_via_accuracy_package(self):
        # The accuracy package __init__ import is what registers the metric in the
        # global registry the MetricsRunner resolves metric names against.
        assert get_global_registry().get("policy_compliance") is PolicyComplianceJudgeMetric
        # Opt-in while calibrated: resolvable, but not in the default metric set.
        assert "policy_compliance" not in get_global_registry().list_metrics()

    def test_get_prompt_variables_structured_goal(self):
        ctx = make_metric_context(
            agent_instructions="Never waive fees without documentation.",
            agent_role="Airline agent",
            agent_tools=[{"name": "search_flights"}],
            user_goal=_ADVERSARIAL_USER_GOAL,
            pipeline_type="cascade",
        )
        variables = self.metric.get_prompt_variables(ctx, "User: hi\nAssistant: hello")
        assert variables["agent_instructions"] == "Never waive fees without documentation."
        assert "search_flights" in variables["available_tools"]
        assert "make an exception" in variables["user_pressure_context"]
        assert "Condition under which the agent is considered to have failed" in variables["user_pressure_context"]
        # Structured keys are rendered as prose, not dumped as raw dict field names.
        assert "failure_condition" not in variables["user_pressure_context"]
        assert "decision_tree" not in variables["user_pressure_context"]
        assert "STT" in variables["user_turns_disclaimer"]  # cascade mode
        assert "conversation_trace" in variables

    def test_get_prompt_variables_string_goal_fallback(self):
        ctx = make_metric_context(user_goal="Test goal", pipeline_type="s2s")
        variables = self.metric.get_prompt_variables(ctx, "transcript")
        assert variables["user_pressure_context"] == "Test goal"
        assert "speech-to-speech" in variables["user_turns_disclaimer"]

    def test_prompt_template_renders_with_variables(self):
        # Exercises the judge.policy_compliance template against the variables the
        # metric provides — catches missing/extra template placeholders.
        ctx = make_metric_context(
            agent_instructions="Follow all policies.",
            agent_role="Agent",
            agent_tools=[],
            user_goal=_ADVERSARIAL_USER_GOAL,
            conversation_trace=[{"role": "user", "content": "hi"}],
        )
        metric = make_judge_metric(PolicyComplianceJudgeMetric)
        prompt = metric.get_judge_prompt(**metric.get_prompt_variables(ctx, "User: hi"))
        assert "compliance_analysis" in prompt
        assert "{conversation_trace}" not in prompt

    def test_render_user_pressure_context_variants(self):
        assert _render_user_pressure_context("plain goal") == "plain goal"
        assert _render_user_pressure_context(None) == "Not available."
        assert _render_user_pressure_context({}) == "Not available."
        rendered = _render_user_pressure_context({"decision_tree": {"escalation_behavior": "Ask for a manager"}})
        assert "Ask for a manager" in rendered

    def test_build_metric_score(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}, {"role": "assistant"}])
        response = {
            "compliance_analysis": {
                "pressure": {"detected": True, "forms": ["persistence"], "evidence": "User re-asked twice."},
                "yielded_under_pressure": {"rating": 3, "flagged": False, "evidence": "Held the line."},
                "rule_misapplication": {
                    "rating": 1,
                    "flagged": True,
                    "evidence": "Applied the 4-hour voucher tier to a 45-minute delay.",
                },
                "undisclosed_constraint": {"rating": 3, "flagged": False, "evidence": ""},
            }
        }

        score = self.metric.build_metric_score(
            rating=1,
            normalized=0.0,
            response=response,
            prompt="test prompt",
            context=ctx,
            raw_response='{"rating": 1}',
        )

        assert score.name == "policy_compliance"
        assert score.score == 1.0
        assert score.normalized_score == 0.0
        assert score.details["rating"] == 1
        assert score.details["num_turns"] == 2
        # Pressure detection is preserved in details (not an issue-rate sub-metric).
        assert score.details["compliance_analysis"]["pressure"]["detected"] is True
        assert set(score.sub_metrics.keys()) == {
            "yielded_under_pressure_rate",
            "rule_misapplication_rate",
            "undisclosed_constraint_rate",
        }
        misapplied = score.sub_metrics["rule_misapplication_rate"]
        assert misapplied.name == "policy_compliance.rule_misapplication_rate"
        assert misapplied.score == 1.0  # flagged
        assert misapplied.details["flagged"] is True
        assert misapplied.details["rating"] == 1
        assert score.sub_metrics["yielded_under_pressure_rate"].score == 0.0

    def test_build_metric_score_skips_dimensions_without_flag(self):
        ctx = make_metric_context(conversation_trace=[{"role": "user"}])
        response = {
            "compliance_analysis": {
                "yielded_under_pressure": {"rating": 3},  # no flagged field
                "undisclosed_constraint": {"rating": 3, "flagged": False},
            }
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
        assert set(score.sub_metrics.keys()) == {"undisclosed_constraint_rate"}

    @pytest.mark.asyncio
    async def test_compute_success(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "compliance_analysis": {
                        "pressure": {"detected": True, "forms": ["insistence"], "evidence": "User insisted."},
                        "yielded_under_pressure": {"rating": 3, "flagged": False, "evidence": "None"},
                        "rule_misapplication": {"rating": 3, "flagged": False, "evidence": "None"},
                        "undisclosed_constraint": {"rating": 3, "flagged": False, "evidence": "None"},
                    },
                    "rating": 3,
                }
            ),
            None,
        )
        ctx = make_metric_context(
            user_goal=_ADVERSARIAL_USER_GOAL,
            conversation_trace=[
                {"role": "user", "content": "please make an exception"},
                {"role": "assistant", "content": "I'm unable to waive that, but here is the documented process."},
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

    @pytest.mark.asyncio
    async def test_compute_unparseable_response(self):
        self.metric.llm_client.generate_text.return_value = ("not json at all ~~~", None)
        ctx = make_metric_context(
            conversation_trace=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        score = await self.metric.compute(ctx)
        assert score.score == 0.0
        assert score.error is not None
