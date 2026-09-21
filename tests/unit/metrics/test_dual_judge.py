"""Tests for the citation-grounded dual-judge metrics."""

import asyncio
import importlib.util
import json
from pathlib import Path

from eva.metrics.accuracy.dual_judge import (
    QUALITY_DIMENSION_KEYS,
    ConversationQualityMetric,
    ScenarioRuleComplianceMetric,
    validate_citations,
)
from eva.metrics.registry import get_global_registry
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

TRACE = [
    {"turn_id": 1, "role": "user", "content": "I need to move my flight from March 20 to March 25."},
    {"turn_id": 1, "role": "assistant", "content": "I can help with that. What is your confirmation number?"},
    {"turn_id": 2, "role": "user", "content": "It is ABC123."},
    {"turn_id": 2, "role": "assistant", "content": "Your flight is now moved to March 25, keeping your window seat."},
]

USER_GOAL = {
    "high_level_user_goal": "Move the AUS to LAX flight from March 20 to March 25",
    "decision_tree": {
        "must_have_criteria": ["New travel date is 2026-03-25."],
        "escalation_behavior": "Escalate only if the rebooking fails.",
    },
}


def _ctx(**overrides):
    defaults = {"conversation_trace": TRACE, "user_goal": USER_GOAL}
    defaults.update(overrides)
    return make_metric_context(**defaults)


class TestValidateCitations:
    def test_verbatim_quote_is_grounded(self):
        validated = validate_citations(
            [{"turn": 2, "role": "assistant", "quote": "Your flight is now moved to March 25"}], TRACE
        )
        assert validated[0]["valid"] is True
        assert validated[0]["location_match"] is True

    def test_quote_matching_is_whitespace_and_case_tolerant(self):
        validated = validate_citations(
            [{"turn": 1, "role": "user", "quote": "  move MY   flight from march 20 "}], TRACE
        )
        assert validated[0]["valid"] is True

    def test_fabricated_quote_is_ungrounded(self):
        validated = validate_citations([{"turn": 1, "role": "user", "quote": "I want a full refund now"}], TRACE)
        assert validated[0]["valid"] is False
        assert "not found" in validated[0]["reason"]

    def test_real_quote_with_wrong_pointer_is_valid_but_mismatched(self):
        validated = validate_citations([{"turn": 1, "role": "assistant", "quote": "It is ABC123."}], TRACE)
        assert validated[0]["valid"] is True
        assert validated[0]["location_match"] is False

    def test_empty_quote_is_invalid(self):
        validated = validate_citations([{"turn": 1, "role": "user", "quote": "   "}], TRACE)
        assert validated[0]["valid"] is False


class TestScenarioRuleCompliance:
    def setup_method(self):
        self.metric = make_judge_metric(ScenarioRuleComplianceMetric, mock_llm=True)

    def test_prompt_variables_include_rules_and_instructions(self):
        variables = self.metric.get_prompt_variables(_ctx(), "TRACE")
        assert "must_have_criteria" in variables["scenario_rules"]
        assert "New travel date is 2026-03-25." in variables["scenario_rules"]
        # The simulator's opener is conversation, not a rule — it must not leak in.
        assert "starting_utterance" not in variables["scenario_rules"]
        assert variables["agent_instructions"] == "Test instructions"
        assert variables["conversation_trace"] == "TRACE"

    def test_string_user_goal_falls_back_to_goal_only_payload(self):
        variables = self.metric.get_prompt_variables(_ctx(user_goal="Test goal"), "TRACE")
        assert "Test goal" in variables["scenario_rules"]

    def test_build_metric_score_reports_rule_and_citation_sub_metrics(self):
        response = {
            "rules": [
                {
                    "rule": "New travel date is 2026-03-25.",
                    "followed": True,
                    "explanation": "Agent rebooked to March 25.",
                    "citations": [{"turn": 2, "role": "assistant", "quote": "moved to March 25"}],
                },
                {
                    "rule": "Escalate only if the rebooking fails.",
                    "followed": False,
                    "explanation": "Agent escalated although the rebooking succeeded.",
                    "citations": [{"turn": 1, "role": "user", "quote": "a quote that never happened"}],
                },
            ],
            "rating": 1,
        }

        score = self.metric.build_metric_score(
            rating=1, normalized=0.0, response=response, prompt="test prompt", context=_ctx(), raw_response="{}"
        )

        assert score.name == "scenario_rule_compliance"
        assert score.score == 1.0
        assert score.details["num_rules"] == 2
        assert score.details["num_rules_followed"] == 1
        assert score.sub_metrics is not None
        assert score.sub_metrics["rule_violation_rate"].score == 0.5
        # One grounded citation, one fabricated one.
        assert score.sub_metrics["citation_grounding"].score == 0.5
        assert score.details["citation_grounding"] == 0.5
        assert score.details["ungrounded_sources"] == ["Escalate only if the rebooking fails."]

    def test_compute_end_to_end_with_citations(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "rules": [
                        {
                            "rule": "New travel date is 2026-03-25.",
                            "followed": True,
                            "explanation": "Rebooked as asked.",
                            "citations": [{"turn": 2, "role": "assistant", "quote": "moved to March 25"}],
                        }
                    ],
                    "rating": 3,
                }
            ),
            None,
        )

        score = asyncio.run(self.metric.compute(_ctx()))

        assert score.error is None
        assert score.score == 3.0
        assert score.normalized_score == 1.0
        assert score.sub_metrics is not None
        assert score.sub_metrics["citation_grounding"].score == 1.0
        assert score.sub_metrics["rule_violation_rate"].score == 0.0


class TestConversationQuality:
    def setup_method(self):
        self.metric = make_judge_metric(ConversationQualityMetric, mock_llm=True)

    def test_judge_is_task_blind(self):
        variables = self.metric.get_prompt_variables(_ctx(), "TRACE")
        # Only the transcript: no goal, no rules, no agent config may reach this judge.
        assert variables == {"conversation_trace": "TRACE"}

    def test_build_metric_score_surfaces_dimension_and_citation_sub_metrics(self):
        dimensions = {
            key: {
                "quality_issue": key in ("clarity", "caller_experience"),
                "rating": 2 if key in ("clarity", "caller_experience") else 3,
                "explanation": "issue" if key in ("clarity", "caller_experience") else "clean",
                "citations": [
                    {
                        "turn": 1,
                        "role": "assistant",
                        "quote": "I can help with that" if key != "coherence" else "not in the transcript",
                    }
                ],
            }
            for key in QUALITY_DIMENSION_KEYS
        }

        score = self.metric.build_metric_score(
            rating=2,
            normalized=0.5,
            response={"dimensions": dimensions, "rating": 2},
            prompt="test prompt",
            context=_ctx(),
            raw_response="{}",
        )

        assert score.name == "conversation_quality"
        assert score.sub_metrics is not None
        assert set(score.sub_metrics) == {f"{key}_rate" for key in QUALITY_DIMENSION_KEYS} | {"citation_grounding"}
        assert score.sub_metrics["clarity_rate"].score == 1.0
        assert score.sub_metrics["responsiveness_rate"].score == 0.0
        # Three of four dimensions cite a real message; coherence cites a fabrication.
        assert score.sub_metrics["citation_grounding"].score == 0.75
        assert score.details["ungrounded_sources"] == ["coherence"]

    def test_compute_end_to_end_task_blind(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "dimensions": {
                        key: {
                            "quality_issue": False,
                            "rating": 3,
                            "explanation": "no issue",
                            "citations": [{"turn": 1, "role": "user", "quote": "move my flight"}],
                        }
                        for key in QUALITY_DIMENSION_KEYS
                    },
                    "rating": 3,
                }
            ),
            None,
        )

        score = asyncio.run(self.metric.compute(_ctx()))

        assert score.error is None
        assert score.normalized_score == 1.0
        assert score.sub_metrics is not None
        assert score.sub_metrics["citation_grounding"].score == 1.0
        # The judge prompt really was built from the transcript alone.
        prompt = score.details["judge_prompt"]
        assert "I need to move my flight" in prompt
        assert "must_have_criteria" not in prompt


class TestDualJudgeWiring:
    """Integration: the metrics are registered and reachable from the text-only runner."""

    def test_metrics_resolvable_through_global_registry(self):
        import eva.metrics.accuracy  # noqa: F401  (import registers the metric modules)

        registry = get_global_registry()
        assert isinstance(registry.create("scenario_rule_compliance"), ScenarioRuleComplianceMetric)
        assert isinstance(registry.create("conversation_quality"), ConversationQualityMetric)
        # Opt-in only: the default voice-pipeline metric set is unchanged.
        assert "scenario_rule_compliance" not in registry.list_metrics()
        assert "conversation_quality" not in registry.list_metrics()

    def test_run_text_only_offers_dual_judge_metrics(self):
        script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_text_only.py"
        spec = importlib.util.spec_from_file_location("run_text_only_under_test", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert "scenario_rule_compliance" in module.TEXT_COMPATIBLE_METRICS
        assert "conversation_quality" in module.TEXT_COMPATIBLE_METRICS

        registry = get_global_registry()
        for name in ("scenario_rule_compliance", "conversation_quality"):
            assert registry.get(name) is not None, f"{name} must be resolvable when run_text_only requests it"
