"""Tests for SimulatorInstructionFollowingMetric.

Covers the graded SimIF-Bench-style scoring: per-goal completion, in-order
progression, early-vs-late goal split, and scenario adherence, plus the
registration wiring through eva.metrics.validation.
"""

import asyncio
import json

import pytest

import eva.metrics
from eva.metrics.registry import get_global_registry
from eva.metrics.validation.simulator_instruction_following import SimulatorInstructionFollowingMetric
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

_USER_GOAL = {
    "high_level_user_goal": "Move the AUS-LAX flight to March 25",
    "decision_tree": {
        "must_have_criteria": [
            "New travel date is 2026-03-25 for AUS-LAX.",
            "Arrival in LAX by 4:00 PM Pacific.",
            "Total rebooking cost is $120 or less.",
        ],
        "nice_to_have_criteria": ["Window seat"],
        "negotiation_behavior": ["Reject standby flights"],
        "resolution_condition": "Flight rebooked",
        "failure_condition": "No suitable flight",
        "escalation_behavior": "Ask for a supervisor",
        "edge_cases": ["Agent offers alternate airports: refuse"],
    },
}

_TRACE = [
    {"turn_id": 0, "role": "user", "content": "I need to move my flight"},
    {"turn_id": 1, "role": "assistant", "content": "Sure, to which date?"},
    {"turn_id": 2, "role": "user", "content": "March 25, and I must arrive by 4 PM"},
    {"turn_id": 3, "role": "assistant", "content": "Done, that is $95 total."},
]


class TestRegistration:
    def test_registered_through_validation_package(self):
        # The wiring lives in eva.metrics.validation.__init__ — importing the
        # package must be enough for the registry to resolve the metric.
        assert hasattr(eva.metrics.validation, "simulator_instruction_following")
        cls = get_global_registry().get("simulator_instruction_following")
        assert cls is SimulatorInstructionFollowingMetric
        assert "simulator_instruction_following" in get_global_registry().list_metrics()

    def test_metric_attributes(self):
        metric = make_judge_metric(SimulatorInstructionFollowingMetric, mock_llm=True)
        assert metric.name == "simulator_instruction_following"
        assert metric.category == "validation"
        assert metric.rating_scale == (0, 3)
        assert metric.version == "v0.1"


class TestPromptVariables:
    def setup_method(self):
        self.metric = make_judge_metric(SimulatorInstructionFollowingMetric, mock_llm=True)

    def test_goals_rendered_as_numbered_list(self):
        ctx = make_metric_context(user_goal=_USER_GOAL, user_persona="Direct traveler", pipeline_type="cascade")
        variables = self.metric.get_prompt_variables(ctx, "transcript text")

        goals = variables["prescribed_goals"]
        assert "1. New travel date is 2026-03-25 for AUS-LAX." in goals
        assert "3. Total rebooking cost is $120 or less." in goals
        # Nice-to-haves must not be listed as goals.
        assert "Window seat" not in goals
        # Scenario block carries the decision-tree behavior rules and persona.
        assert "Direct traveler" in variables["prescribed_scenario"]
        assert "Reject standby flights" in variables["prescribed_scenario"]
        assert "transcripts" in variables["user_turns_disclaimer"]  # cascade variant

    def test_prose_criteria_fall_back_to_goal_extraction(self):
        goal = {"high_level_user_goal": "Rebook", "decision_tree": {"must_have_criteria": "Same date and cheap"}}
        ctx = make_metric_context(user_goal=goal)
        goals = self.metric.get_prompt_variables(ctx, "transcript text")["prescribed_goals"]
        assert "Same date and cheap" in goals
        assert "goal_id 1..N" in goals

    def test_missing_criteria_fall_back_to_scenario(self):
        ctx = make_metric_context(user_goal={"high_level_user_goal": "Rebook"})
        goals = self.metric.get_prompt_variables(ctx, "transcript text")["prescribed_goals"]
        assert "No explicit must-have criteria" in goals

    def test_s2s_disclaimer_used_for_audio_native(self):
        ctx = make_metric_context(user_goal=_USER_GOAL, pipeline_type="s2s")
        variables = self.metric.get_prompt_variables(ctx, "transcript text")
        assert "speech-to-speech" in variables["user_turns_disclaimer"]

    def test_judge_prompt_renders_with_variables(self):
        ctx = make_metric_context(user_goal=_USER_GOAL)
        prompt = self.metric.get_judge_prompt(**self.metric.get_prompt_variables(ctx, "transcript text"))
        assert "Prescribed Goals, in required order" in prompt
        assert "1. New travel date is 2026-03-25 for AUS-LAX." in prompt


class TestBuildMetricScore:
    def setup_method(self):
        self.metric = make_judge_metric(SimulatorInstructionFollowingMetric, mock_llm=True)
        self.ctx = make_metric_context(user_goal=_USER_GOAL, conversation_trace=_TRACE)

    def test_perfect_following(self):
        response = {
            "goal_analysis": [
                {"goal_id": 1, "completed": True, "turn_index": 2},
                {"goal_id": 2, "completed": True, "turn_index": 2},
                {"goal_id": 3, "completed": True, "turn_index": 4},
            ],
            "scenario_adherence": {"violations": [], "analysis": "stayed on scenario"},
        }
        score = self.metric.build_metric_score(
            rating=3, normalized=1.0, response=response, prompt="p", context=self.ctx
        )

        assert score.score == 3.0
        subs = score.sub_metrics
        assert subs is not None
        assert subs["goal_completion_accuracy"].score == 1.0
        assert subs["ordered_goal_completion_accuracy"].score == 1.0
        assert subs["early_goal_completion_accuracy"].score == 1.0
        assert subs["late_goal_completion_accuracy"].score == 1.0
        assert subs["scenario_violation_rate"].score == 0.0
        assert score.details["early_late_completion_gap"] == 0.0

    def test_late_goal_failure_yields_degradation_gap(self):
        # Goals 1-2 (early half) completed; goal 3 (late half) missed.
        response = {
            "goal_analysis": [
                {"goal_id": 1, "completed": True, "turn_index": 2},
                {"goal_id": 2, "completed": True, "turn_index": 2},
                {"goal_id": 3, "completed": False, "turn_index": None},
            ],
            "scenario_adherence": {"violations": [], "analysis": ""},
        }
        score = self.metric.build_metric_score(
            rating=1, normalized=1 / 3, response=response, prompt="p", context=self.ctx
        )

        subs = score.sub_metrics
        assert subs is not None
        assert subs["goal_completion_accuracy"].score == round(2 / 3, 3)
        assert subs["ordered_goal_completion_accuracy"].score == round(2 / 3, 3)
        assert subs["early_goal_completion_accuracy"].score == 1.0
        assert subs["late_goal_completion_accuracy"].score == 0.0
        assert score.details["early_late_completion_gap"] == 1.0

    def test_out_of_order_completion_truncates_ordered_progress(self):
        # Goal 1 completed late (turn 6) and goal 2 earlier (turn 2): the
        # in-order prefix stops at the first backwards turn index.
        response = {
            "goal_analysis": [
                {"goal_id": 1, "completed": True, "turn_index": 6},
                {"goal_id": 2, "completed": True, "turn_index": 2},
                {"goal_id": 3, "completed": True, "turn_index": 8},
            ],
            "scenario_adherence": {"violations": [], "analysis": ""},
        }
        score = self.metric.build_metric_score(
            rating=2, normalized=2 / 3, response=response, prompt="p", context=self.ctx
        )

        subs = score.sub_metrics
        assert subs is not None
        assert subs["goal_completion_accuracy"].score == 1.0
        assert subs["ordered_goal_completion_accuracy"].score == round(1 / 3, 3)

    def test_goal_ids_resorted_to_prescribed_order(self):
        # Judge returns entries out of order; scoring must follow goal_id order.
        # Unsorted, turn 8 -> 4 would truncate the prefix at 1/3.
        response = {
            "goal_analysis": [
                {"goal_id": 3, "completed": True, "turn_index": 8},
                {"goal_id": 1, "completed": True, "turn_index": 4},
                {"goal_id": 2, "completed": True, "turn_index": 6},
            ],
            "scenario_adherence": {"violations": [], "analysis": ""},
        }
        score = self.metric.build_metric_score(
            rating=2, normalized=2 / 3, response=response, prompt="p", context=self.ctx
        )

        assert score.sub_metrics["ordered_goal_completion_accuracy"].score == 1.0

    def test_scenario_violation_rate_over_user_turns(self):
        response = {
            "goal_analysis": [],
            "scenario_adherence": {
                "violations": [
                    {"turn_index": 2, "description": "invented a hotel booking"},
                    {"turn_index": 2, "dup": True},
                ],
                "analysis": "went off-script",
            },
        }
        score = self.metric.build_metric_score(
            rating=0, normalized=0.0, response=response, prompt="p", context=self.ctx
        )

        # 1 distinct violating turn out of 2 user turns in the trace.
        subs = score.sub_metrics
        assert subs is not None
        assert subs["scenario_violation_rate"].score == 0.5
        assert subs["scenario_violation_rate"].details["violating_turns"] == [2]
        # No goals analyzed -> no goal sub-metrics, no early/late split.
        assert "goal_completion_accuracy" not in subs
        assert score.details["early_late_completion_gap"] == 0.0

    def test_malformed_goal_entries_skipped(self):
        response = {
            "goal_analysis": ["not-a-dict", {"goal_id": 1, "completed": True, "turn_index": 2}, "also-not-a-dict"],
            "scenario_adherence": {"violations": "not-a-list", "analysis": ""},
        }
        score = self.metric.build_metric_score(
            rating=2, normalized=2 / 3, response=response, prompt="p", context=self.ctx
        )

        subs = score.sub_metrics
        assert subs is not None
        # Only the well-formed dict counts as a goal.
        assert subs["goal_completion_accuracy"].score == 1.0
        assert subs["goal_completion_accuracy"].details["num_goals"] == 1
        # Single goal -> no early/late split emitted.
        assert "early_goal_completion_accuracy" not in subs
        assert "late_goal_completion_accuracy" not in subs
        # A non-list violations field is treated as no violations.
        assert subs["scenario_violation_rate"].score == 0.0


class TestCompute:
    def setup_method(self):
        self.metric = make_judge_metric(SimulatorInstructionFollowingMetric, mock_llm=True)

    def test_compute_end_to_end(self):
        # asyncio.run keeps this runnable without pytest-asyncio (same pattern
        # as tests/unit/assistant/test_pipecat_server.py).
        self.metric.llm_client.generate_text.return_value = (
            json.dumps(
                {
                    "goal_analysis": [
                        {
                            "goal_id": 1,
                            "description": "date",
                            "completed": True,
                            "turn_index": 2,
                            "evidence": "gave date",
                        },
                        {
                            "goal_id": 2,
                            "description": "arrival",
                            "completed": True,
                            "turn_index": 2,
                            "evidence": "gave time",
                        },
                        {
                            "goal_id": 3,
                            "description": "cost",
                            "completed": False,
                            "turn_index": None,
                            "evidence": "never confirmed",
                        },
                    ],
                    "scenario_adherence": {
                        "violations": [{"turn_index": 2, "description": "invented a hotel request"}],
                        "analysis": "mostly on scenario",
                    },
                    "rating": 1,
                }
            ),
            None,
        )
        ctx = make_metric_context(user_goal=_USER_GOAL, conversation_trace=_TRACE)
        score = asyncio.run(self.metric.compute(ctx))

        assert score.error is None
        assert score.score == 1.0
        assert score.normalized_score == pytest.approx(1 / 3)
        assert score.sub_metrics is not None
        assert score.sub_metrics["goal_completion_accuracy"].score == round(2 / 3, 3)
        assert score.sub_metrics["scenario_violation_rate"].score == 0.5
        assert score.details["explanation"] == "mostly on scenario"

    def test_compute_survives_unparseable_judge_response(self):
        self.metric.llm_client.generate_text.return_value = ("not json at all", None)
        ctx = make_metric_context(user_goal=_USER_GOAL, conversation_trace=_TRACE)
        score = asyncio.run(self.metric.compute(ctx))

        assert score.error == "Failed to parse judge response"
        assert score.score == 0.0
