"""Tests for the derived tool-argument reference and its faithfulness wiring.

The derived reference is adapted from DeRef (arXiv:2610.18497): the judge gets
an explicit per-argument reference instead of having to re-derive one from the
raw tool-schema blob. These tests cover the builder itself and the wiring into
FaithfulnessJudgeMetric's prompt variables and rendered judge prompt.
"""

import asyncio
import json

from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.tool_argument_reference import build_tool_argument_reference
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

AGENT_TOOLS = [
    {
        "name": "rebook_flight",
        "description": "Rebook a passenger onto another flight.",
        "tool_type": "write",
        "required_parameters": [{"name": "reservation_id", "type": "string", "description": "Booking id"}],
        "optional_parameters": [
            {
                "name": "rebooking_type",
                "type": "string",
                "description": "Reason for the rebooking.",
                "enum": ["voluntary", "same_day", "irrops_delay", "missed_flight_passenger_fault"],
            },
            {"name": "notify_channel", "type": "string", "description": "How to notify.", "default": "email"},
        ],
    }
]

TRACE = [
    {"turn_id": 1, "role": "user", "content": "My reservation is ABC123 and I missed my connection."},
    {
        "turn_id": 2,
        "type": "tool_call",
        "tool_name": "rebook_flight",
        "parameters": {
            "reservation_id": "ABC123",
            "rebooking_type": "same_day",
            "notify_channel": "email",
            "agent_note": "passenger was rebooked on the next available flight",
        },
    },
]


class TestBuildToolArgumentReference:
    def test_enum_member_is_referenced_as_schema_allowed(self):
        reference = build_tool_argument_reference(AGENT_TOOLS, TRACE)
        assert "rebooking_type = \"same_day\"" in reference
        assert "allowed by schema" in reference
        assert "voluntary" in reference  # the allowed values are spelled out

    def test_value_with_no_source_is_marked_unreferenced(self):
        reference = build_tool_argument_reference(AGENT_TOOLS, TRACE)
        # agent_note is undeclared and appears nowhere else in the trace.
        line = next(ln for ln in reference.splitlines() if ln.startswith("- agent_note"))
        assert "not among the parameters declared for this tool" in line
        assert "no matching text" in line

    def test_declared_default_is_referenced(self):
        reference = build_tool_argument_reference(AGENT_TOOLS, TRACE)
        line = next(ln for ln in reference.splitlines() if ln.startswith("- notify_channel"))
        assert "matches the schema-declared default" in line

    def test_value_traced_to_user_turn(self):
        reference = build_tool_argument_reference(AGENT_TOOLS, TRACE)
        line = next(ln for ln in reference.splitlines() if ln.startswith("- reservation_id"))
        assert "appears in user turn 1" in line

    def test_value_traced_to_tool_response(self):
        trace = TRACE + [
            {
                "turn_id": 3,
                "type": "tool_response",
                "tool_name": "get_reservation",
                "tool_response": {"reservation_id": "ABC123", "flights": ["UA100"]},
            },
            {
                "turn_id": 4,
                "type": "tool_call",
                "tool_name": "rebook_flight",
                "parameters": {"reservation_id": "ABC123"},
            },
        ]
        reference = build_tool_argument_reference(AGENT_TOOLS, trace)
        assert "get_reservation response (turn 3)" in reference

    def test_value_outside_the_enum_is_called_out(self):
        trace = [
            {"turn_id": 1, "role": "user", "content": "rebook me"},
            {
                "turn_id": 2,
                "type": "tool_call",
                "tool_name": "rebook_flight",
                "parameters": {"reservation_id": "ABC123", "rebooking_type": "same_day_change"},
            },
        ]
        reference = build_tool_argument_reference(AGENT_TOOLS, trace)
        line = next(ln for ln in reference.splitlines() if ln.startswith("- rebooking_type"))
        assert "NOT one of the schema's allowed values" in line

    def test_no_tool_calls_yields_empty_reference(self):
        assert build_tool_argument_reference(AGENT_TOOLS, [{"turn_id": 1, "role": "user", "content": "hi"}]) == ""

    def test_unknown_tool_never_claims_its_parameters_are_undeclared(self):
        trace = [
            {
                "turn_id": 2,
                "type": "tool_call",
                "tool_name": "unknown_tool",
                "parameters": {"q": "ABC123"},
            }
        ]
        reference = build_tool_argument_reference(AGENT_TOOLS, trace)
        assert "unknown_tool(q)" in reference
        assert "no tool schema available" in reference
        # The judge must not read "undeclared" into a schema we simply do not have.
        assert "no schema entry" not in reference

    def test_long_conversations_are_truncated(self):
        trace = [
            {
                "turn_id": turn,
                "type": "tool_call",
                "tool_name": "rebook_flight",
                "parameters": {"reservation_id": "ABC123"},
            }
            for turn in range(1, 8)
        ]
        reference = build_tool_argument_reference(AGENT_TOOLS, trace, max_calls=3)
        assert "### Turn 3 —" in reference
        assert "### Turn 4 —" not in reference
        assert "Only the first 3 tool calls are indexed" in reference


class TestFaithfulnessWiring:
    def setup_method(self):
        self.metric = make_judge_metric(FaithfulnessJudgeMetric, mock_llm=True)

    def make_context(self, **overrides):
        defaults = {
            "agent_instructions": "Be helpful",
            "agent_role": "Airline agent",
            "agent_tools": AGENT_TOOLS,
            "current_date_time": "2026-06-23 10:45 EST",
            "pipeline_type": "cascade",
            "conversation_trace": TRACE,
        }
        defaults.update(overrides)
        return make_metric_context(**defaults)

    def test_prompt_variables_expose_derived_reference(self):
        variables = self.metric.get_prompt_variables(self.make_context(), "transcript")
        assert "derived_tool_reference" in variables
        assert 'rebooking_type = "same_day"' in variables["derived_tool_reference"]

    def test_rendered_judge_prompt_contains_the_reference(self):
        variables = self.metric.get_prompt_variables(self.make_context(), "transcript")
        prompt = self.metric.get_judge_prompt(**variables)
        # Rendered through the real configs/prompts/judge.yaml template.
        assert "## Derived Reference for Tool-Call Arguments" in prompt
        assert "### Turn 2 — rebook_flight(" in prompt

    def test_prompt_without_tool_calls_has_no_dangling_section(self):
        ctx = self.make_context(conversation_trace=[{"turn_id": 1, "role": "user", "content": "hi"}])
        variables = self.metric.get_prompt_variables(ctx, "transcript")
        assert variables["derived_tool_reference"] == ""
        prompt = self.metric.get_judge_prompt(**variables)
        assert "## Derived Reference for Tool-Call Arguments" not in prompt

    def test_compute_sends_the_reference_to_the_judge(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 3, "dimensions": {"hallucination": {"rating": 3, "flagged": False}}}),
            None,
        )
        score = asyncio.run(self.metric.compute(self.make_context()))
        assert score.score == 3.0
        sent_prompt = self.metric.llm_client.generate_text.call_args[0][0][0]["content"]
        assert "## Derived Reference for Tool-Call Arguments" in sent_prompt
        assert 'reservation_id = "ABC123"' in sent_prompt
