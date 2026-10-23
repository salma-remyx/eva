"""Tests for derived tool-argument references and their injection into the faithfulness judge."""

import copy

from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.derived_references import build_tool_argument_references
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

AGENT_TOOLS = [
    {
        "id": "get_reservation",
        "name": "get_reservation",
        "description": "Retrieve a reservation.",
        "tool_type": "read",
        "required_parameters": [
            {"name": "confirmation_number", "type": "string", "description": "6-character booking code"},
            {"name": "last_name", "type": "string", "description": "Passenger last name"},
        ],
    },
    {
        "id": "apply_change_fee",
        "name": "apply_change_fee",
        "description": "Charge a change fee.",
        "tool_type": "write",
        "required_parameters": [
            {"name": "confirmation_number", "type": "string", "description": "6-character booking code"},
            {"name": "amount", "type": "number", "description": "Amount to charge"},
            {"name": "reason", "type": "string", "enum": ["voluntary", "irrops_cancellation", "goodwill"]},
            {"name": "notify", "type": "string", "default": "email", "description": "Notification channel"},
        ],
    },
]

CONVERSATION_TRACE = [
    {"role": "user", "content": "Hi, my confirmation number is ZK3FFW and my last name is Flores.", "turn_id": 1},
    {
        "tool_name": "get_reservation",
        "parameters": {"confirmation_number": "ZK3FFW", "last_name": "Flores"},
        "type": "tool_call",
        "turn_id": 1,
    },
    {
        "tool_name": "get_reservation",
        "tool_response": {
            "status": "success",
            "reservation": {"confirmation_number": "ZK3FFW", "fare_paid": 260.0, "change_fee": 75.0},
        },
        "type": "tool_response",
        "turn_id": 1,
    },
    {"role": "assistant", "content": "I found your reservation.", "turn_id": 1},
    {
        "tool_name": "apply_change_fee",
        "parameters": {"confirmation_number": "ZK3FFW", "amount": 335.0, "reason": "voluntary", "notify": "email"},
        "type": "tool_call",
        "turn_id": 2,
    },
]


class TestDerivedReferences:
    def test_schema_default_is_derived_as_a_reference(self):
        block = build_tool_argument_references(AGENT_TOOLS, CONVERSATION_TRACE)
        assert "schema default `email`" in block
        assert "grounded even though the user never stated it" in block

    def test_enum_is_derived_as_a_categorical_reference(self):
        block = build_tool_argument_references(AGENT_TOOLS, CONVERSATION_TRACE)
        assert "categorical — allowed values voluntary/irrops_cancellation/goodwill" in block

    def test_prior_tool_result_field_is_derived(self):
        block = build_tool_argument_references(AGENT_TOOLS, CONVERSATION_TRACE)
        assert "prior tool result get_reservation [turn 1] → reservation.confirmation_number = `ZK3FFW`" in block

    def test_user_stated_literals_are_derived(self):
        block = build_tool_argument_references(AGENT_TOOLS, CONVERSATION_TRACE)
        assert "user-stated `ZK3FFW`" in block  # confirmation_number → code
        assert "user-stated `Flores`" in block  # last_name → name

    def test_arithmetic_over_prior_results_is_derived_for_numeric_parameters(self):
        block = build_tool_argument_references(AGENT_TOOLS, CONVERSATION_TRACE)
        assert "arithmetic over prior results: 260 + 75 = 335" in block

    def test_dates_are_not_split_into_numeric_literals(self):
        tools = [
            {
                "name": "get_status",
                "required_parameters": [{"name": "flight_date", "type": "string", "description": "YYYY-MM-DD"}],
            }
        ]
        trace = [
            {"role": "user", "content": "The flight on 2026-03-20 was cancelled.", "turn_id": 1},
            {"tool_name": "get_status", "parameters": {"flight_date": "2026-03-20"}, "type": "tool_call", "turn_id": 1},
        ]
        block = build_tool_argument_references(tools, trace)
        assert "user-stated `2026-03-20`" in block
        assert "user-stated `2026`" not in block
        assert "user-stated `20`" not in block

    def test_numeric_sounding_boolean_parameter_gets_no_numeric_literals(self):
        tools = [
            {
                "name": "apply_waiver",
                "required_parameters": [{"name": "waive_change_fee", "type": "boolean", "description": "waive?"}],
            }
        ]
        trace = [
            {"role": "user", "content": "The fee is 75 dollars for 2 bags.", "turn_id": 1},
            {"tool_name": "apply_waiver", "parameters": {"waive_change_fee": True}, "type": "tool_call", "turn_id": 1},
        ]
        block = build_tool_argument_references(tools, trace)
        assert "user-stated `75`" not in block
        assert "no reference derived" in block

    def test_derivation_is_candidate_blind(self):
        """Tampering with the assistant's argument values must not change the reference."""
        tampered = copy.deepcopy(CONVERSATION_TRACE)
        tampered[1]["parameters"]["confirmation_number"] = "H4CKED"
        tampered[4]["parameters"]["amount"] = 99999.0
        assert build_tool_argument_references(AGENT_TOOLS, tampered) == build_tool_argument_references(
            AGENT_TOOLS, CONVERSATION_TRACE
        )

    def test_parameter_with_no_reference_says_so_without_alleging_fabrication(self):
        tools = [
            {
                "name": "file_note",
                "required_parameters": [{"name": "note_text", "type": "string", "description": "free text"}],
            }
        ]
        trace = [{"tool_name": "file_note", "parameters": {"note_text": "anything"}, "type": "tool_call", "turn_id": 0}]
        block = build_tool_argument_references(tools, trace)
        assert "`note_text`" in block
        assert "**not** evidence of fabrication" in block

    def test_undeclared_tool_is_called_out(self):
        trace = [
            {"tool_name": "delete_everything", "parameters": {"confirm": "yes"}, "type": "tool_call", "turn_id": 3},
        ]
        block = build_tool_argument_references(AGENT_TOOLS, trace)
        assert "not in the declared toolset" in block
        assert "delete_everything" in block

    def test_no_tool_calls_yields_a_short_note(self):
        trace = [{"role": "user", "content": "hello", "turn_id": 0}]
        block = build_tool_argument_references(AGENT_TOOLS, trace)
        assert block == "No tool calls with arguments in this conversation."

    def test_empty_inputs_are_tolerated(self):
        block = build_tool_argument_references(None, None)
        assert block == "No tool calls with arguments in this conversation."


class TestFaithfulnessJudgeWiring:
    """The derived references must reach the rendered faithfulness judge prompt."""

    def setup_method(self):
        self.metric = make_judge_metric(FaithfulnessJudgeMetric, mock_llm=True)

    def test_get_prompt_variables_exposes_derived_references(self):
        ctx = make_metric_context(agent_tools=AGENT_TOOLS, conversation_trace=CONVERSATION_TRACE)
        variables = self.metric.get_prompt_variables(ctx, "transcript")
        assert "ZK3FFW" in variables["derived_tool_references"]
        assert "user-stated" in variables["derived_tool_references"]

    def test_rendered_judge_prompt_carries_the_reference_block(self):
        ctx = make_metric_context(agent_tools=AGENT_TOOLS, conversation_trace=CONVERSATION_TRACE)
        prompt = self.metric.get_judge_prompt(**self.metric.get_prompt_variables(ctx, "transcript"))
        assert "## Derived References for Tool-Call Arguments" in prompt
        assert "prior tool result get_reservation [turn 1] → reservation.confirmation_number = `ZK3FFW`" in prompt

    def test_conversation_without_tools_still_renders(self):
        ctx = make_metric_context(
            agent_tools=[],
            conversation_trace=[{"role": "user", "content": "hi", "turn_id": 0}],
        )
        prompt = self.metric.get_judge_prompt(**self.metric.get_prompt_variables(ctx, "transcript"))
        assert "No tool calls with arguments in this conversation." in prompt
