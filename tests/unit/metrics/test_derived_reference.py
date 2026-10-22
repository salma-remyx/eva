"""Tests for the candidate-blind derived tool-argument reference.

Covers the derived_reference module itself and its integration into the
faithfulness judge: ``FaithfulnessJudgeMetric.get_prompt_variables`` feeds the
reference into the ``judge.faithfulness.user_prompt`` template.
"""

import json

import pytest

from eva.metrics.accuracy.derived_reference import (
    build_argument_reference_note,
    derive_argument_reference,
    extract_user_utterances,
    format_argument_reference,
)
from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

BOOKING_TOOLS = [
    {
        "id": "search_flights",
        "name": "search_flights",
        "tool_type": "read",
        "required_parameters": [
            {"name": "destination", "type": "string", "description": "Destination airport code"},
            {
                "name": "cabin",
                "type": "string",
                "enum": ["economy", "premium", "business"],
                "default": "economy",
                "description": "Cabin class; economy unless requested",
            },
        ],
        "optional_parameters": [
            {
                "name": "seat_preference",
                "type": "string",
                "enum": ["window", "aisle", "middle", "no_preference"],
                "description": "Seat preference; omit for no preference",
            },
        ],
    }
]


class TestDeriveArgumentReference:
    def test_schema_default_is_reported(self):
        references = derive_argument_reference([], BOOKING_TOOLS)

        cabin = next(ref for ref in references if ref.parameter == "cabin")
        assert cabin.required is True
        assert cabin.default == "economy"
        assert '"economy"' in format_argument_reference([cabin])

    def test_plain_required_parameter_without_signal_is_skipped(self):
        references = derive_argument_reference([], BOOKING_TOOLS)

        assert all(ref.parameter != "destination" for ref in references)

    def test_enum_attestation_from_user_speech(self):
        references = derive_argument_reference(["I'd like a window seat please"], BOOKING_TOOLS)

        seat = next(ref for ref in references if ref.parameter == "seat_preference")
        assert seat.attested_values == ("window",)
        assert "no enum value appears" not in format_argument_reference([seat])

    def test_enum_attestation_uses_word_boundaries(self):
        # "many" must not attest the enum value "any"; "no preference" must attest "no_preference".
        tools = [
            {
                "name": "pick_option",
                "required_parameters": [
                    {"name": "choice", "type": "string", "enum": ["any", "no_preference"]}
                ],
            }
        ]

        many_only = derive_argument_reference(["there are many options"], tools)
        assert many_only[0].attested_values == ()

        spoken = derive_argument_reference(["I have no preference, choose for me"], tools)
        assert spoken[0].attested_values == ("no_preference",)

    def test_optional_parameter_signals_omitting_is_grounded(self):
        note = build_argument_reference_note([], BOOKING_TOOLS)

        assert "**search_flights.seat_preference** (optional):" in note

    def test_openai_function_dialect_supported(self):
        tools = [
            {
                "name": "book",
                "parameters": {
                    "properties": {
                        "cabin": {"type": "string", "enum": ["economy", "business"]},
                        "hold": {"type": "boolean", "default": False},
                    },
                    "required": ["cabin"],
                },
            }
        ]

        references = derive_argument_reference(["put me in economy"], tools)
        by_parameter = {ref.parameter: ref for ref in references}

        assert by_parameter["cabin"].required is True
        assert by_parameter["cabin"].attested_values == ("economy",)
        assert by_parameter["hold"].required is False
        assert by_parameter["hold"].default is False

    def test_no_signal_tools_render_empty_note(self):
        assert derive_argument_reference(["hi there"], [{"name": "search"}]) == []
        assert build_argument_reference_note([{"role": "user", "content": "hi"}], [{"name": "search"}]) == ""


class TestCandidateBlindness:
    def test_assistant_speech_and_tool_calls_do_not_leak_into_reference(self):
        users_only = [{"role": "user", "content": "book me a seat in economy", "turn_id": 0}]

        with_candidate = users_only + [
            {"role": "assistant", "content": "Done, I put you in business class!", "turn_id": 1},
            {
                "type": "tool_call",
                "tool_name": "search_flights",
                "tool_params": {"destination": "LAX", "cabin": "premium", "seat_preference": "aisle"},
                "turn_id": 1,
            },
        ]

        blind = derive_argument_reference(extract_user_utterances(users_only), BOOKING_TOOLS)
        exposed = derive_argument_reference(extract_user_utterances(with_candidate), BOOKING_TOOLS)

        assert blind == exposed
        seat = next(ref for ref in blind if ref.parameter == "seat_preference")
        assert seat.attested_values == ()  # "aisle" was only ever in the tool call


class TestFaithfulnessIntegration:
    def setup_method(self):
        self.metric = make_judge_metric(FaithfulnessJudgeMetric, mock_llm=True)

    def test_prompt_variables_include_derived_reference(self):
        ctx = make_metric_context(
            agent_tools=BOOKING_TOOLS,
            conversation_trace=[{"role": "user", "content": "I'd like a window seat", "turn_id": 0}],
        )

        variables = self.metric.get_prompt_variables(ctx, "User: hi")

        note = variables["derived_tool_reference"]
        assert "Derived Tool-Argument Reference (candidate-blind)" in note
        assert "user-attested enum value" in note
        assert '"window"' in note

    def test_derived_reference_flows_into_rendered_judge_prompt(self):
        ctx = make_metric_context(
            agent_tools=BOOKING_TOOLS,
            conversation_trace=[{"role": "user", "content": "I'd like a window seat", "turn_id": 0}],
        )

        prompt = self.metric.get_judge_prompt(**self.metric.get_prompt_variables(ctx, "User: hi"))

        assert "Derived Tool-Argument Reference (candidate-blind)" in prompt
        assert "**search_flights.seat_preference** (optional)" in prompt

    @pytest.mark.asyncio
    async def test_compute_end_to_end_carries_reference_in_judge_prompt(self):
        self.metric.llm_client.generate_text.return_value = (
            json.dumps({"rating": 3, "dimensions": {"hallucination": "none"}}),
            None,
        )
        ctx = make_metric_context(
            agent_tools=BOOKING_TOOLS,
            conversation_trace=[
                {"role": "user", "content": "I'd like a window seat", "turn_id": 0},
                {"role": "assistant", "content": "Reserved a window seat for you.", "turn_id": 1},
            ],
        )

        score = await self.metric.compute(ctx)

        assert score.error is None
        assert score.score == 3.0
        assert "Derived Tool-Argument Reference" in score.details["judge_prompt"]
