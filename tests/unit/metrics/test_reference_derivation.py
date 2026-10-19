"""Tests for the derived-reference (DeRef) stage and its faithfulness wiring.

Exercises the integration end to end: FaithfulnessJudgeMetric.compute() with
reference derivation enabled must (1) issue one derivation call per tool-call
argument on DeRef's per-argument prompt — the question plus the earlier calls
and their returns, never the proposed value (candidate-blind) — and (2) embed
each answer in the judge prompt as its own block under DeRef's neutral header,
with no consumption instructions. With derivation disabled the metric must
behave exactly as before — a single judge call.
"""

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.reference_derivation import (
    DEREF_SYSTEM,
    FALLBACK_NOTE,
    HISTORY_HEADER,
    NO_TOOL_CALLS_NOTE,
    answer_text,
    argument_slots,
    derive_tool_references,
    function_line,
    has_tool_calls,
    locator_context,
    parameter_line,
)
from eva.utils.prompt_manager import get_prompt_manager
from tests.unit.metrics.conftest import make_metric_context

# Candidate values ("9C", "SW999") appear ONLY in the tool_call entries — nowhere
# in the questions, earlier calls, or returns — so they detect candidate leakage.
TRACE: list[dict[str, Any]] = [
    {
        "turn_id": 1,
        "role": "user",
        "content": "My confirmation number is B five O D Z X, and the last name is Ramirez.",
    },
    {
        "turn_id": 2,
        "type": "tool_call",
        "tool_name": "get_reservation",
        "parameters": {
            "confirmation_number": "B5ODZX",
            "last_name": "Ramirez",
            "seat_preference": "9C",
        },
    },
    {
        "turn_id": 2,
        "type": "tool_response",
        "tool_name": "get_reservation",
        "tool_response": {
            "status": "success",
            "reservation": {"confirmation_number": "B5ODZX", "flight_number": "SW123"},
        },
    },
    {"turn_id": 3, "role": "assistant", "content": "I found your reservation."},
    {"turn_id": 4, "role": "user", "content": "What seats are open on that flight?"},
    {
        "turn_id": 5,
        "type": "tool_call",
        "tool_name": "get_seat_map",
        "parameters": {"flight_number": "SW999"},
    },
]

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_reservation",
        "description": "Retrieve flight reservation using confirmation number and passenger last name.",
        "required_parameters": [
            {"name": "confirmation_number", "type": "string", "description": "6-character alphanumeric booking code"},
            {"name": "last_name", "type": "string", "description": "Passenger last name for verification"},
        ],
        "optional_parameters": [
            {
                "name": "seat_preference",
                "type": "string",
                "enum": ["window", "aisle"],
                "description": "Seat preference to apply",
            },
        ],
    },
    {
        "name": "get_seat_map",
        "description": "Get the seat map for a flight.",
        "required_parameters": [{"name": "flight_number", "type": "string", "description": "Flight number"}],
    },
]

# Free-form derivation answers (DeRef F2: value + source, or "the context does
# not determine it", in at most 3 sentences).
ANSWERS = {
    "confirmation_number": (
        'The value should be "B5ODZX". The user said "B five O D Z X" — spoken letters for that code.'
    ),
    "last_name": 'The value should be "Ramirez", from the user statement "the last name is Ramirez".',
    "seat_preference": (
        "The context does not determine the value of seat_preference. The user stated no seat preference."
    ),
    "flight_number": 'The value should be "SW123", the flight number returned in the get_reservation response.',
}

JUDGE_RESPONSE = json.dumps({"rating": 3, "dimensions": {}})

LOGGER = logging.getLogger("test_reference_derivation")


def _derivation_answer(prompt: str) -> str:
    """The mocked derivation answer for a prompt, keyed on F2's parameter name."""
    for name, answer in ANSWERS.items():
        if f"value of parameter {name} be" in prompt:
            return answer
    raise AssertionError(f"unexpected derivation prompt:\n{prompt}")


def _route(messages: list[dict]) -> tuple[str, dict | None]:
    """Route one LLM call: the judge prompt gets the judge response, everything
    else is a derivation prompt and gets its F2 answer."""
    prompt = messages[-1]["content"]
    if "expert evaluator" in prompt:
        return JUDGE_RESPONSE, None
    return _derivation_answer(prompt), {"prompt_tokens": 10, "completion_tokens": 5}


def _make_metric(config: dict[str, Any] | None = None) -> FaithfulnessJudgeMetric:
    """Instantiate the faithfulness judge with a mocked LLM client (conftest pattern)."""
    metric = FaithfulnessJudgeMetric(config=config)
    metric.llm_client = MagicMock()
    metric.llm_client.model = "test-judge"
    metric.llm_client.params = {}
    metric.llm_client.generate_text = AsyncMock(side_effect=_route)
    return metric


def _make_trace_context() -> Any:
    return make_metric_context(
        agent_instructions="You are an airline service agent.",
        agent_tools=AGENT_TOOLS,
        conversation_trace=TRACE,
    )


def _sent_prompts(mock_generate_text: AsyncMock) -> list[str]:
    """User-prompt content of every LLM call, in call order."""
    return [c.args[0][-1]["content"] for c in mock_generate_text.call_args_list]


class TestTemplateFidelity:
    def test_template_carries_derefs_texts_verbatim(self):
        template = get_prompt_manager().get_template("judge.faithfulness.reference_derivation_prompt")
        # DeRef's history header and F2 instruction, byte-exact (parameter name
        # substituted for the reference's {name}).
        assert HISTORY_HEADER + "\n{history}" in template
        assert (
            "Before seeing any proposed value: what should the value of parameter {parameter_name} be in "
            "this call, and where in the context does it come from? Quote the source text if there is one, "
            "or say that the context does not determine it. Answer in at most 3 sentences." in template
        )


class TestLocatorContext:
    def test_has_tool_calls(self):
        assert has_tool_calls(TRACE) is True
        assert has_tool_calls([{"role": "user", "content": "hi"}]) is False

    def test_first_call_history_is_none(self):
        question, history = locator_context(TRACE, 1)
        assert question == TRACE[0]["content"]
        assert history == "none"

    def test_earlier_calls_kept_assistant_prose_dropped(self):
        question, history = locator_context(TRACE, 5)
        # Question folds in every user utterance before the call.
        assert TRACE[0]["content"] in question
        assert TRACE[4]["content"] in question
        # Earlier calls stay in the history, with their arguments and returns.
        call_line = (
            '[call, step 1] get_reservation({"confirmation_number": "B5ODZX", '
            '"last_name": "Ramirez", "seat_preference": "9C"})'
        )
        assert call_line in history
        assert "[return, step 1]" in history and "SW123" in history
        # Assistant prose is not part of the reference's history.
        assert "I found your reservation." not in question + history


class TestArgumentSlots:
    def test_one_slot_per_tool_call_argument(self):
        assert argument_slots(TRACE) == [
            (1, 1, "get_reservation", "confirmation_number"),
            (1, 1, "get_reservation", "last_name"),
            (1, 1, "get_reservation", "seat_preference"),
            (2, 5, "get_seat_map", "flight_number"),
        ]

    def test_skips_calls_without_parameter_dict(self):
        trace = [{"type": "tool_call", "tool_name": "t", "parameters": "not-a-dict"}]
        assert argument_slots(trace) == []


class TestAnswerText:
    def test_strips_thinking_block(self):
        output = "<think>reasoning</think>\nThe value should be \"B5ODZX\"."
        assert answer_text(output) == 'The value should be "B5ODZX".'

    def test_uses_last_think_tag(self):
        output = "<think>a</think>first<think>b</think>\nsecond"
        assert answer_text(output) == "second"

    def test_plain_output_is_the_answer(self):
        assert answer_text("  The context does not determine it. ") == "The context does not determine it."

    def test_empty_answer_is_none(self):
        assert answer_text("<think>only reasoning</think>\n") is None
        assert answer_text("   ") is None


class TestSchemaLines:
    def test_function_line(self):
        assert function_line("get_reservation", AGENT_TOOLS[0]) == (
            "Function: get_reservation — Retrieve flight reservation using confirmation number and passenger last name."
        )
        assert function_line("unknown_tool", None) == "Function: unknown_tool"

    def test_parameter_line_with_schema_addition(self):
        assert parameter_line(AGENT_TOOLS[0], "seat_preference") == (
            'Parameter: seat_preference — Seat preference to apply; type: string; enum: ["window", "aisle"]'
        )

    def test_parameter_line_without_spec(self):
        assert (
            parameter_line(AGENT_TOOLS[0], "unknown_param")
            == "Parameter: unknown_param — (no description available)"
        )


class TestDeriveToolReferences:
    def test_one_candidate_blind_prompt_per_argument(self):
        metric = _make_metric()
        ctx = _make_trace_context()

        result = asyncio.run(derive_tool_references(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        prompts = _sent_prompts(metric.llm_client.generate_text)
        assert len(prompts) == 4  # one per tool-call argument
        assert len(result.calls) == 4
        assert result.calls[0].usage == {"prompt_tokens": 10, "completion_tokens": 5}
        # Candidate value of each parameter (appears only in its own tool_call entry).
        candidates = {"seat_preference": "9C", "flight_number": "SW999"}
        all_parameters = list(ANSWERS)
        for prompt in prompts:
            # DeRef's per-argument layout: question, history, Function/Parameter lines, F2.
            assert prompt.startswith("Question: My confirmation number")
            assert "History (all earlier calls and their returns, oldest first):" in prompt
            assert "Function: " in prompt and "Parameter: " in prompt
            assert "Before seeing any proposed value: what should the value of parameter" in prompt
            assert "Answer in at most 3 sentences." in prompt
            # Candidate-blind: an argument's own proposed value never appears in its
            # prompt. Earlier calls' arguments do appear (kept in the history, as in
            # the reference) — including their values, e.g. "9C" in the flight_number
            # prompt's history line.
            name = next(n for n in all_parameters if f"value of parameter {n} be" in prompt)
            if name in candidates:
                assert candidates[name] not in prompt
            assert "SW999" not in prompt
            # The reference derives from the call context, not the agent instructions
            # or a whole-schema dump.
            assert "airline service agent" not in prompt

    def test_messages_carry_deref_system_role(self):
        metric = _make_metric()
        ctx = _make_trace_context()

        asyncio.run(derive_tool_references(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        messages = metric.llm_client.generate_text.call_args_list[0].args[0]
        assert messages[0] == {"role": "system", "content": DEREF_SYSTEM}

    def test_later_arguments_see_earlier_calls(self):
        metric = _make_metric()
        ctx = _make_trace_context()

        asyncio.run(derive_tool_references(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        flight_prompt = next(p for p in _sent_prompts(metric.llm_client.generate_text) if "flight_number" in p)
        assert '[call, step 1] get_reservation(' in flight_prompt  # earlier call arguments kept
        assert "SW123" in flight_prompt  # earlier return kept

    def test_blocks_rendered_under_neutral_header(self):
        metric = _make_metric()
        ctx = _make_trace_context()

        result = asyncio.run(derive_tool_references(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        seat_block = (
            "get_reservation.seat_preference (call 1)\nExpected value (written before seeing the proposed value):"
        )
        assert seat_block in result.text
        assert ANSWERS["seat_preference"] in result.text
        assert "get_seat_map.flight_number (call 2)" in result.text
        assert result.text.count("Expected value (written before seeing the proposed value):") == 4

    def test_failed_derivation_skips_only_its_block(self):
        metric = _make_metric()

        def fail_seat_preference(messages):
            if "value of parameter seat_preference be" in messages[-1]["content"]:
                raise RuntimeError("aux model down")
            return _route(messages)

        metric.llm_client.generate_text = AsyncMock(side_effect=fail_seat_preference)
        ctx = _make_trace_context()

        result = asyncio.run(derive_tool_references(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        assert "seat_preference" not in result.text
        assert "confirmation_number" in result.text
        assert len(result.calls) == 3

    def test_empty_answer_skips_block(self):
        metric = _make_metric()

        def blank_seat_preference(messages):
            if "value of parameter seat_preference be" in messages[-1]["content"]:
                return ("<think>only reasoning</think>\n", None)
            return _route(messages)

        metric.llm_client.generate_text = AsyncMock(side_effect=blank_seat_preference)
        ctx = _make_trace_context()

        result = asyncio.run(derive_tool_references(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        assert "seat_preference" not in result.text
        assert "flight_number" in result.text

    def test_all_derivations_failed_returns_fallback_note(self):
        metric = _make_metric()
        metric.llm_client.generate_text = AsyncMock(side_effect=RuntimeError("aux model down"))
        ctx = _make_trace_context()

        result = asyncio.run(derive_tool_references(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        assert result.text == FALLBACK_NOTE
        assert result.calls == []

    def test_no_tool_call_arguments_skips_llm_call(self):
        metric = _make_metric()
        ctx = make_metric_context(
            conversation_trace=[
                {"turn_id": 1, "role": "user", "content": "hi"},
                {"turn_id": 2, "role": "assistant", "content": "hello"},
            ]
        )

        result = asyncio.run(derive_tool_references(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        assert result.text == NO_TOOL_CALLS_NOTE
        metric.llm_client.generate_text.assert_not_awaited()


class TestFaithfulnessDerivedReferenceIntegration:
    def test_compute_embeds_per_argument_blocks_without_consumption_policy(self):
        metric = _make_metric(config={"derived_reference": True})
        ctx = _make_trace_context()

        score = asyncio.run(metric.compute(ctx))

        assert score.score == 3.0
        assert score.error is None
        assert metric.llm_client.generate_text.await_count == 5  # 4 derivations + judge
        judge_prompt = score.details["judge_prompt"]
        assert "## Derived References for Tool-Call Arguments" in judge_prompt
        assert "Expected value (written before seeing the proposed value):" in judge_prompt
        assert ANSWERS["flight_number"] in judge_prompt
        # The reference adds no consumption policy: no trace-over-sheet
        # precedence, no ungrounded-implies-fabricated rule.
        assert "trust the trace" not in judge_prompt
        assert "fabricated unless" not in judge_prompt
        assert "<ungrounded>" not in judge_prompt
        assert ctx.derived_tool_reference.count("(call ") == 4

    def test_compute_default_off_is_unchanged_single_call(self):
        metric = _make_metric()
        ctx = _make_trace_context()

        score = asyncio.run(metric.compute(ctx))

        assert score.score == 3.0
        assert metric.llm_client.generate_text.await_count == 1
        assert ctx.derived_tool_reference is None
        assert "No derived reference was requested" in score.details["judge_prompt"]

    def test_compute_derivation_failure_still_scores(self):
        metric = _make_metric(config={"derived_reference": True})

        def fail_derivations(messages):
            if "expert evaluator" in messages[-1]["content"]:
                return JUDGE_RESPONSE, None
            raise RuntimeError("aux model down")

        metric.llm_client.generate_text = AsyncMock(side_effect=fail_derivations)
        ctx = _make_trace_context()

        score = asyncio.run(metric.compute(ctx))

        assert score.score == 3.0
        assert score.error is None
        assert FALLBACK_NOTE in score.details["judge_prompt"]

    def test_derivation_enabled_via_env_var(self, monkeypatch):
        monkeypatch.setenv("FAITHFULNESS_DERIVED_REFERENCE", "1")
        assert _make_metric()._reference_derivation_enabled() is True

    def test_derivation_config_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("FAITHFULNESS_DERIVED_REFERENCE", "1")
        assert _make_metric(config={"derived_reference": False})._reference_derivation_enabled() is False

    def test_deriver_model_configures_separate_client(self, monkeypatch):
        metric = _make_metric(config={"derived_reference": True, "deriver_model": "test-deriver"})

        deriver = metric._deriver_client()
        assert deriver is not metric.llm_client
        assert deriver.model == "test-deriver"
        assert deriver.params == {"max_tokens": 8192}  # DeRef's derivation generation budget

        merged = _make_metric(
            {"derived_reference": True, "deriver_model": "test-deriver", "deriver_params": {"max_tokens": 512}}
        )._deriver_client()
        assert merged.params == {"max_tokens": 512}

    def test_deriver_model_unset_reuses_judge_client(self):
        metric = _make_metric()
        assert metric._deriver_client() is metric.llm_client

        same = _make_metric(config={"deriver_model": "test-judge"})
        assert same._deriver_client() is same.llm_client

    def test_deriver_model_env_var(self, monkeypatch):
        monkeypatch.setenv("FAITHFULNESS_DERIVER_MODEL", "env-deriver")
        metric = _make_metric()
        deriver = metric._deriver_client()
        assert deriver is not metric.llm_client
        assert deriver.model == "env-deriver"

    def test_compute_routes_derivation_through_deriver_client(self, monkeypatch):
        metric = _make_metric(config={"derived_reference": True, "deriver_model": "test-deriver"})
        deriver = _make_metric().llm_client  # a fully mocked client
        deriver.model = "test-deriver"

        created = []

        class FakeLLMClient:
            def __init__(self, model, params=None):
                self.model, self.params = model, params
                self.generate_text = deriver.generate_text
                created.append(self)

        monkeypatch.setattr("eva.metrics.accuracy.faithfulness.LLMClient", FakeLLMClient)
        ctx = _make_trace_context()

        score = asyncio.run(metric.compute(ctx))

        assert score.score == 3.0
        assert created and created[0].model == "test-deriver"
        # Derivations went through the separate deriver; the judge made its own single call.
        assert deriver.generate_text.await_count == 4
        assert metric.llm_client.generate_text.await_count == 1
