"""Tests for the derived-reference (DeRef) stage and its faithfulness wiring.

Exercises the integration end to end: FaithfulnessJudgeMetric.compute() with
reference derivation enabled must (1) send the aux model a candidate-blind
view of the conversation (grounding sources only), and (2) embed the derived
reference sheet in the judge prompt. With derivation disabled the metric must
behave exactly as before — a single judge call.
"""

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.reference_derivation import (
    FALLBACK_NOTE,
    NO_TOOL_CALLS_NOTE,
    derive_tool_reference,
    format_grounding_transcript,
    has_tool_calls,
)
from tests.unit.metrics.conftest import make_metric_context

# "window" (the candidate tool-call parameter below) appears ONLY in the tool_call
# entry — nowhere in the grounding sources — so it detects candidate leakage.
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
            "seat_preference": "window",
        },
    },
    {
        "turn_id": 2,
        "type": "tool_response",
        "tool_name": "get_reservation",
        "tool_response": {
            "status": "success",
            "reservation": {"confirmation_number": "B5ODZX", "last_name": "Ramirez"},
        },
    },
    {"turn_id": 3, "role": "assistant", "content": "I found your reservation."},
]

SHEET = (
    "get_reservation.confirmation_number = B5ODZX — source: user turn\n"
    "get_reservation.last_name = Ramirez — source: user turn\n"
    "get_reservation.seat_preference = <ungrounded>"
)

JUDGE_RESPONSE = json.dumps({"rating": 3, "dimensions": {}})

LOGGER = logging.getLogger("test_reference_derivation")


def _make_metric(config: dict[str, Any] | None = None) -> FaithfulnessJudgeMetric:
    """Instantiate the faithfulness judge with a mocked LLM client (conftest pattern)."""
    metric = FaithfulnessJudgeMetric(config=config)
    metric.llm_client = MagicMock()
    metric.llm_client.generate_text = AsyncMock()
    metric.llm_client.model = "test-judge"
    metric.llm_client.params = {}
    return metric


def _make_trace_context() -> Any:
    return make_metric_context(
        agent_instructions="You are an airline service agent.",
        agent_tools=[
            {"name": "get_reservation", "parameters": {"confirmation_number": "string", "last_name": "string"}}
        ],
        conversation_trace=TRACE,
    )


def _sent_prompt(mock_generate_text: AsyncMock, call: int) -> str:
    """Return the user prompt content of the nth LLM call."""
    messages = mock_generate_text.call_args_list[call].args[0]
    return messages[0]["content"]


class TestGroundingTranscript:
    def test_has_tool_calls(self):
        assert has_tool_calls(TRACE) is True
        assert has_tool_calls([{"role": "user", "content": "hi"}]) is False

    def test_format_keeps_user_and_tool_responses_only(self):
        text = format_grounding_transcript(TRACE)
        assert "Ramirez" in text  # user statement
        assert "B5ODZX" in text  # returned by the tool response (a grounding source)
        assert "window" not in text  # candidate tool-call parameter must not leak
        assert "I found your reservation." not in text  # assistant claims are not sources


class TestDeriveToolReference:
    def test_derives_sheet_from_grounding_sources_only(self):
        metric = _make_metric()
        metric.llm_client.generate_text.return_value = (SHEET, {"prompt_tokens": 10, "completion_tokens": 5})
        ctx = _make_trace_context()

        result = asyncio.run(derive_tool_reference(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        assert result.text == SHEET
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}
        aux_prompt = _sent_prompt(metric.llm_client.generate_text, 0)
        # Candidate-blind: grounding sources in, assistant tool calls and claims out.
        assert "Ramirez" in aux_prompt
        assert "B5ODZX" in aux_prompt  # from the tool response, not the tool call
        assert "window" not in aux_prompt
        assert "I found your reservation." not in aux_prompt
        assert "airline service agent" in aux_prompt
        assert "get_reservation" in aux_prompt  # tool schema is a grounding source

    def test_falls_back_never_raises(self):
        metric = _make_metric()
        metric.llm_client.generate_text.side_effect = RuntimeError("aux model down")
        ctx = _make_trace_context()

        result = asyncio.run(derive_tool_reference(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        assert result.text == FALLBACK_NOTE
        assert result.prompt == ""

    def test_falls_back_on_empty_output(self):
        metric = _make_metric()
        metric.llm_client.generate_text.return_value = ("   ", None)
        ctx = _make_trace_context()

        result = asyncio.run(derive_tool_reference(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        assert result.text == FALLBACK_NOTE

    def test_skips_llm_call_when_no_tool_calls(self):
        metric = _make_metric()
        ctx = make_metric_context(
            conversation_trace=[
                {"turn_id": 1, "role": "user", "content": "hi"},
                {"turn_id": 2, "role": "assistant", "content": "hello"},
            ]
        )

        result = asyncio.run(derive_tool_reference(metric.llm_client, metric.prompt_manager, ctx, LOGGER))

        assert result.text == NO_TOOL_CALLS_NOTE
        metric.llm_client.generate_text.assert_not_awaited()


class TestFaithfulnessDerivedReferenceIntegration:
    def test_compute_injects_derived_reference_into_judge_prompt(self):
        metric = _make_metric(config={"derived_reference": True})
        metric.llm_client.generate_text.side_effect = [(SHEET, {"prompt_tokens": 10}), (JUDGE_RESPONSE, None)]
        ctx = _make_trace_context()

        score = asyncio.run(metric.compute(ctx))

        assert score.score == 3.0
        assert score.error is None
        assert metric.llm_client.generate_text.await_count == 2  # aux derivation + judge
        # The aux call stayed candidate-blind.
        assert "window" not in _sent_prompt(metric.llm_client.generate_text, 0)
        # The judge prompt embeds the reference sheet under its dedicated section.
        judge_prompt = score.details["judge_prompt"]
        assert "## Derived Reference for Tool-Call Arguments" in judge_prompt
        assert "get_reservation.confirmation_number = B5ODZX" in judge_prompt
        assert "get_reservation.seat_preference = <ungrounded>" in judge_prompt
        assert ctx.derived_tool_reference == SHEET

    def test_compute_default_off_is_unchanged_single_call(self):
        metric = _make_metric()
        metric.llm_client.generate_text.return_value = (JUDGE_RESPONSE, None)
        ctx = _make_trace_context()

        score = asyncio.run(metric.compute(ctx))

        assert score.score == 3.0
        assert metric.llm_client.generate_text.await_count == 1
        assert ctx.derived_tool_reference is None
        assert "No derived reference was requested" in score.details["judge_prompt"]

    def test_compute_derivation_failure_still_scores(self):
        metric = _make_metric(config={"derived_reference": True})
        metric.llm_client.generate_text.side_effect = [RuntimeError("aux model down"), (JUDGE_RESPONSE, None)]
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
