"""Tests for the candidate-blind derived-reference pass (DeRef adaptation).

Covers the derived_tool_references module itself (redaction, parsing,
formatting, degradation) and its wiring into FaithfulnessJudgeMetric: the
judge prompt must carry the derived reference sheet whenever the conversation
contains tool calls.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import eva.metrics.accuracy.derived_tool_references as dtr
from eva.metrics.accuracy.faithfulness import FaithfulnessJudgeMetric
from eva.metrics.base import MetricContext
from tests.unit.metrics.conftest import make_judge_metric, make_metric_context

AGENT_TOOLS = [
    {
        "name": "search_flights",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "enum": ["ORD", "JFK"]},
                "trip_type": {"type": "string", "default": "one_way"},
            },
        },
    }
]

TRACE_WITH_TOOL_CALL = [
    {"turn_id": 1, "role": "user", "content": "I want to fly out of Chicago O'Hare."},
    {
        "turn_id": 1,
        "type": "tool_call",
        "tool_name": "search_flights",
        "parameters": {"destination": "ORD", "trip_type": "one_way"},
    },
    {
        "turn_id": 1,
        "type": "tool_response",
        "tool_name": "search_flights",
        "tool_response": '{"flight": "UA420"}',
    },
    {"turn_id": 1, "role": "assistant", "content": "Found a flight."},
]

HELPER_RESPONSE = json.dumps(
    {
        "tools": [
            {
                "tool": "search_flights",
                "parameters": [
                    {
                        "name": "destination",
                        "derived_value": "ORD",
                        "source": "user_statement",
                        "evidence": 'Turn 1: "Chicago O\'Hare" maps to ORD per schema enum',
                    },
                    {
                        "name": "trip_type",
                        "derived_value": "one_way",
                        "source": "schema_default",
                        "evidence": "schema default; user did not state a trip type",
                    },
                ],
            }
        ]
    }
)

JUDGE_RESPONSE = json.dumps(
    {
        "rating": 3,
        "dimensions": {
            "fabricating_tool_parameters": {"rating": 3, "flagged": False, "evidence": "None"},
            "misrepresenting_tool_result": {"rating": 3, "flagged": False, "evidence": "None"},
            "violating_policies": {"rating": 3, "flagged": False, "evidence": "None"},
            "failing_to_disambiguate": {"rating": 3, "flagged": False, "evidence": "None"},
            "hallucination": {"rating": 3, "flagged": False, "evidence": "None"},
        },
    }
)


def patch_helper_llm(monkeypatch, return_value: str = HELPER_RESPONSE, side_effect=None) -> MagicMock:
    """Patch the helper LLMClient in the module namespace and return its mock."""
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.generate_text = AsyncMock(side_effect=side_effect)
    else:
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        mock_client.generate_text = AsyncMock(return_value=(return_value, usage))
    mock_factory = MagicMock(return_value=mock_client)
    monkeypatch.setattr(dtr, "LLMClient", mock_factory)
    return mock_client


def make_tool_context(**overrides) -> MetricContext:
    """MetricContext with a tool-calling conversation."""
    defaults = {
        "agent_tools": AGENT_TOOLS,
        "conversation_trace": TRACE_WITH_TOOL_CALL,
    }
    defaults.update(overrides)
    return make_metric_context(**defaults)


class TestRedaction:
    def test_redact_hides_values_keeps_parameter_names(self):
        redacted = dtr.redact_tool_call_arguments(TRACE_WITH_TOOL_CALL)

        tool_call = next(e for e in redacted if e.get("type") == "tool_call")
        assert tool_call["parameters"] == {"destination": dtr.REDACTED_VALUE, "trip_type": dtr.REDACTED_VALUE}

        # Other entries pass through untouched, and the original trace is not mutated.
        response = next(e for e in redacted if e.get("type") == "tool_response")
        assert "UA420" in response["tool_response"]
        assert TRACE_WITH_TOOL_CALL[1]["parameters"]["destination"] == "ORD"

    def test_redact_preserves_nested_parameter_names(self):
        trace = [
            {
                "type": "tool_call",
                "tool_name": "update_record",
                "parameters": {"record": {"name": "Alice", "flags": ["a", "b"]}},
            }
        ]
        redacted = dtr.redact_tool_call_arguments(trace)
        assert redacted[0]["parameters"] == {
            "record": {"name": dtr.REDACTED_VALUE, "flags": [dtr.REDACTED_VALUE, dtr.REDACTED_VALUE]}
        }

    def test_trace_has_tool_calls(self):
        assert dtr.trace_has_tool_calls(TRACE_WITH_TOOL_CALL)
        assert not dtr.trace_has_tool_calls([{"turn_id": 1, "role": "user", "content": "hi"}])


class TestHelperPass:
    @pytest.mark.asyncio
    async def test_generate_formats_derived_references(self, monkeypatch):
        mock_client = patch_helper_llm(monkeypatch)
        ctx = make_tool_context()

        result = await dtr.generate_derived_tool_references(ctx, model="test-helper")

        assert 'destination = "ORD" — user_statement' in result
        assert 'trip_type = "one_way" — schema_default' in result
        assert "- search_flights:" in result
        mock_client.generate_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_helper_prompt_is_candidate_blind(self, monkeypatch):
        mock_client = patch_helper_llm(monkeypatch)
        # A value that appears ONLY in the tool-call arguments — the candidate.
        trace = [
            {"turn_id": 1, "role": "user", "content": "Book a flight for me."},
            {
                "turn_id": 1,
                "type": "tool_call",
                "tool_name": "search_flights",
                "parameters": {"destination": "ORD", "trip_type": "round_trip"},
            },
        ]
        ctx = make_tool_context(conversation_trace=trace)

        await dtr.generate_derived_tool_references(ctx, model="test-helper")

        prompt = mock_client.generate_text.await_args[0][0][0]["content"]
        assert dtr.REDACTED_VALUE in prompt
        assert "destination" in prompt and "trip_type" in prompt  # parameter names kept
        assert "round_trip" not in prompt  # candidate value withheld from the helper

    @pytest.mark.asyncio
    async def test_generate_skips_llm_when_no_tool_calls(self, monkeypatch):
        mock_client = patch_helper_llm(monkeypatch)
        ctx = make_tool_context(conversation_trace=[{"turn_id": 1, "role": "user", "content": "hi"}])

        result = await dtr.generate_derived_tool_references(ctx, model="test-helper")

        assert result == dtr.REFERENCES_UNAVAILABLE_NOTE
        mock_client.generate_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generate_degrades_on_unparseable_output(self, monkeypatch):
        patch_helper_llm(monkeypatch, return_value="not json ~~~")
        ctx = make_tool_context()

        result = await dtr.generate_derived_tool_references(ctx, model="test-helper")

        assert result == dtr.REFERENCES_UNAVAILABLE_NOTE

    @pytest.mark.asyncio
    async def test_generate_degrades_on_llm_failure(self, monkeypatch):
        patch_helper_llm(monkeypatch, side_effect=RuntimeError("helper down"))
        ctx = make_tool_context()

        result = await dtr.generate_derived_tool_references(ctx, model="test-helper")

        assert result == dtr.REFERENCES_UNAVAILABLE_NOTE

    def test_parse_helper_response_rejects_malformed_rows(self):
        raw = json.dumps({"tools": [{"tool": "", "parameters": []}, {"tool": "t", "parameters": "nope"}]})
        assert dtr.parse_helper_response(raw) is None
        assert dtr.parse_helper_response(None) is None


class TestFaithfulnessIntegration:
    """The wiring: FaithfulnessJudgeMetric.compute must put the sheet into the judge prompt."""

    def setup_method(self):
        self.metric = make_judge_metric(FaithfulnessJudgeMetric, mock_llm=True)
        self.metric.llm_client.generate_text.return_value = (JUDGE_RESPONSE, None)

    def judge_prompt(self) -> str:
        """Judge prompt captured from the (mocked) judge call."""
        messages = self.metric.llm_client.generate_text.await_args[0][0]
        return messages[0]["content"]

    @pytest.mark.asyncio
    async def test_compute_wires_derived_references_into_judge_prompt(self, monkeypatch):
        patch_helper_llm(monkeypatch)
        ctx = make_tool_context()

        score = await self.metric.compute(ctx)

        assert score.error is None
        assert score.score == 3.0
        prompt = self.judge_prompt()
        assert "Derived references for tool-call arguments" in prompt
        assert 'destination = "ORD" — user_statement' in prompt
        assert 'trip_type = "one_way" — schema_default' in prompt

    @pytest.mark.asyncio
    async def test_compute_without_tool_calls_skips_helper(self, monkeypatch):
        mock_client = patch_helper_llm(monkeypatch)
        ctx = make_tool_context(conversation_trace=[{"turn_id": 1, "role": "user", "content": "hi"}])

        score = await self.metric.compute(ctx)

        assert score.score == 3.0
        mock_client.generate_text.assert_not_awaited()
        assert "No derived reference sheet is available" in self.judge_prompt()

    @pytest.mark.asyncio
    async def test_compute_helper_failure_still_judges(self, monkeypatch):
        patch_helper_llm(monkeypatch, side_effect=RuntimeError("helper down"))
        ctx = make_tool_context()

        score = await self.metric.compute(ctx)

        assert score.error is None
        assert score.score == 3.0
        assert "No derived reference sheet is available" in self.judge_prompt()

    def test_get_prompt_variables_reports_unavailable_outside_compute(self):
        # get_prompt_variables is also called outside compute (e.g. prompt previews):
        # without a contextvar value it must still render the fallback note.
        ctx = make_tool_context()
        variables = self.metric.get_prompt_variables(ctx, "transcript")
        assert variables["derived_tool_references"] == dtr.REFERENCES_UNAVAILABLE_NOTE
