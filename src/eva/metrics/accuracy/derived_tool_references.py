"""Derived references for tool-call arguments (candidate-blind reference pass).

Adapted from "Judges Follow the Reference: The Derivation Gap in Verifying
Tool-Call Arguments" (arXiv:2610.18497). The paper shows that LLM judges
under-accept *correct* tool-call argument values when their input holds no
reference for them — the value is a schema default the user never stated, it
must be inferred, or its source text was compressed out of the judge input.
It calls this the **derivation gap**, and closes it with DeRef (Derived
Reference): a helper pass reads the conversation history *without seeing the
candidate values* and writes an explicit reference sheet that is handed to the
judge alongside the conversation.

This module is the DeRef helper for EVA's faithfulness judge: it redacts the
argument values from the conversation trace, asks a helper model to derive
what each called tool's parameters *should* be (with the source of each
derivation), and renders the result as a block injected into the
``fabricating_tool_parameters`` dimension of the faithfulness prompt.

Target-native substitutions relative to the paper:
- The paper's per-argument judge becomes the whole-conversation faithfulness
  judge's ``fabricating_tool_parameters`` dimension.
- The helper call goes through the repo's ``LLMClient`` (litellm router), not
  the paper's standalone harness.
- The paper's benchmark/AUROC evaluation harness is intentionally out of
  scope; measuring the judge delta is a downstream effort.
"""

import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from typing_extensions import TypedDict

from eva.metrics.base import MetricContext
from eva.metrics.utils import format_transcript_with_tools
from eva.utils.json_utils import extract_and_load_json
from eva.utils.llm_client import LLMClient
from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Sentinel replacing every tool-call argument value in the helper's input.
# Parameter *names* come from the tool schema and are kept; the values are the
# candidate the helper must not see (the paper's candidate-blind constraint).
REDACTED_VALUE = "<candidate value withheld>"

# Rendered into the judge prompt when no reference sheet could be produced
# (no tool calls, helper failure, or unparseable helper output).
REFERENCES_UNAVAILABLE_NOTE = (
    "(No derived reference sheet is available for this conversation. Judge tool-call argument "
    "grounding solely on the transcript, the tool schemas, and the agent instructions.)"
)

# Per-compute transport for the reference sheet. The metric's compute() runs
# inside its own asyncio task (see MetricsRunner.compute_record), so a
# contextvar set there is isolated from concurrent records — same pattern as
# _CURRENT_PROMPT_HASH in eva.metrics.versioning.
_CURRENT_DERIVED_REFERENCES: ContextVar[str | None] = ContextVar("derived_tool_references", default=None)

# Callback fired after a helper call so the caller can account for its token
# usage: (prompt, response_text, usage-or-None).
UsageCallback = Callable[[str, str | None, dict | None], None]


class DerivedReference(TypedDict):
    """One row of the reference sheet: a derived value for a tool parameter."""

    tool: str
    parameter: str
    value: Any  # JSON value; None when nothing supports a derivation
    source: str  # user_statement | prior_tool_result | schema_default | inference | not_derivable
    evidence: str


def trace_has_tool_calls(trace: list[dict]) -> bool:
    """Return True if the conversation trace contains at least one tool call."""
    return any(entry.get("type") == "tool_call" for entry in trace)


def _redact_value(value: Any) -> Any:
    """Recursively replace leaf values with the redaction sentinel, keeping all keys."""
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return REDACTED_VALUE


def redact_tool_call_arguments(trace: list[dict]) -> list[dict]:
    """Copy the trace with tool-call argument values replaced by the redaction sentinel.

    Parameter names (at every nesting depth) are kept so the helper still knows
    which parameters exist for each call; only the candidate values are hidden.

    Args:
        trace: Conversation trace entries (user/assistant turns, tool calls, tool responses).

    Returns:
        A new trace with the same structure, safe to show to the helper pass.
    """
    redacted: list[dict] = []
    for entry in trace:
        if entry.get("type") == "tool_call" and isinstance(entry.get("parameters"), dict):
            copy = dict(entry)
            copy["parameters"] = _redact_value(entry["parameters"])
            redacted.append(copy)
        else:
            redacted.append(entry)
    return redacted


_HELPER_PROMPT_TEMPLATE = """You are a reference writer for tool-call argument verification.

You will read a conversation between a user and a voice agent, plus the agent's tools. In this transcript the VALUES of tool-call arguments have been withheld (shown as "{redacted_sentinel}"): your job is to derive what those values should be, WITHOUT seeing what the agent actually sent.

For each tool call in the conversation, list every parameter of that call and derive the value a faithful agent should pass, based ONLY on:
- user_statement: a value the user explicitly stated, mapped through unambiguous standard or domain mappings when the tool schema supports it (e.g., "Chicago O'Hare" -> "ORD" if that is a schema enum value)
- prior_tool_result: a value an earlier tool response in the conversation returned
- schema_default: the parameter's default in the tool schema, when the user said nothing about it
- inference: a value unambiguously implied by the context (e.g., "tomorrow" resolved against the current date/time, or simple arithmetic over prior tool results)

If nothing supports a value, use source "not_derivable" and set derived_value to null. NEVER guess, and NEVER copy a value just because it looks plausible — every derivation needs evidence you can point to.

## Agent Instructions
{agent_instructions}

## Available Tools
{available_tools}

## Current Date/Time
{current_date_time}

## Conversation (tool-call argument values withheld)
{conversation_trace}

Respond in JSON only:
{{
    "tools": [
        {{
            "tool": "<tool name>",
            "parameters": [
                {{"name": "<parameter name>", "derived_value": <value or null>, "source": "user_statement|prior_tool_result|schema_default|inference|not_derivable", "evidence": "<short quote from the conversation or pointer into the schema>"}}
            ]
        }}
    ]
}}"""


def build_helper_prompt(context: MetricContext, redacted_trace_text: str) -> str:
    """Build the prompt for the candidate-blind reference-writing pass.

    Args:
        context: MetricContext for the conversation being judged.
        redacted_trace_text: Transcript rendering with tool-call argument values redacted.

    Returns:
        The helper prompt string.
    """
    return _HELPER_PROMPT_TEMPLATE.format(
        redacted_sentinel=REDACTED_VALUE,
        agent_instructions=context.agent_instructions,
        available_tools=json.dumps(context.agent_tools, indent=4),
        current_date_time=context.current_date_time,
        conversation_trace=redacted_trace_text,
    )


def parse_helper_response(raw_response: str | None) -> list[DerivedReference] | None:
    """Parse the helper's JSON response into reference-sheet rows.

    Args:
        raw_response: Raw helper response text.

    Returns:
        List of DerivedReference rows, or None if no usable rows were found.
    """
    if not raw_response:
        return None
    parsed = extract_and_load_json(raw_response)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("tools"), list):
        return None

    rows: list[DerivedReference] = []
    for tool_entry in parsed["tools"]:
        if not isinstance(tool_entry, dict):
            continue
        tool_name = str(tool_entry.get("tool") or "").strip()
        parameters = tool_entry.get("parameters")
        if not tool_name or not isinstance(parameters, list):
            continue
        for param in parameters:
            if not isinstance(param, dict) or not param.get("name"):
                continue
            rows.append(
                DerivedReference(
                    tool=tool_name,
                    parameter=str(param["name"]),
                    value=param.get("derived_value"),
                    source=str(param.get("source") or "not_derivable"),
                    evidence=str(param.get("evidence") or ""),
                )
            )
    return rows or None


def format_derived_references(rows: list[DerivedReference]) -> str:
    """Render reference-sheet rows as the block injected into the judge prompt.

    Args:
        rows: Parsed DerivedReference rows.

    Returns:
        Human/LLM-readable reference sheet, grouped by tool.
    """
    lines = []
    for tool_name in dict.fromkeys(row["tool"] for row in rows):
        lines.append(f"- {tool_name}:")
        for row in (r for r in rows if r["tool"] == tool_name):
            if row["source"] == "not_derivable" or row["value"] is None:
                lines.append(f"  - {row['parameter']}: not derivable from the conversation ({row['evidence']})")
            else:
                value = json.dumps(row["value"])
                lines.append(f"  - {row['parameter']} = {value} — {row['source']} ({row['evidence']})")
    return "\n".join(lines)


async def generate_derived_tool_references(
    context: MetricContext,
    *,
    model: str,
    params: dict | None = None,
    log_usage: UsageCallback | None = None,
) -> str:
    """Run the candidate-blind helper pass and return the reference sheet for the judge.

    Redacts tool-call argument values from the trace, asks the helper model to
    derive what each called tool's parameters should be, and renders the result.
    Any failure degrades to REFERENCES_UNAVAILABLE_NOTE so the judge flow never
    breaks because of the helper.

    Args:
        context: MetricContext for the conversation being judged.
        model: Helper model name (a model_name in EVA_MODEL_LIST).
        params: Optional model parameters for the helper call.
        log_usage: Optional callback invoked with (prompt, response_text, usage)
            after the call for token-usage accounting.

    Returns:
        The formatted reference sheet, or the unavailable note.
    """
    if not trace_has_tool_calls(context.conversation_trace):
        return REFERENCES_UNAVAILABLE_NOTE

    try:
        redacted_trace_text = format_transcript_with_tools(redact_tool_call_arguments(context.conversation_trace))
        prompt = build_helper_prompt(context, redacted_trace_text)

        llm_client = LLMClient(model=model, params=params or {"max_tokens": 8192})
        response_text, usage = await llm_client.generate_text([{"role": "user", "content": prompt}])
        if log_usage is not None:
            log_usage(prompt, response_text, usage)

        rows = parse_helper_response(response_text)
        if rows is None:
            logger.warning(f"Derived-reference helper returned unparseable output for {context.record_id}")
            return REFERENCES_UNAVAILABLE_NOTE
        return format_derived_references(rows)
    except Exception as e:
        logger.error(f"Derived-reference helper failed for {context.record_id}: {type(e).__name__}: {e}")
        return REFERENCES_UNAVAILABLE_NOTE
