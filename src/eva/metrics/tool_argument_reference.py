"""Derived reference for tool-call arguments, supplied to LLM judges.

Adapted from "Judges Follow the Reference: The Derivation Gap in Verifying
Tool-Call Arguments" (arXiv:2610.18497). That paper shows judges accept a
correct argument value less often when their input holds no reference for it —
typically because the value is a schema default the user never mentioned, or
because it has to be inferred rather than copied. Its fix (DeRef) is to write
an explicit reference for the judge instead of relying on prompt instructions.

This module derives that reference deterministically from data the judge
already receives but cannot easily re-derive by hand: the agent's tool
descriptors and the transcript. For every tool call in the trace it states,
per argument, whether the value is allowed by the schema (enum member or
declared default) and where the value literally appears in user turns or tool
responses. The paper derives its reference with a separate model that reads
the full history; here the tool schema *is* the authority for defaults and
categorical values, so no extra model call is needed.

The output is an evidence index, never a verdict: it records where a value
comes from, and leaves the grounded-or-fabricated decision to the judge.
"""

import json
from typing import Any

# Bounds so a long conversation cannot balloon the judge prompt.
_MAX_CALLS = 25
_MAX_ARGS = 12
_MAX_SOURCES = 3
_MAX_VALUE_DISPLAY = 80

# Values shorter than this match too much incidental text to be a useful
# reference (e.g. "en", "1", booleans).
_MIN_SOURCE_VALUE_LEN = 3


def build_tool_argument_reference(
    agent_tools: list[dict],
    conversation_trace: list[dict],
    *,
    max_calls: int = _MAX_CALLS,
) -> str:
    """Render the derived-reference block for every tool call in the trace.

    Args:
        agent_tools: The agent's tool descriptors (``MetricContext.agent_tools``),
            each holding ``required_parameters`` / ``optional_parameters`` lists.
        conversation_trace: The trace entries the judge is evaluated on.
        max_calls: Maximum number of tool calls to describe before truncating.

    Returns:
        A markdown section, or an empty string when the trace holds no tool
        calls (so the prompt placeholder collapses to nothing).
    """
    specs = _parameter_specs(agent_tools)
    all_calls = _tool_calls(conversation_trace)
    if not all_calls:
        return ""

    lines = [
        "## Derived Reference for Tool-Call Arguments",
        "",
        "For each tool call below, every argument is checked against the tool schema and the "
        "transcript. \"Allowed by schema\" means the value is an enum member or the declared "
        "default for that parameter; \"appears in\" points at the user turn or tool response "
        "where the value is literally present. This is a mechanical index of where values come "
        "from, not a judgment of whether a call is grounded.",
        "",
    ]

    for turn_id, tool_name, parameters in all_calls[:max_calls]:
        declared = ", ".join(parameters) or "(no arguments)"
        lines.append(f"### Turn {turn_id} — {tool_name}({declared})")
        if tool_name not in specs:
            # Without a schema we cannot say anything about declared parameters —
            # only report where the values appear. Claiming they are undeclared
            # would itself bias the judge.
            lines.append("- no tool schema available; the notes below only report transcript matches")
        params_spec = specs.get(tool_name, {})
        for arg_name, arg_value in list(parameters.items())[:_MAX_ARGS]:
            lines.append(
                _describe_argument(
                    arg_name,
                    arg_value,
                    params_spec.get(arg_name),
                    conversation_trace,
                    schema_known=tool_name in specs,
                )
            )
        lines.append("")

    if len(all_calls) > max_calls:
        lines.append(f"(Only the first {max_calls} tool calls are indexed.)")

    return "\n".join(lines).strip()


def _parameter_specs(agent_tools: list[dict]) -> dict[str, dict[str, dict[str, Any]]]:
    """Map tool name -> parameter name -> parameter descriptor."""
    specs: dict[str, dict[str, dict[str, Any]]] = {}
    for tool in agent_tools or []:
        if not isinstance(tool, dict):
            continue
        tool_name = tool.get("name") or tool.get("id")
        if not isinstance(tool_name, str):
            continue
        params: dict[str, dict[str, Any]] = {}
        for key in ("required_parameters", "optional_parameters"):
            for param in tool.get(key) or []:
                if isinstance(param, dict) and isinstance(param.get("name"), str):
                    params[str(param["name"])] = param
        specs[tool_name] = params
    return specs


def _tool_calls(conversation_trace: list[dict]) -> list[tuple[int, str, dict]]:
    """Collect ``(turn_id, tool_name, parameters)`` for trace tool-call entries."""
    calls: list[tuple[int, str, dict]] = []
    for entry in conversation_trace or []:
        if not isinstance(entry, dict) or entry.get("type") != "tool_call":
            continue
        tool_name = entry.get("tool_name")
        parameters = entry.get("parameters")
        if not isinstance(tool_name, str) or not isinstance(parameters, dict):
            continue
        calls.append((int(entry.get("turn_id") or 0), tool_name, parameters))
    return calls


def _describe_argument(
    arg_name: str,
    arg_value: Any,
    param_spec: dict[str, Any] | None,
    conversation_trace: list[dict],
    *,
    schema_known: bool,
) -> str:
    """Describe where one argument value comes from, as a single bullet line."""
    schema_note = _schema_note(arg_value, param_spec, schema_known=schema_known)

    # A value the schema itself allows (enum member / declared default) already
    # has its reference, so skip the transcript search to keep the block short.
    schema_grounded = schema_note.startswith(("allowed by schema", "matches the schema"))
    sources = [] if schema_grounded else _value_sources(arg_value, conversation_trace)

    source_note = f"appears in {', '.join(sources)}" if sources else "no matching text in user turns or tool responses"
    return f"- {arg_name} = {_display(arg_value)}: {schema_note}; {source_note}"


def _schema_note(arg_value: Any, param_spec: dict[str, Any] | None, *, schema_known: bool) -> str:
    """State what the tool schema says about this value."""
    if not schema_known:
        return "no schema available"
    if param_spec is None:
        return "not among the parameters declared for this tool"

    needle = _needle(arg_value)
    enum = param_spec.get("enum")
    if isinstance(enum, list) and enum:
        allowed = ", ".join(str(option) for option in enum)
        if any(_needle(option) == needle for option in enum):
            return f"allowed by schema (one of: {allowed})"
        return f"NOT one of the schema's allowed values ({allowed})"

    default = param_spec.get("default")
    if default is not None and _needle(default) == needle:
        return "matches the schema-declared default"

    return "no schema constraint"


def _value_sources(arg_value: Any, conversation_trace: list[dict]) -> list[str]:
    """Find user turns and tool responses that literally contain the value."""
    needle = _needle(arg_value)
    if len(needle) < _MIN_SOURCE_VALUE_LEN:
        return []

    sources: list[str] = []
    for entry in conversation_trace or []:
        if not isinstance(entry, dict):
            continue
        turn_id = int(entry.get("turn_id") or 0)
        if entry.get("role") == "user":
            text, label = _as_text(entry.get("content")), f"user turn {turn_id}"
        elif entry.get("type") == "tool_response":
            tool_name = entry.get("tool_name") or "tool"
            text = _as_text(entry.get("tool_response"))
            label = f"{tool_name} response (turn {turn_id})"
        else:
            continue
        if needle in text.casefold():
            sources.append(label)
            if len(sources) >= _MAX_SOURCES:
                break
    return sources


def _display(arg_value: Any) -> str:
    """Render an argument value compactly for the prompt."""
    if isinstance(arg_value, str):
        text = arg_value
    else:
        text = json.dumps(arg_value, default=str)
    if len(text) > _MAX_VALUE_DISPLAY:
        text = text[:_MAX_VALUE_DISPLAY] + "…"
    return f'"{text}"' if isinstance(arg_value, str) else text


def _needle(arg_value: Any) -> str:
    """Normalise a value for schema and transcript matching."""
    if isinstance(arg_value, str):
        return arg_value.strip().casefold()
    if isinstance(arg_value, bool) or arg_value is None:
        return ""  # never used as an occurrence needle
    return json.dumps(arg_value, default=str).strip().casefold()


def _as_text(value: Any) -> str:
    """Coerce a trace field to text for substring matching."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, default=str)
