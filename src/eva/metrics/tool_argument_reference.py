"""Derived reference for tool-call arguments, supplied to LLM judges.

Adapted from "Judges Follow the Reference: The Derivation Gap in Verifying
Tool-Call Arguments" (arXiv:2610.18497). That paper shows judges accept a
correct argument value less often when their input holds no reference for it —
typically because the value is a schema default the user never mentioned, or
because it has to be inferred rather than copied. Its fix (DeRef) is to write
an explicit reference for the judge instead of relying on prompt instructions.

The paper's deriver is a separate model prompted, per parameter and *before
seeing any proposed value*, with "what should the value of parameter {name} be
in this call, and where in the context does it come from?". This module keeps
the declared eva substitution: the reference is derived deterministically from
data the judge already receives (the agent's tool descriptors and the
transcript), with no extra model call, and it necessarily sees the proposed
values because the judge scores the whole trace. It is therefore presented as
a mechanical provenance index, never a verdict, and the
grounded-or-fabricated decision stays with the judge.

Where the deterministic derivation makes a factual claim, it uses the
reference implementation's own rules rather than raw substring matching:

- schema comparison (enum membership, declared default): exact equality, with
  case-insensitivity only between two strings, so ``True`` never "matches" a
  ``False`` default (``deref/build/paths.py::is_default``);
- transcript occurrence: exact text with word boundaries for non-string and
  short values, numeric boundaries for numbers and numeric strings, a case-
  and whitespace-insensitive pass, and the lenient longest-common-substring
  check (at least 30 characters and 60% of the value), searched over the
  reference's history — user turns, tool responses, and earlier tool calls
  only, never the described call itself or anything after it
  (``deref/build/paths.py::found``, ``deref/inputs/evidence.py::verify2``).
"""

import difflib
import json
import re
from typing import Any

# Bounds so a long conversation cannot balloon the judge prompt.
_MAX_CALLS = 25
_MAX_ARGS = 12
_MAX_SOURCES = 3
_MAX_VALUE_DISPLAY = 80

# Lenient-occurrence thresholds of the reference's quote verification
# (deref/inputs/evidence.py::verify2): a longest common substring of at least
# 30 characters that covers at least 60% of the value counts as an occurrence.
_LCS_MIN_CHARS = 30
_LCS_MIN_FRACTION = 0.6


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
        "default for that parameter; \"appears in\" points at the user turn, tool response, or "
        "earlier tool call where the value occurs. This is a mechanical index of where values "
        "come from, not a judgment of whether a call is grounded.",
        "",
    ]

    for turn_id, tool_name, parameters, call_index in all_calls[:max_calls]:
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
                    call_index=call_index,
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


def _tool_calls(conversation_trace: list[dict]) -> list[tuple[int, str, dict, int]]:
    """Collect ``(turn_id, tool_name, parameters, trace index)`` for trace tool-call entries."""
    calls: list[tuple[int, str, dict, int]] = []
    for index, entry in enumerate(conversation_trace or []):
        if not isinstance(entry, dict) or entry.get("type") != "tool_call":
            continue
        tool_name = entry.get("tool_name")
        parameters = entry.get("parameters")
        if not isinstance(tool_name, str) or not isinstance(parameters, dict):
            continue
        calls.append((int(entry.get("turn_id") or 0), tool_name, parameters, index))
    return calls


def _describe_argument(
    arg_name: str,
    arg_value: Any,
    param_spec: dict[str, Any] | None,
    conversation_trace: list[dict],
    *,
    schema_known: bool,
    call_index: int,
) -> str:
    """Describe where one argument value comes from, as a single bullet line."""
    schema_note = _schema_note(arg_value, param_spec, schema_known=schema_known)

    # A value the schema itself allows (enum member / declared default) already
    # has its reference, so skip the transcript search to keep the block short.
    schema_grounded = schema_note.startswith(("allowed by schema", "matches the schema"))
    sources = [] if schema_grounded else _value_sources(arg_value, conversation_trace, before=call_index)

    source_note = f"appears in {', '.join(sources)}" if sources else "no matching text in user turns or tool responses"
    return f"- {arg_name} = {_display(arg_value)}: {schema_note}; {source_note}"


def _schema_note(arg_value: Any, param_spec: dict[str, Any] | None, *, schema_known: bool) -> str:
    """State what the tool schema says about this value."""
    if not schema_known:
        return "no schema available"
    if param_spec is None:
        return "not among the parameters declared for this tool"

    enum = param_spec.get("enum")
    if isinstance(enum, list) and enum:
        allowed = ", ".join(str(option) for option in enum)
        if any(_same_value(arg_value, option) for option in enum):
            return f"allowed by schema (one of: {allowed})"
        return f"NOT one of the schema's allowed values ({allowed})"

    if "default" in param_spec and _same_value(arg_value, param_spec["default"]):
        return "matches the schema-declared default"

    return "no schema constraint"


def _same_value(value: Any, declared: Any) -> bool:
    """The reference's schema comparison: exact equality, with case-insensitivity only
    between two strings, so ``True`` does not match ``False`` and a boolean never
    matches a string default (``deref/build/paths.py::is_default``)."""
    if value == declared:
        return True
    return isinstance(value, str) and isinstance(declared, str) and value.lower() == declared.lower()


def _value_sources(arg_value: Any, conversation_trace: list[dict], *, before: int) -> list[str]:
    """Find earlier trace entries that contain the value.

    The reference searches the call's history only (question, earlier calls'
    arguments, earlier returns), so nothing at or after the described call is a
    source: a value cannot be grounded in the call that proposes it or in later
    text. Assistant prose is not searched either — the reference's history is
    user turns, tool responses, and earlier tool calls.
    """
    sources: list[str] = []
    for position, entry in enumerate(conversation_trace or []):
        if position >= before:
            break
        if not isinstance(entry, dict):
            continue
        turn_id = int(entry.get("turn_id") or 0)
        if entry.get("role") == "user":
            text, label = _as_text(entry.get("content")), f"user turn {turn_id}"
        elif entry.get("type") == "tool_response":
            tool_name = entry.get("tool_name") or "tool"
            text = _as_text(entry.get("tool_response"))
            label = f"{tool_name} response (turn {turn_id})"
        elif entry.get("type") == "tool_call":
            tool_name = entry.get("tool_name") or "tool"
            text, label = _as_text(entry.get("parameters")), f"earlier {tool_name} call (turn {turn_id})"
        else:
            continue
        if _occurs(arg_value, text):
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


# ---------------------------------------------------------------- occurrence matching (reference rules)
def _norm(value: Any) -> str:
    """Value as searchable text: strings as they are, lists / dicts as compact
    sorted JSON, other values as JSON (``deref/build/paths.py::norm``)."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return json.dumps(value, default=str)


def _numeric(value: Any) -> bool:
    """A number (not a boolean) or a string that is one number, so that e.g. "1,250"
    gets the stricter numeric boundaries (``deref/build/paths.py::numeric``)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value.strip()) is not None


def _contains(needle: str, text: str, boundary: bool, numeric: bool = False) -> bool:
    """Substring test with the reference's boundary rules: word boundaries
    (``[A-Za-z0-9]``) for non-string and short values, and for numbers the rule
    that no digit / letter / "." may precede and no digit / letter / decimal may
    follow (``deref/build/paths.py::contains``)."""
    if numeric:
        pattern = r"(?<![A-Za-z0-9.])" + re.escape(needle) + r"(?![A-Za-z0-9])(?!\.\d)"
        return re.search(pattern, text) is not None
    if not boundary:
        return needle in text
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])", text) is not None


def _lowws(text: str) -> str:
    """Lower-cased text with runs of whitespace collapsed (``deref/build/paths.py::lowws``)."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _lenient_occurs(needle: str, text: str) -> bool:
    """Lenient occurrence of the reference's quote verification: the longest common
    substring with the text is at least 30 characters and at least 60% of the value
    (``deref/inputs/evidence.py::verify2``)."""
    if len(needle) < _LCS_MIN_CHARS:
        return False
    match = difflib.SequenceMatcher(None, text, needle, autojunk=False).find_longest_match(
        0, len(text), 0, len(needle)
    )
    return match.size >= _LCS_MIN_CHARS and match.size >= _LCS_MIN_FRACTION * len(needle)


def _occurs_normalised(value: Any, text: str) -> bool:
    """Case- and whitespace-insensitive pass of the reference's normalised matching
    (the "case / whitespace" rule of ``deref/build/paths.py::p3_scalar``)."""
    if isinstance(value, list):
        return all(_occurs_normalised(x, text) for x in value)
    folded = _lowws(_norm(value))
    return bool(folded) and _contains(folded, _lowws(text), len(folded) < 4, _numeric(value))


def _occurs(value: Any, text: str) -> bool:
    """Whether the value occurs in `text`, by the reference's occurrence rules.

    Exact text first (word / numeric boundaries for non-string, short, and numeric
    values; a list occurs when every element occurs), then the case- and
    whitespace-insensitive pass, then the lenient longest-common-substring check
    for string values.
    """
    if value is None or value == "" or value == [] or value == {}:
        return False
    if isinstance(value, list) and all(_occurs(x, text) for x in value):
        return True
    exact = _norm(value)
    if _contains(exact, text, not isinstance(value, str) or len(exact) < 4, _numeric(value)):
        return True
    if _occurs_normalised(value, text):
        return True
    return isinstance(value, str) and _lenient_occurs(exact, text)


def _as_text(value: Any) -> str:
    """Coerce a trace field to text for substring matching."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, default=str)
