"""Derived references for tool-call arguments, injected into LLM judge prompts.

Adapted from "Judges Follow the Reference: The Derivation Gap in Verifying
Tool-Call Arguments" (arXiv:2610.18497). The paper shows that tool-call judges
accept a correct argument value less often when their input holds no reference
for it: the value is a schema default the user never mentioned, has to be
inferred from context, or comes from text the judge was not shown. The authors
call this the *derivation gap*, and show it is a property of the judge input —
judges follow the reference they can see, favouring a visible incorrect value
over a correct value whose source is hidden.

Their remedy (DeRef) derives a reference for each argument from the history
*without looking at the candidate value* and hands that reference to the judge,
so verification happens against an explicit reference instead of a derivation
the judge has to perform implicitly.

This module is a deterministic adaptation: rather than a second model reading
the history, references are derived mechanically from the agent's tool schemas
and from the conversation's user turns and tool responses. Two properties of
DeRef are preserved deliberately:

* **Candidate-blindness** — derivation reads parameter *names* only. The values
  the assistant actually sent are never consulted, so the derived reference
  cannot be anchored on the answer it is meant to check.
* **Explicit absence** — when nothing can be derived that is stated plainly, and
  labelled as *not* evidence of fabrication. A missing reference is a limitation
  of this heuristic ledger, not a signal about the assistant.
"""

import json
import re
from collections.abc import Iterator
from typing import Any

from eva.utils.logging import get_logger

logger = get_logger(__name__)

# Bounds on the ledger, so one verbose tool response cannot flood the judge prompt.
_MAX_VALUES_PER_FIELD = 4
_MAX_REFERENCES_PER_PARAM = 3
_MAX_NUMBERS = 16
_MAX_ARITHMETIC_VALUES = 4
_MAX_ARITHMETIC_DERIVATIONS = 6
_MAX_USER_LITERALS = 8

# Salient literals in user speech, keyed by the parameter semantics they ground.
_USER_LITERAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "code": re.compile(r"\b[A-Z][A-Z0-9]{2,11}\b"),
    "number": re.compile(r"\b\d[\d,]*(?:\.\d+)?\b"),
    "name": re.compile(r"\b[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?\b"),
}

# Declared types that rule out name-based numeric inference.
_NON_NUMERIC_TYPES = ("string", "boolean", "bool", "array", "object")

# Capitalised words that open a sentence rather than name an entity.
_NAME_STOPWORDS = frozenset(
    "A And Are But Can Could Do Does He Hello Hi I If It My No Okay Or She So "
    "Thank Thanks That The Then They This We Well What When Where Yes You".split()
)

_NO_REFERENCE_NOTE = (
    "no reference derived — this is a limitation of the derivation above, **not** evidence of "
    "fabrication; trace the value in the transcript yourself before flagging"
)

_NO_TOOL_CALLS_NOTE = "No tool calls with arguments in this conversation."


class _SourceLedger:
    """Candidate-blind index of the reference values a conversation offers.

    Populated while walking the trace in order, so at any tool call the ledger
    holds exactly the sources that preceded it — the same view the assistant had.
    """

    def __init__(self) -> None:
        self.fields: dict[str, list[tuple[str, Any]]] = {}
        self.numbers: list[float] = []
        self.user_literals: dict[str, list[str]] = {kind: [] for kind in _USER_LITERAL_PATTERNS}

    def add_tool_response(self, tool_name: str, turn_id: Any, payload: Any) -> None:
        """Index every scalar field of a tool response as a candidate reference."""
        for path, value in _scalar_leaves(_coerce_json(payload)):
            if not path:
                continue
            label = f"{tool_name} [turn {turn_id}] → {path}"
            leaf = path.rsplit(".", 1)[-1]
            for key in (leaf, path):
                bucket = self.fields.setdefault(key, [])
                if len(bucket) < _MAX_VALUES_PER_FIELD:
                    bucket.append((label, value))
            if isinstance(value, (int, float)) and not isinstance(value, bool) and len(self.numbers) < _MAX_NUMBERS:
                self.numbers.append(float(value))

    def add_user_turn(self, text: str) -> None:
        """Extract salient literals from a user utterance, keyed by what they can ground."""
        # Dates are captured whole; strip them first so their parts are not also read as numbers.
        remainder = _USER_LITERAL_PATTERNS["date"].sub(" ", text)
        for kind, pattern in _USER_LITERAL_PATTERNS.items():
            bucket = self.user_literals[kind]
            haystack = text if kind == "date" else remainder
            for raw in pattern.findall(haystack):
                literal = raw.strip().rstrip(",.")
                if len(bucket) >= _MAX_USER_LITERALS:
                    break
                if not literal or literal in bucket:
                    continue
                if kind == "code" and not any(char.isdigit() for char in literal):
                    continue
                if kind == "name" and literal in _NAME_STOPWORDS:
                    continue
                bucket.append(literal)

    def lookup(self, param_name: str) -> list[tuple[str, Any]]:
        """Return prior tool-result values whose field name matches ``param_name``."""
        target = param_name.lower().rstrip("s")
        exact: list[tuple[str, Any]] = []
        partial: list[tuple[str, Any]] = []
        for key, hits in self.fields.items():
            candidate = key.lower().rstrip("s")
            if candidate == target:
                exact.extend(hits)
            elif (
                len(candidate) >= 4 and len(target) >= 4 and (candidate.endswith(target) or target.endswith(candidate))
            ):
                partial.extend(hits)
        return (exact or partial)[:_MAX_REFERENCES_PER_PARAM]

    def user_literals_for(self, param_name: str, spec: dict[str, Any] | None) -> list[str]:
        """Return user-stated literals that could ground ``param_name``."""
        hits: list[str] = []
        for kind in _semantic_kinds(param_name, spec):
            for literal in self.user_literals.get(kind, []):
                if literal not in hits:
                    hits.append(literal)
        return hits[:_MAX_REFERENCES_PER_PARAM]

    def arithmetic(self) -> list[str]:
        """Return sum and difference derivations over the numbers seen in prior results."""
        # Prefer the most recent results, but present them in the order they appeared.
        values: list[float] = []
        for value in reversed(self.numbers):
            if value not in values:
                values.append(value)
            if len(values) >= _MAX_ARITHMETIC_VALUES:
                break
        values.reverse()

        derivations: list[str] = []
        for index, first in enumerate(values):
            for second in values[index + 1 :]:
                left, right = _fmt_number(first), _fmt_number(second)
                derivations.append(f"{left} + {right} = {_fmt_number(first + second)}")
                derivations.append(f"|{left} - {right}| = {_fmt_number(abs(first - second))}")
        return derivations[:_MAX_ARITHMETIC_DERIVATIONS]


def build_tool_argument_references(
    agent_tools: list[dict] | None,
    conversation_trace: list[dict] | None,
) -> str:
    """Derive a prompt-ready reference block for every tool-call argument in a trace.

    Args:
        agent_tools: The agent's declared tool schemas (``MetricContext.agent_tools``).
        conversation_trace: The conversation trace, including tool calls and responses.

    Returns:
        A markdown block naming, per tool call and per parameter, the reference the
        argument should be verified against. Argument values are never read.
    """
    try:
        return _build(agent_tools or [], conversation_trace or [])
    except Exception as error:  # pragma: no cover - defensive: a prompt aid must never fail the metric
        logger.warning(f"Failed to derive tool-argument references: {error}")
        return _NO_TOOL_CALLS_NOTE


def _build(agent_tools: list[dict], conversation_trace: list[dict]) -> str:
    """Walk the trace in order, deriving references from the sources seen so far."""
    specs = _tool_parameter_specs(agent_tools)
    ledger = _SourceLedger()
    blocks: list[str] = []
    for entry in conversation_trace:
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("type")
        if entry_type == "tool_call":
            block = _call_block(entry, specs, ledger)
            if block:
                blocks.append(block)
        elif entry_type == "tool_response":
            tool_name = entry.get("tool_name")
            ledger.add_tool_response(
                tool_name if isinstance(tool_name, str) else "tool",
                entry.get("turn_id", 0),
                entry.get("tool_response"),
            )
        elif entry.get("role") == "user":
            content = entry.get("content")
            if isinstance(content, str):
                ledger.add_user_turn(content)

    if not blocks:
        return _NO_TOOL_CALLS_NOTE
    return "\n\n".join(blocks)


def _call_block(
    entry: dict,
    specs: dict[str, dict[str, dict[str, Any]]],
    ledger: _SourceLedger,
) -> str | None:
    """Render the reference card for one tool call, using parameter names only."""
    tool_name = entry.get("tool_name")
    parameters = entry.get("parameters")
    if not isinstance(tool_name, str) or not isinstance(parameters, dict) or not parameters:
        return None

    lines = [f"**Turn {entry.get('turn_id', '?')} — {tool_name}**"]
    param_specs = specs.get(tool_name)
    if param_specs is None:
        lines.append("  - this tool is not in the declared toolset — verify it against Available Tools above")
        return "\n".join(lines)

    for param_name in parameters:
        bits = _parameter_bits(param_name, param_specs.get(param_name), ledger)
        if not bits:
            lines.append(f"  - `{param_name}`: {_NO_REFERENCE_NOTE}")
        else:
            lines.append(f"  - `{param_name}`:")
            lines.extend(f"      - {bit}" for bit in bits)
    return "\n".join(lines)


def _parameter_bits(
    param_name: str,
    spec: dict[str, Any] | None,
    ledger: _SourceLedger,
) -> list[str]:
    """Derive every reference source available for one parameter name."""
    bits: list[str] = []
    if spec:
        default = spec.get("default")
        if default not in (None, ""):
            bits.append(f"schema default `{default}` — using it is grounded even though the user never stated it")
        enum = spec.get("enum")
        if isinstance(enum, list) and enum:
            allowed = "/".join(str(option) for option in enum)
            bits.append(f"categorical — allowed values {allowed}; grounded when it matches the user's situation")

    bits.extend(f"prior tool result {label} = `{value}`" for label, value in ledger.lookup(param_name))
    bits.extend(f"user-stated `{literal}`" for literal in ledger.user_literals_for(param_name, spec))
    if _is_numeric_parameter(param_name, spec):
        bits.extend(f"arithmetic over prior results: {derivation}" for derivation in ledger.arithmetic())
    return bits


def _tool_parameter_specs(agent_tools: list[dict]) -> dict[str, dict[str, dict[str, Any]]]:
    """Map tool name to parameter name to parameter spec, for EVA and OpenAI tool shapes."""
    specs: dict[str, dict[str, dict[str, Any]]] = {}
    for tool in agent_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = tool.get("name") or (function.get("name") if isinstance(function, dict) else None)
        if not isinstance(name, str) or not name:
            continue

        params: dict[str, dict[str, Any]] = {}
        for key in ("required_parameters", "optional_parameters"):
            for entry in tool.get(key) or []:
                param_name = entry.get("name") if isinstance(entry, dict) else None
                if isinstance(param_name, str):
                    params[param_name] = entry
        if not params and isinstance(function, dict):
            properties = (function.get("parameters") or {}).get("properties")
            if isinstance(properties, dict):
                params = {key: value for key, value in properties.items() if isinstance(value, dict)}
        specs[name] = params
    return specs


def _semantic_kinds(param_name: str, spec: dict[str, Any] | None) -> list[str]:
    """Guess what kind of value a parameter wants, from its name and declared type."""
    lowered = param_name.lower()
    declared = spec.get("type") if spec else None
    ptype = declared.lower() if isinstance(declared, str) else ""

    kinds: list[str] = []
    if "date" in lowered or ptype == "date":
        kinds.append("date")
    if "name" in lowered:
        kinds.append("name")
    numeric_names = ("amount", "count", "quantity", "fee", "total", "num_")
    # A declared non-numeric type outranks a numeric-sounding name (e.g. a `waive_fee` boolean).
    if ptype in ("integer", "number") or (ptype not in _NON_NUMERIC_TYPES and any(t in lowered for t in numeric_names)):
        kinds.append("number")
    if any(token in lowered for token in ("number", "code", "id")) and "number" not in kinds:
        kinds.append("code")
    return kinds


def _is_numeric_parameter(param_name: str, spec: dict[str, Any] | None) -> bool:
    """Return whether a parameter is expected to hold a number."""
    return "number" in _semantic_kinds(param_name, spec)


def _scalar_leaves(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every scalar leaf of a JSON-like value."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _scalar_leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for item in node:
            yield from _scalar_leaves(item, path)
    elif node is not None:
        yield path, node


def _coerce_json(payload: Any) -> Any:
    """Parse a string payload as JSON so stringified tool responses still contribute."""
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except ValueError:
            return None
    return payload


def _fmt_number(value: float) -> str:
    """Render a number without a trailing ``.0`` when it is integral."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
