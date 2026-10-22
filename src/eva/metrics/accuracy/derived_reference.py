"""Candidate-blind derived reference for tool-argument grounding.

Adapted from DeRef in "Judges Follow the Reference: The Derivation Gap in
Verifying Tool-Call Arguments" (arXiv:2610.18497). The paper finds that
tool-call judges accept a correct argument value less often when their input
holds no reference for it — typically because the value is a schema default
the user never mentioned, or a value whose source text is absent from the
judge input. It calls this the *derivation gap*, and closes it by having a
separate pass read the history WITHOUT the candidate argument and write a
reference for the judge.

This module is a deterministic adaptation of that idea for eva's tool
schemas: the reference is derived from the tool specifications (declared
defaults, optional parameters) plus a vocabulary-overlap scan of the user's
utterances. Both inputs exist independently of what the assistant actually
sent, so the derivation is candidate-blind by construction — the agent's
tool-call arguments are structurally never seen (see
:func:`extract_user_utterances`).
"""

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

# Keys a parameter spec may use to declare a default value.
_DEFAULT_KEYS = ("default", "default_value")


@dataclass(frozen=True)
class ParameterReference:
    """Derived grounding reference for a single tool parameter."""

    tool: str
    parameter: str
    required: bool
    default: Any = None
    enum_values: tuple[str, ...] = ()
    attested_values: tuple[str, ...] = ()


def extract_user_utterances(conversation_trace: list[dict[str, Any]] | None) -> list[str]:
    """Return the user-role utterances from a conversation trace.

    Only user turns are kept: they are the sole speech source the reference is
    allowed to ground values in. Assistant turns and tool entries (which carry
    the assistant's actual argument values — the candidate) are excluded so
    the derivation cannot see, or leak, the values it is judging.
    """
    utterances: list[str] = []
    for entry in conversation_trace or []:
        if isinstance(entry, dict) and entry.get("role") == "user" and entry.get("content"):
            utterances.append(str(entry["content"]))
    return utterances


def derive_argument_reference(
    user_utterances: Sequence[str], agent_tools: list[dict[str, Any]] | None
) -> list[ParameterReference]:
    """Derive grounding references for tool parameters without seeing the candidate.

    Args:
        user_utterances: User speech only (see :func:`extract_user_utterances`).
        agent_tools: Raw agent tool schemas. Both the eva agent-YAML dialect
            (``required_parameters``/``optional_parameters`` lists) and the
            OpenAI function-calling dialect (``parameters.properties`` with a
            ``required`` name list) are understood.

    Returns:
        One entry per parameter that carries a grounding signal (a declared
        default, optionality, or an enum vocabulary). Required parameters with
        no default and no enum produce no entry — their grounding is simply
        "trace it to the conversation", which the judge already does.
    """
    normalized_user_text = "\n".join(user_utterances).lower()
    references: list[ParameterReference] = []
    for tool_name, parameter_name, spec, required in _iter_parameters(agent_tools or []):
        default = next((spec[key] for key in _DEFAULT_KEYS if spec.get(key) is not None), None)
        enum_values = tuple(value for value in spec.get("enum") or [] if isinstance(value, str))
        attested = tuple(value for value in enum_values if _is_attested(value, normalized_user_text))
        if default is None and required and not enum_values:
            continue
        references.append(
            ParameterReference(
                tool=tool_name,
                parameter=parameter_name,
                required=required,
                default=default,
                enum_values=enum_values,
                attested_values=attested,
            )
        )
    return references


def format_argument_reference(references: Sequence[ParameterReference]) -> str:
    """Render derived references as a judge-prompt section.

    Returns an empty string when no parameter carries a grounding signal, so
    the prompt section collapses gracefully for agents without defaults or
    enums in their tool schemas.
    """
    if not references:
        return ""
    lines = [
        "## Derived Tool-Argument Reference (candidate-blind)",
        "",
        "This reference was derived WITHOUT seeing the assistant's actual tool-call",
        "arguments — only from the tool schemas and the user's own utterances. When",
        "judging `fabricating_tool_parameters`, the following values are grounded, not",
        "fabricated, even if the user never said them aloud:",
        "- a parameter set to its schema-declared default value",
        "- omitting an optional parameter, or the value its description documents for the omit case",
        "- an enum value reported below as attested in the user's own words",
        "",
        "An enum parameter reported with no attested value means none of its values",
        "appear in the user's speech — check prior tool results before flagging such a",
        "value as fabricated.",
        "",
    ]
    lines.extend(_format_bullet(ref) for ref in references)
    return "\n".join(lines)


def build_argument_reference_note(
    conversation_trace: list[dict[str, Any]] | None, agent_tools: list[dict[str, Any]] | None
) -> str:
    """Build the derived-reference prompt section from a conversation trace and tool schemas."""
    return format_argument_reference(
        derive_argument_reference(extract_user_utterances(conversation_trace), agent_tools)
    )


def _iter_parameters(agent_tools: Iterable[Any]) -> Iterator[tuple[str, str, dict[str, Any], bool]]:
    """Yield ``(tool_name, parameter_name, parameter_spec, required)`` for each schema dialect."""
    for tool in agent_tools:
        if not isinstance(tool, dict):
            continue
        tool_name = str(tool.get("name") or tool.get("id") or "unknown_tool")
        # eva agent-YAML dialect: explicit required/optional parameter lists.
        for list_key, required in (("required_parameters", True), ("optional_parameters", False)):
            for spec in tool.get(list_key) or []:
                if isinstance(spec, dict) and spec.get("name"):
                    yield tool_name, str(spec["name"]), spec, required
        # OpenAI function-calling dialect: parameters.properties + parameters.required.
        parameters = tool.get("parameters")
        if isinstance(parameters, dict) and isinstance(parameters.get("properties"), dict):
            required_names = set(parameters.get("required") or [])
            for parameter_name, spec in parameters["properties"].items():
                spec = spec if isinstance(spec, dict) else {}
                yield tool_name, str(parameter_name), spec, parameter_name in required_names


def _is_attested(enum_value: str, normalized_user_text: str) -> bool:
    """Check whether any surface form of an enum value appears as whole word(s) in user speech.

    Word boundaries matter: the enum value ``any`` must not match inside
    ``many``, and ``no_preference`` should match the spoken phrase
    ``no preference``.
    """
    forms = {enum_value.lower(), enum_value.lower().replace("_", " ")}
    return any(re.search(rf"\b{re.escape(form)}\b", normalized_user_text) for form in forms)


def _format_bullet(ref: ParameterReference) -> str:
    """Render one parameter's grounding signals as a reference bullet."""
    label = "required" if ref.required else "optional"
    signals: list[str] = []
    if ref.default is not None:
        signals.append(f"schema default: {json.dumps(ref.default, ensure_ascii=False)}")
    if ref.enum_values:
        if ref.attested_values:
            quoted = ", ".join(json.dumps(value, ensure_ascii=False) for value in ref.attested_values)
            plural = "s" if len(ref.attested_values) > 1 else ""
            signals.append(f"user-attested enum value{plural}: {quoted}")
        else:
            signals.append("no enum value appears in the user's words")
    elif not ref.required and ref.default is None:
        signals.append("no schema default declared; omitting is grounded")
    suffix = f": {'; '.join(signals)}" if signals else ""
    return f"- **{ref.tool}.{ref.parameter}** ({label}){suffix}"
