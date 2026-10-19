"""Derived-reference stage for tool-call argument judges.

LLM judges that verify tool-call arguments accept a correct value less often
when their input holds no reference for it — the value is a schema default the
user never mentioned, must be inferred (a standard domain mapping, a policy
entitlement, simple arithmetic on tool results), or its source text was cut
from the judge input. Judges follow the reference in their input: they favor a
visible incorrect value over a correct value whose source is hidden.

This module adapts DeRef (Derived Reference, arXiv:2610.18497, "Judges Follow
the Reference: The Derivation Gap in Verifying Tool-Call Arguments") for eva's
faithfulness judge. An auxiliary LLM call reads the conversation's grounding
sources — agent instructions, tool schemas, user utterances, and raw tool
responses — *without* seeing the assistant's tool calls (the candidates under
evaluation), and writes a reference sheet naming which values are grounded for
which tool parameters. The judge prompt embeds that sheet so verification runs
against an explicit reference instead of relying only on hand-written
"Is NOT a flag" exception clauses.

Opt-in via the faithfulness metric config (``derived_reference: true``) or the
``FAITHFULNESS_DERIVED_REFERENCE`` environment variable. Derivation failures
degrade to a neutral note — the judge always runs.
"""

import json
from logging import Logger
from typing import Any, NamedTuple

from eva.metrics.base import MetricContext
from eva.utils.llm_client import LLMClient
from eva.utils.prompt_manager import PromptManager

# Note returned when derivation is enabled but the conversation has no tool calls.
NO_TOOL_CALLS_NOTE = (
    "No tool calls appear in this conversation, so there are no tool-call arguments "
    "to verify against a derived reference."
)

# Note returned when the derivation call fails or returns nothing usable.
FALLBACK_NOTE = (
    "Reference derivation is unavailable for this conversation. Verify tool-call "
    "arguments directly against the user statements, tool responses, and agent "
    "instructions in the conversation trace."
)

# PromptManager path of the aux derivation template (configs/prompts/judge.yaml).
DEFAULT_PROMPT_PATH = "judge.faithfulness.reference_derivation_prompt"


class DerivedReference(NamedTuple):
    """Result of the derivation stage.

    text is the reference sheet (or a fallback note) to embed in the judge
    prompt and is never empty; prompt and usage are the aux call's prompt and
    token usage, empty/None when no LLM call was made, so callers can account
    for the extra cost alongside the judge's own usage.
    """

    text: str
    prompt: str = ""
    usage: dict[str, Any] | None = None


def has_tool_calls(conversation_trace: list[dict[str, Any]]) -> bool:
    """Return True if the trace contains at least one tool-call entry."""
    return any(entry.get("type") == "tool_call" for entry in conversation_trace)


def format_grounding_transcript(conversation_trace: list[dict[str, Any]]) -> str:
    """Format the grounding sources only: user utterances and raw tool responses.

    Assistant utterances and tool calls are deliberately excluded — tool-call
    arguments are the candidates the judge will evaluate, and the derivation
    must stay blind to them (DeRef's candidate-blind property) so the reference
    reflects what the sources support rather than what the assistant claimed.
    """
    lines: list[str] = []
    for entry in conversation_trace:
        if entry.get("role") == "user":
            lines.append(f"user: {entry.get('content', '')}")
        elif entry.get("type") == "tool_response":
            lines.append(f"tool_response {entry.get('tool_name', '')}: {entry.get('tool_response', '')}")
    return "\n".join(lines)


async def derive_tool_reference(
    llm_client: LLMClient,
    prompt_manager: PromptManager,
    context: MetricContext,
    logger: Logger,
    prompt_path: str = DEFAULT_PROMPT_PATH,
) -> DerivedReference:
    """Derive a candidate-blind reference sheet for tool-call arguments.

    Never raises: on any failure the fallback note is returned so the judge
    still runs with its usual grounding rules.
    """
    if not has_tool_calls(context.conversation_trace):
        return DerivedReference(NO_TOOL_CALLS_NOTE)

    try:
        prompt = prompt_manager.get_prompt(
            prompt_path,
            agent_instructions=context.agent_instructions,
            available_tools=json.dumps(context.agent_tools, indent=4),
            grounding_transcript=format_grounding_transcript(context.conversation_trace),
        )
        response_text, usage = await llm_client.generate_text([{"role": "user", "content": prompt}])
    except Exception as e:
        logger.warning(f"[{context.record_id}] reference derivation failed, falling back to judge-only grounding: {e}")
        return DerivedReference(FALLBACK_NOTE)

    sheet = (response_text or "").strip()
    if not sheet:
        logger.warning(f"[{context.record_id}] reference derivation returned empty output, falling back")
        return DerivedReference(FALLBACK_NOTE)
    return DerivedReference(sheet, prompt=prompt, usage=usage)
