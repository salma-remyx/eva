"""Derived-reference stage for tool-call argument judges.

LLM judges that verify tool-call arguments accept a correct value less often
when their input holds no reference for it — the value is a schema default the
user never mentioned, must be inferred (a standard domain mapping, a policy
entitlement, simple arithmetic on tool results), or its source text was cut
from the judge input. Judges follow the reference in their input: they favor a
visible incorrect value over a correct value whose source is hidden.

This module adapts DeRef (arXiv:2610.18497, "Judges Follow the Reference: The
Derivation Gap in Verifying Tool-Call Arguments") for eva's faithfulness judge.
As in the reference implementation, one derivation prompt is issued per
tool-call argument: the deriver sees the question and the history of all
earlier calls with their returns (assistant-authored arguments included), plus
the Function and Parameter lines of the argument, and is asked — instruction
F2, kept verbatim — what the value of the parameter should be and where in the
context it comes from, quoting the source or saying the context does not
determine it, in at most 3 sentences. The proposed value itself is never shown
(the candidate-blind property). Each answer is embedded in the judge prompt as
its own block under the reference's neutral header ("Expected value (written
before seeing the proposed value):") with no consumption instructions — the
reference deliberately adds none, and the paper's headline finding is that
judges follow the reference in their input.

Harness differences from the reference, declared here:

- Deriver model: the paper derives references with its own model (Qwen3.8-27B;
  the deref_qwen3_8b ablation shows deriver choice moves results). Configure a
  separate deriver with the faithfulness metric's ``deriver_model`` setting;
  when unset, the aux calls run on the judge's own LLMClient.
- Token-budget ladders: the reference fits each derivation prompt to 32,768
  Qwen tokens and each judge input to 2,048 CLM tokens with deletion ladders
  (drop earliest returns, then earliest steps, then shorten the question).
  eva's judge and deriver are API models that already receive the full
  history, so there is no fixed context budget to fit and no ladder is
  applied — the blocks supplement the trace rather than restore cut source
  text.

Opt-in via the faithfulness metric config (``derived_reference: true``) or the
``FAITHFULNESS_DERIVED_REFERENCE`` environment variable. Derivation failures
drop that argument's block — the judge always runs, as the reference keeps the
Plain input for arguments whose derivation produced no answer.
"""

import asyncio
import json
from logging import Logger
from typing import Any, NamedTuple

from eva.metrics.base import MetricContext
from eva.utils.llm_client import LLMClient
from eva.utils.prompt_manager import PromptManager

# DeRef's derivation system message (deref.generate.render.SYSTEM) and history
# header (deref.generate.render.HISTORY_HEADER).
DEREF_SYSTEM = "You are a careful evaluator of tool calls made by an AI assistant."
HISTORY_HEADER = "History (all earlier calls and their returns, oldest first):"

# Header of the block each answer is embedded under (deref.inputs.reference.HEAD).
BLOCK_HEADER = "Expected value (written before seeing the proposed value):"

# DeRef's thinking-mode derivation generation budget (scripts/22_generate.py --max-new 8192).
DERIVATION_MAX_TOKENS = 8192

# Parameter with no description in the tool schema (deref.inputs.state.NO_DESC).
NO_DESCRIPTION = "(no description available)"

# Note returned when derivation is enabled but the conversation has no tool-call
# arguments to derive a reference for.
NO_TOOL_CALLS_NOTE = (
    "No tool-call arguments appear in this conversation, so there are no expected values to derive."
)

# Note returned when no derivation produced an answer.
FALLBACK_NOTE = "Reference derivation is unavailable for this conversation."

# PromptManager path of the per-argument derivation template (configs/prompts/judge.yaml).
DEFAULT_PROMPT_PATH = "judge.faithfulness.reference_derivation_prompt"


class AuxCall(NamedTuple):
    """One derivation LLM call: its prompt, token usage, and raw response."""

    prompt: str
    usage: dict[str, Any] | None = None
    response: str = ""


class DerivedReferences(NamedTuple):
    """Result of the derivation stage.

    text is the per-argument reference-block section (or a fallback note) to
    embed in the judge prompt and is never empty; calls lists the aux LLM calls
    so callers can account for their cost alongside the judge's own usage.
    """

    text: str
    calls: list[AuxCall] = []


def has_tool_calls(conversation_trace: list[dict[str, Any]]) -> bool:
    """Return True if the trace contains at least one tool-call entry."""
    return any(entry.get("type") == "tool_call" for entry in conversation_trace)


def answer_text(output: str) -> str | None:
    """Answer part of a derivation output, or None if there is none.

    With a thinking deriver the answer is the text after the last </think>
    (deref.inputs.reference.answer); without thinking tags the whole output is
    the answer (the reference's thinking-off form).
    """
    if "</think>" in output:
        output = output.rsplit("</think>", 1)[1]
    text = output.strip()
    return text or None


def _tool_definition(context: MetricContext, tool_name: str) -> dict[str, Any] | None:
    """The agent's tool definition named ``tool_name``, or None."""
    for tool in context.agent_tools or []:
        if isinstance(tool, dict) and tool.get("name") == tool_name:
            return tool
    return None


def function_line(tool_name: str, tool_def: dict[str, Any] | None) -> str:
    """``Function: <name> — <description>`` of the reference's argument tail."""
    description = (tool_def or {}).get("description")
    return f"Function: {tool_name} — {description}" if description else f"Function: {tool_name}"


def _parameter_spec(tool_def: dict[str, Any] | None, parameter: str) -> dict[str, Any]:
    if not isinstance(tool_def, dict):
        return {}
    params = list(tool_def.get("required_parameters") or []) + list(tool_def.get("optional_parameters") or [])
    for spec in params:
        if isinstance(spec, dict) and spec.get("name") == parameter:
            return spec
    return {}


def parameter_line(tool_def: dict[str, Any] | None, parameter: str) -> str:
    """``Parameter: <param> — <info>; type: …; enum: […]`` of the reference's argument tail."""
    spec = _parameter_spec(tool_def, parameter)
    info = spec.get("description") or NO_DESCRIPTION
    addition = ""
    if spec.get("type"):
        addition += f"; type: {spec['type']}"
    if spec.get("enum"):
        addition += f"; enum: {json.dumps(spec['enum'])}"
    return f"Parameter: {parameter} — {info}{addition}"


def locator_context(conversation_trace: list[dict[str, Any]], call_index: int) -> tuple[str, str]:
    """(question, history) of the derivation input for the tool call at ``call_index``.

    Reference layout: the user request as the Question, then every earlier call
    with its arguments and its return as the History. Candidate-blind means the
    proposed value is never rendered — earlier calls are part of the history,
    exactly as in the reference. DeRef's benchmark data has a single request;
    eva conversations have several user turns, so all user utterances before
    the call are folded into the Question.
    """
    questions: list[str] = []
    history: list[str] = []
    step = 0
    for entry in conversation_trace[:call_index]:
        if entry.get("role") == "user":
            questions.append(str(entry.get("content", "")))
        elif entry.get("type") == "tool_call":
            step += 1
            args = json.dumps(entry.get("parameters") or {}, ensure_ascii=False)
            history.append(f"[call, step {step}] {entry.get('tool_name', '')}({args})")
        elif entry.get("type") == "tool_response":
            response = entry.get("tool_response", "")
            if not isinstance(response, str):
                response = json.dumps(response, ensure_ascii=False)
            history.append(f"[return, step {step}] {response}")
    return "\n".join(questions), ("\n".join(history) if history else "none")


def argument_slots(conversation_trace: list[dict[str, Any]]) -> list[tuple[int, int, str, str]]:
    """[(call_ordinal, trace_index, tool_name, parameter)] of every tool-call argument, in trace order."""
    slots: list[tuple[int, int, str, str]] = []
    ordinal = 0
    for index, entry in enumerate(conversation_trace):
        if entry.get("type") != "tool_call":
            continue
        ordinal += 1
        parameters = entry.get("parameters")
        if not isinstance(parameters, dict):
            continue
        for parameter in parameters:
            slots.append((ordinal, index, entry.get("tool_name", ""), parameter))
    return slots


async def derive_tool_references(
    llm_client: LLMClient,
    prompt_manager: PromptManager,
    context: MetricContext,
    logger: Logger,
    prompt_path: str = DEFAULT_PROMPT_PATH,
) -> DerivedReferences:
    """Derive one expected-value block per tool-call argument. Never raises.

    Each argument gets DeRef's per-argument derivation prompt (question, earlier
    calls and their returns, Function / Parameter lines, F2 — never the proposed
    value). An argument whose call fails or yields no answer simply gets no
    block, as the reference keeps the Plain input for it; if none produce an
    answer, the fallback note is returned so the judge still runs.
    """
    slots = argument_slots(context.conversation_trace)
    if not slots:
        return DerivedReferences(NO_TOOL_CALLS_NOTE)

    async def derive(slot: tuple[int, int, str, str]) -> tuple[str | None, AuxCall | None]:
        ordinal, index, tool_name, parameter = slot
        label = f"{tool_name}.{parameter}"
        try:
            question, history = locator_context(context.conversation_trace, index)
            tool_def = _tool_definition(context, tool_name)
            prompt = prompt_manager.get_prompt(
                prompt_path,
                question=question,
                history=history,
                function_line=function_line(tool_name, tool_def),
                parameter_line=parameter_line(tool_def, parameter),
                parameter_name=parameter,
            )
        except Exception as e:
            logger.warning(f"[{context.record_id}] reference derivation prompt failed for {label}: {e}")
            return None, None
        try:
            response, usage = await llm_client.generate_text(
                [{"role": "system", "content": DEREF_SYSTEM}, {"role": "user", "content": prompt}]
            )
        except Exception as e:
            logger.warning(f"[{context.record_id}] reference derivation failed for {label}, skipping its block: {e}")
            return None, None
        answer = answer_text(response or "")
        if answer is None:
            logger.warning(
                f"[{context.record_id}] reference derivation returned no answer for {label}, skipping its block"
            )
            return None, None
        block = f"{label} (call {ordinal})\n{BLOCK_HEADER}\n{answer}"
        return block, AuxCall(prompt, usage, response)

    results = await asyncio.gather(*(derive(slot) for slot in slots))
    blocks = [block for block, _ in results if block]
    calls = [call for _, call in results if call]
    return DerivedReferences("\n\n".join(blocks) if blocks else FALLBACK_NOTE, calls)
