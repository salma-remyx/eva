"""Spoken function calling metric — spoken-intent to function-call correctness.

Adapted from "Spoken Function Calling: A New Perspective on Spoken Language
Understanding for Large Audio Language Models" (arXiv:2608.05126). Spoken
Function Calling (SFC) reframes SLU as scoring whether a model, given a spoken
instruction, emits the *semantically correct* function (name + arguments)
against structured function definitions — moving beyond closed-set slot filling.

This is a Mode 2 adapted port. The paper's core mechanism — judging per-call
function-name and argument-value correctness for the user's spoken intent — is
kept at full fidelity. The paper's auxiliary machinery is substituted with
target-native equivalents: the SFC-Bench dataset and its multi-agent synthesis
pipeline are replaced by EVA's existing tool-using conversations (airline / HRSD
/ ITSM), the paper's "structured rule definitions" map onto the agent's
``agent_tools`` JSON schemas, and the paper's LALM post-training recipe is out of
scope for an evaluation framework.

This fills a measured gap in EVA's accuracy suite: ``tool_call_validity`` only
checks call *format* (it explicitly disclaims intent/correctness) and
``task_completion`` only checks the end-state DB hash. No existing metric scores
spoken-intent -> function-call correctness — a distinction that matters most on
the audio-native (Gemini ALM / S2S) path.
"""

import json
from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import get_assistant_turns_disclaimer, get_user_turns_disclaimer
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.results import MetricScore

# Spoken-function-calling sub-metric keys. The ``_accuracy`` suffix signals
# higher-is-better to eva.metrics.utils.direction_for_sub_metric.
_FUNCTION_NAME_ACCURACY = "function_name_accuracy"
_ARGUMENT_VALUE_ACCURACY = "argument_value_accuracy"

# Pipeline-specific framing for how the spoken instruction reaches the model.
_CASCADE_PERCEPTION_NOTE = (
    "The assistant works from a speech-to-text transcript of the user's spoken instruction. "
    "Transcription may distort the spoken intent (e.g. mis-transcribed confirmation numbers, "
    "names, or amounts). When judging correctness, distinguish a WRONG FUNCTION CHOICE — the "
    "model selected an action that does not match what the user asked for — from a correct "
    "function whose argument value was merely a transcription artifact. Score "
    "`correct_function` on whether the chosen action matches the user's intent; score "
    "`correct_arguments` on whether the values match what the user said (per the transcript) "
    "or a prior tool result returned."
)

_AUDIO_NATIVE_PERCEPTION_NOTE = (
    "The assistant processes the user's speech directly (audio-native, no intermediate "
    "transcript). It must map the spoken instruction to the correct function and argument "
    "values from raw audio. Audio perception errors — misheard letters, digits, names, and "
    "alphanumeric codes — are common with spoken input. This audio-native spoken-intent to "
    "function-call mapping is the core of Spoken Function Calling evaluation. Distinguish a "
    "WRONG FUNCTION CHOICE (a semantic misunderstanding of the spoken intent) from a correct "
    "function whose argument value was misheard."
)


@register_metric
class SpokenFunctionCallingMetric(ConversationTextJudgeMetric):
    """LLM judge of spoken-intent -> function-call correctness (whole conversation).

    Evaluates whether each tool call the assistant made selected the correct
    function for the user's spoken instruction and used semantically correct
    argument values (grounded in the spoken instruction or prior tool results),
    as opposed to merely well-formed calls.

    Rating scale: 3 (correct function calling), 2 (right functions, wrong arg
    values), 1 (wrong function chosen). Normalized: 3->1.0, 1->0.0.

    Sub-metrics ``function_name_accuracy`` and ``argument_value_accuracy`` are
    the per-call rates that make this Spoken Function Calling rather than a
    vague holistic verdict.
    """

    name = "spoken_function_calling"
    version = "v0.1"
    description = (
        "LLM judge of whether the assistant emitted the semantically correct function "
        "(name + arguments) for each spoken instruction"
    )
    category = "accuracy"
    default_model = "us.anthropic.claude-opus-4-6-v1"
    default_params = {"max_tokens": 100000}  # Drop the OpenAI-only flex tier inherited from TextJudgeMetric.
    rating_scale = (1, 3)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Skip when there are no tool calls to evaluate; otherwise judge."""
        if not context.tool_responses:
            return MetricScore(
                name=self.name,
                score=None,
                normalized_score=None,
                skipped=True,
                details={"note": "No tool calls to evaluate"},
            )
        return await super().compute(context)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "agent_instructions": context.agent_instructions,
            "agent_role": context.agent_role,
            "available_tools": json.dumps(context.agent_tools, indent=4),
            "conversation_trace": transcript_text,
            "current_date_time": context.current_date_time,
            "user_turns_disclaimer": get_user_turns_disclaimer(context.is_audio_native),
            "assistant_turns_disclaimer": get_assistant_turns_disclaimer(context.is_audio_native),
            "perception_note": _AUDIO_NATIVE_PERCEPTION_NOTE if context.is_audio_native else _CASCADE_PERCEPTION_NOTE,
        }

    def build_metric_score(
        self,
        rating: int,
        normalized: float,
        response: dict,
        prompt: str,
        context: MetricContext,
        raw_response: str | None = None,
    ) -> MetricScore:
        """Build MetricScore with per-call correctness breakdown sub-metrics."""
        calls = response.get("calls", []) if isinstance(response, dict) else []
        if not isinstance(calls, list):
            calls = []
        sub_metrics = _build_sfc_sub_metrics(self.name, calls)

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": response.get("explanation", ""),
                "calls": calls,
                "num_tool_calls": len(context.tool_responses),
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )


def _build_sfc_sub_metrics(parent_name: str, calls: list[dict]) -> dict[str, MetricScore]:
    """Build per-call correctness rate sub-metrics from the judge's ``calls`` list.

    ``function_name_accuracy`` = calls that selected the correct function / total
    calls; ``argument_value_accuracy`` = calls with semantically correct argument
    values / total calls. Both are higher-is-better rates (``_accuracy`` suffix).
    """
    sub_metrics: dict[str, MetricScore] = {}
    total = len(calls)
    if total == 0:
        return sub_metrics

    correct_function = sum(1 for call in calls if call.get("correct_function") is True)
    correct_arguments = sum(1 for call in calls if call.get("correct_arguments") is True)

    sub_metrics[_FUNCTION_NAME_ACCURACY] = make_rate_sub_metric(
        parent_name=parent_name,
        key=_FUNCTION_NAME_ACCURACY,
        numerator=correct_function,
        denominator=total,
        details={"correct": correct_function, "total": total},
        precision=4,
    )
    sub_metrics[_ARGUMENT_VALUE_ACCURACY] = make_rate_sub_metric(
        parent_name=parent_name,
        key=_ARGUMENT_VALUE_ACCURACY,
        numerator=correct_arguments,
        denominator=total,
        details={"correct": correct_arguments, "total": total},
        precision=4,
    )
    return sub_metrics
