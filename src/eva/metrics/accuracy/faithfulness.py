"""Faithfulness metric using LLM-as-judge (whole conversation)."""

import json
from typing import Any

from eva.metrics.accuracy.derived_tool_references import (
    REFERENCES_UNAVAILABLE_NOTE,
    _CURRENT_DERIVED_REFERENCES,
    generate_derived_tool_references,
    trace_has_tool_calls,
)
from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import (
    get_assistant_turns_disclaimer,
    get_misrepresentation_pipeline_note,
    get_user_turns_disclaimer,
)
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

_FAITHFULNESS_DIMENSION_KEYS = (
    "fabricating_tool_parameters",
    "misrepresenting_tool_result",
    "violating_policies",
    "failing_to_disambiguate",
    "hallucination",
)

# --- Pipeline-specific prompt text for faithfulness evaluation ---

_CASCADE_DISAMBIGUATION_CONTEXT = (
    "Since the assistant is working from a speech-to-text transcript, it should account for potential "
    "transcription errors, and clarify any ambiguity in the user's intent, especially when they lead to "
    "write/irreversible operations. It's not needed to clarify if the tools called are simple lookups, "
    "but if the lookups fail, the assistant is expected to clarify the user's intent."
)

_S2S_DISAMBIGUATION_CONTEXT = (
    "Since the assistant processes raw audio directly (speech-to-speech), it should account for potential "
    "audio perception errors — mishearing letters, numbers, names, or codes is common with spoken input. "
    "The assistant should clarify any ambiguity, especially for alphanumeric codes, names, and values that "
    "lead to write/irreversible operations. It's not needed to clarify if the tools called are simple "
    "lookups, but if the lookups fail, the assistant is expected to clarify the user's intent. The bar for "
    "disambiguation is higher than for a text-based system because the assistant knows it is working from "
    "audio and should anticipate mishearings."
)


@register_metric
class FaithfulnessJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based faithfulness metric (whole conversation).

    Evaluates whether the assistant remains faithful to information, policies,
    and instructions (no hallucinations, grounded tool calls, policy adherence,
    proper disambiguation).

    Rating scale: 1 (faithful), 0 (violations)
    Normalized: 1→1.0, 0→0.0
    """

    name = "faithfulness"
    version = "v0.3"
    description = (
        "LLM judge evaluation of whether the assistant remains faithful to information, policies, and instructions"
    )
    category = "accuracy"
    default_model = "us.anthropic.claude-opus-4-6-v1"
    default_params = {"max_tokens": 100000}  # Drop the OpenAI-only flex tier inherited from TextJudgeMetric.
    rating_scale = (1, 3)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Standard judge flow, preceded by a candidate-blind derived-reference pass.

        A helper model reads the conversation with tool-call argument values
        redacted and writes a reference sheet for the judge, so arguments that
        are correct but not literally present in the transcript (schema
        defaults, inferred values) are not flagged as fabricated. Each compute
        runs inside its own asyncio task, so the contextvar set here is
        isolated from concurrent records.
        """
        derived_references = await self._generate_derived_references(context)
        token = _CURRENT_DERIVED_REFERENCES.set(derived_references)
        try:
            return await super().compute(context)
        finally:
            _CURRENT_DERIVED_REFERENCES.reset(token)

    async def _generate_derived_references(self, context: MetricContext) -> str:
        """Run the derived-reference helper when the conversation has tool calls.

        Skipped (no helper cost) for conversations without tool calls, where no
        derivation gap is possible. Failures degrade to the unavailable note.
        """
        if not trace_has_tool_calls(context.conversation_trace):
            return REFERENCES_UNAVAILABLE_NOTE

        model = self.config.get("derived_reference_model") or self.llm_client.model
        params: dict[str, Any] = {"max_tokens": 8192}
        params.update(self.config.get("derived_reference_params", {}))

        def log_usage(prompt: str, response_text: str | None, usage: dict | None) -> None:
            self._log_token_usage(context, model, params, prompt, usage, response_text)

        return await generate_derived_tool_references(context, model=model, params=params, log_usage=log_usage)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        if context.is_audio_native:
            disambiguation_context = _S2S_DISAMBIGUATION_CONTEXT
        else:
            disambiguation_context = _CASCADE_DISAMBIGUATION_CONTEXT

        return {
            "agent_instructions": context.agent_instructions,
            "agent_role": context.agent_role,
            "available_tools": json.dumps(context.agent_tools, indent=4),
            "conversation_trace": transcript_text,
            "current_date_time": context.current_date_time,
            "user_turns_disclaimer": get_user_turns_disclaimer(context.is_audio_native),
            "assistant_turns_disclaimer": get_assistant_turns_disclaimer(context.is_audio_native),
            "misrepresentation_pipeline_note": get_misrepresentation_pipeline_note(context.is_audio_native),
            "disambiguation_context": disambiguation_context,
            "derived_tool_references": _CURRENT_DERIVED_REFERENCES.get() or REFERENCES_UNAVAILABLE_NOTE,
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
        """Build MetricScore with analysis details and per-dimension issue-flag sub-metrics."""
        dimensions = response.get("dimensions", {}) if isinstance(response, dict) else {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_FAITHFULNESS_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        analysis = {"dimensions": dimensions}
        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": analysis,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )
