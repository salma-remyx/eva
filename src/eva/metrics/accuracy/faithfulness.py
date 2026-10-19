"""Faithfulness metric using LLM-as-judge (whole conversation)."""

import json
import os
from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import (
    get_assistant_turns_disclaimer,
    get_misrepresentation_pipeline_note,
    get_user_turns_disclaimer,
)
from eva.metrics.reference_derivation import DERIVATION_MAX_TOKENS, derive_tool_references
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore
from eva.utils.llm_client import LLMClient

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

# Embedded in the judge prompt when reference derivation is disabled or not yet run,
# so the {derived_reference} template variable always renders.
_NO_DERIVED_REFERENCE_NOTE = "No derived reference was requested for this conversation."


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
    version = "v0.4"
    description = (
        "LLM judge evaluation of whether the assistant remains faithful to information, policies, and instructions"
    )
    category = "accuracy"
    default_model = "us.anthropic.claude-opus-4-6-v1"
    default_params = {"max_tokens": 100000}  # Drop the OpenAI-only flex tier inherited from TextJudgeMetric.
    rating_scale = (1, 3)

    def _reference_derivation_enabled(self) -> bool:
        """Opt-in flag: metric config ``derived_reference`` or the FAITHFULNESS_DERIVED_REFERENCE env var."""
        if "derived_reference" in self.config:
            return bool(self.config["derived_reference"])
        return os.environ.get("FAITHFULNESS_DERIVED_REFERENCE", "").strip().lower() in {"1", "true", "yes"}

    def _deriver_client(self) -> LLMClient:
        """The derivation stage's client.

        DeRef derives references with its own model, separate from the judge
        (the paper's deref_qwen3_8b ablation shows deriver choice moves
        results). Set ``deriver_model`` (or FAITHFULNESS_DERIVER_MODEL) to
        configure one; when unset, the aux calls run on the judge's client.
        """
        model = self.config.get("deriver_model") or os.environ.get("FAITHFULNESS_DERIVER_MODEL")
        if not model or model == self.llm_client.model:
            return self.llm_client
        params = {"max_tokens": DERIVATION_MAX_TOKENS}  # DeRef's derivation generation budget.
        params.update(self.config.get("deriver_params", {}))
        return LLMClient(model=model, params=params)

    async def compute(self, context: MetricContext) -> MetricScore:
        """Derive per-argument expected values (when enabled), then run the judge.

        One auxiliary LLM call per tool-call argument reads the question and the
        earlier calls with their returns — never the proposed value — and writes
        the expected value of that parameter (see eva.metrics.reference_derivation).
        The judge prompt embeds each answer as a block with DeRef's neutral
        header and no consumption instructions.
        """
        if self._reference_derivation_enabled():
            deriver = self._deriver_client()
            references = await derive_tool_references(deriver, self.prompt_manager, context, self.logger)
            for call in references.calls:
                # Account for each aux call's tokens alongside the judge's own usage.
                self._log_token_usage(
                    context,
                    deriver.model,
                    {"stage": "reference_derivation"},
                    call.prompt,
                    call.usage,
                    call.response,
                )
            context.derived_tool_reference = references.text
        return await super().compute(context)

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
            "derived_reference": context.derived_tool_reference or _NO_DERIVED_REFERENCE_NOTE,
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
