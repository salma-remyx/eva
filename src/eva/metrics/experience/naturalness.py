"""Conversational naturalness metric using LLM-as-judge (whole conversation).

Adapted port (Mode 2) of the multidimensional naturalness protocol from
SPEARBench (arXiv:2607.05365). The paper evaluates streaming speech-to-speech
models on the conversational qualities that make them feel *natural* —
emotional tone, interpersonal stance, prosody, language/dialect consistency,
and relationship-aware appropriateness.

Adaptation choices (Mode 2 — auxiliary components swapped for target-native
equivalents, core mechanism preserved):

* Core mechanism kept: a single multidimensional judge that scores the whole
  conversation and emits one binary "issue" sub-metric per naturalness
  dimension (SPEARBench's per-dimension distributional scoring).
* Auxiliary substituted: SPEARBench judges raw audio with a Gemini audio
  judge over the Seamless Interaction corpus. EVA already has a mature
  transcript-based LLM-judge path for experience metrics (conciseness,
  conversation_progression), so this metric judges the conversation
  *transcript* instead. Prosody is therefore limited to text-judgeable
  markers (fillers, disfluencies, register, pacing cues visible in the
  transcript) — acoustic prosody (f0 contour, energy) is intentionally out
  of scope.
* Deliberately excluded dimensions: SPEARBench also covers response latency,
  overlap, and interruptions — those already live in EVA's
  ``turn_taking`` / ``response_speed`` metrics, and speech signal quality /
  ASR robustness live in ``speech_fidelity`` / ``stt_wer``. This metric fills
  the verified gap (the *naturalness* axes) rather than duplicating coverage.

Rating scale: 3 (natural), 2 (adequate), 1 (unnatural).
Normalized: 3->1.0, 2->0.5, 1->0.0.
"""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import get_assistant_turns_disclaimer, get_user_turns_disclaimer
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

# SPEARBench naturalness dimensions, expressed as binary issue flags. Each maps
# to a ``{key}_rate`` sub-metric (mean across records = fraction of
# conversations where this naturalness issue occurred; lower is better).
_NATURALNESS_DIMENSION_KEYS = (
    "emotional_naturalness",
    "interpersonal_stance",
    "prosody_appropriateness",
    "dialect_language_consistency",
    "relationship_appropriateness",
)


@register_metric
class NaturalnessJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based conversational naturalness metric (whole conversation).

    Evaluates whether the assistant behaves *naturally* in conversation across
    SPEARBench's naturalness dimensions, emitting one issue-rate sub-metric
    per dimension alongside an overall 1-3 naturalness rating.
    """

    name = "naturalness"
    version = "v0.1"
    description = "LLM judge evaluation of conversational naturalness (emotional tone, stance, prosody, dialect, relationship appropriateness)"
    category = "experience"
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "conversation_trace": transcript_text,
            "user_turns_disclaimer": get_user_turns_disclaimer(context.is_audio_native),
            "assistant_turns_disclaimer": get_assistant_turns_disclaimer(context.is_audio_native),
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
        """Build MetricScore with per-dimension naturalness issue-flag sub-metrics."""
        dimensions = response.get("dimensions", {}) or {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_NATURALNESS_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": response.get("explanation", ""),
                "dimensions": dimensions,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )
