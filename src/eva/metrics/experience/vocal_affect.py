"""Vocal affect expressiveness metric using audio + LLM judge (conversation-level).

Adapted from the "expressiveness" / "vocal affect" evaluation dimension of
RW-Voice-EQ Bench (arXiv:2607.14846). That benchmark reports that many
speech-to-speech agents are *transcript-driven*: they transcribe the user's
audio and speak it back while ignoring the vocal affect (emotion, prosody,
emphasis) that distinguishes lively spoken language from flat read-aloud text.

This metric operationalizes that dimension for EVA. An audio judge listens to
the agent's spoken turns and rates how expressively and affect-appropriately it
delivers them, surfacing the transcript-driven failure mode as a sub-metric.
It is an experience metric restricted to audio-native pipelines (S2S / audio
LLM): cascade TTS has no input affect to mirror, and the transcript alone
cannot reveal vocal affect.
"""

from eva.metrics.base import AudioJudgeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics, normalize_rating, validate_rating
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.json_utils import extract_and_load_json

# Judge dimensions surfaced as binary sub-metrics (mean across records reads as
# the fraction of conversations where each issue was observed).
_VOCAL_AFFECT_DIMENSION_KEYS = (
    "flat_or_monotone",  # little pitch / energy / pacing variation
    "transcript_driven",  # sounds read aloud; ignores affective content
    "affect_mismatch",  # emotion incongruent with what is said
    "overacted",  # exaggerated / unnatural expressiveness
)


@register_metric
class VocalAffectMetric(AudioJudgeMetric):
    """Audio judge metric for the agent's use of vocal affect / expressiveness.

    Listens to the agent's spoken audio and rates how expressively and
    affect-appropriately it delivers its turns, rather than reading them flatly.

    Rating scale: 3 (expressive & affect-appropriate), 2 (adequate), 1 (flat / monotone)
    Normalized: 3 -> 1.0, 2 -> 0.5, 1 -> 0.0
    """

    name = "vocal_affect"
    version = "v0.1"
    description = "Audio judge evaluation of the agent's vocal expressiveness / use of vocal affect"
    category = "experience"
    rating_scale = (1, 3)
    # Vocal affect only applies to audio-native (end-to-end) agents.
    supported_pipeline_types = frozenset({PipelineType.S2S, PipelineType.AUDIO_LLM})

    async def compute(self, context: MetricContext) -> MetricScore:
        """Judge the agent's vocal affect from its spoken audio."""
        try:
            audio_segment = self.load_role_audio(context, "assistant")
            if audio_segment is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No assistant audio file available",
                )

            audio_b64 = self.encode_audio_segment(audio_segment)
            prompt = self.get_judge_prompt(expected_language=context.language_display_name)
            messages = self.create_audio_message(audio_b64, prompt)

            response_text, usage = await self.llm_client.generate_text(messages)
            self._log_token_usage(context, self.llm_client.model, self.llm_client.params, prompt, usage, response_text)

            if response_text is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No response from judge",
                    details={"judge_prompt": prompt},
                )

            parsed = extract_and_load_json(response_text)
            if not isinstance(parsed, dict):
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="Failed to parse judge response",
                    details={"judge_prompt": prompt, "judge_raw_response": response_text},
                )

            min_rating, max_rating = self.rating_scale
            rating = validate_rating(
                parsed.get("rating"),
                list(range(min_rating, max_rating + 1)),
                default=min_rating,
                record_id=context.record_id,
                metric_logger=self.logger,
            )
            normalized = normalize_rating(rating, min_rating, max_rating)

            dimensions = parsed.get("dimensions")
            if not isinstance(dimensions, dict):
                dimensions = {}
            sub_metrics = build_binary_flag_sub_metrics(
                parent_name=self.name,
                entries=dimensions,
                entry_keys=_VOCAL_AFFECT_DIMENSION_KEYS,
                flag_field="flagged",
                detail_fields=("evidence",),
            )

            return MetricScore(
                name=self.name,
                score=float(rating),
                normalized_score=normalized,
                details={
                    "rating": rating,
                    "explanation": parsed.get("explanation", ""),
                    "dimensions": dimensions,
                    "judge_prompt": prompt,
                    "judge_raw_response": response_text,
                },
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
