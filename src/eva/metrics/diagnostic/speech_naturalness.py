"""Multi-dimensional speech naturalness diagnostic metric (audio + LLM judge).

Adapted from "Beyond Naturalness: Probing Automated Text-To-Speech Evaluators
on Linguistically Grounded Dimensions" (arxiv:2608.09930). The paper's core
contribution is a methodology that deconstructs the single, opaque "naturalness"
score into a set of linguistically grounded perceptual dimensions, then probes an
Audio-LLM judge for each dimension independently — exposing that judges detect
errors selectively rather than across the board.

This metric ports that *mechanism* onto EVA's existing audio-judge contract: a
Gemini Audio-LLM judge rates the assistant's spoken audio
(``context.audio_assistant_path``) per turn along 10 linguistic dimensions, and
each dimension is surfaced as a binary rate sub-metric
(``flagged turns / rated turns``) — exactly as ``tts_fidelity`` surfaces
per-failure-mode rates via ``build_per_category_rate_sub_metrics``.

What is intentionally adapted / scoped out (Mode 2):
  - The paper's exact 10-dimension taxonomy (trained-linguist annotation schema)
    is not reproduced verbatim; the dimensions below are a target-native
    linguistic taxonomy (segmental, prosodic, fluency, voice-quality, affect,
    register) inspired by the same framing. The *mechanism* — independent
    per-dimension binary probing by an audio judge with per-dimension rates — is
    ported at full fidelity.
  - The paper's 860-utterance benchmark, MOS-predictor comparison, and the
    meta-evaluation ("does the judge agree with linguists?") are out of scope:
    those are downstream evaluation concerns, not a metric EVA can host. EVA's
    existing bootstrap-confidence-interval machinery already supplies the
    statistical-rigor layer the paper relies on.

Complementarity: ``tts_fidelity`` measures *word-level fidelity* to the intended
text (entity errors, truncation, hallucinations). This metric measures
*perceptual naturalness* — how human the speech sounds. The two are orthogonal.

The judge prompt lives at ``judge.speech_naturalness.user_prompt`` in
``configs/prompts/judge.yaml`` and is hashed into the metric's drift signature.
"""

from eva.metrics.base import MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric
from eva.metrics.utils import build_per_category_rate_sub_metrics
from eva.models.config import PipelineType
from eva.models.results import MetricScore

# Linguistically grounded naturalness dimensions. Each is a binary "is this kind
# of unnaturalness present in the turn?" probe; the per-dimension rate sub-metric
# is the fraction of rated turns where the judge flagged it.
_NATURALNESS_DIMENSIONS: tuple[str, ...] = (
    "mispronunciation",
    "unnatural_intonation",
    "inappropriate_stress",
    "unnatural_pacing",
    "pause_placement",
    "disfluency",
    "unnatural_voice_quality",
    "inexpressive_affect",
    "phrasing_boundary_error",
    "register_tone_mismatch",
)


@register_metric
class SpeechNaturalnessMetric(SpeechFidelityBaseMetric):
    """Audio-based speech naturalness metric probing 10 linguistic dimensions.

    Uses a Gemini Audio-LLM judge to rate each assistant turn as natural (1) or
    unnatural (0) and tag which of the 10 perceptual dimensions are defective.
    Surfaces one binary rate sub-metric per dimension
    (``speech_naturalness.<dimension>_rate``) so the *distribution* of naturalness
    failures is visible — not just a single collapsed score.

    Complementary to ``tts_fidelity`` (word fidelity), not duplicative.
    """

    name = "speech_naturalness"
    version = "v0.1"
    description = "Diagnostic metric: 10-dimension linguistic naturalness probing of agent audio"
    category = "diagnostic"
    role = "assistant"
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE, PipelineType.AUDIO_LLM})
    rating_scale = (0, 1)

    def build_sub_metrics(
        self,
        context: MetricContext,
        per_turn_ratings: dict[int, int | None],
        per_turn_failure_modes: dict[int, list[str]],
    ) -> dict[str, MetricScore] | None:
        """Surface one rate sub-metric per naturalness dimension.

        rate = turns flagged for that dimension / rated turns.
        """
        rated_turn_ids = [tid for tid, r in per_turn_ratings.items() if r is not None]
        return (
            build_per_category_rate_sub_metrics(
                parent_name=self.name,
                categories=_NATURALNESS_DIMENSIONS,
                rated_turn_ids=rated_turn_ids,
                per_turn_categories=per_turn_failure_modes,
            )
            or None
        )
