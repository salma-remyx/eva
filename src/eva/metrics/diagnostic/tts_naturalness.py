"""TTS naturalness diagnostic metric using audio + LLM judge (Gemini).

Adapted from "Beyond Naturalness: Probing Automated Text-To-Speech Evaluators
on Linguistically Grounded Dimensions" (arXiv:2608.09930). That work argues a
scalar naturalness / MOS score collapses several independent linguistic
phenomena, and deconstructs "naturalness" into 10 linguistically grounded
perceptual dimensions (word, prosodic, and paralinguistic levels). It further
shows MOS predictors collapse onto acoustic signal quality and Audio-LLM judges
miss structured linguistic errors.

This metric operationalises that schema as a Gemini AudioJudge over the
assistant audio, mirroring ``tts_fidelity``: each assistant turn is rated for
overall naturalness and any of the 10 dimensions with an audible deficiency is
tagged. Each dimension becomes a per-dimension rate sub-metric, so specific
linguistic error types stay visible instead of being flattened into one number.

It is *complementary* to ``tts_fidelity``: fidelity asks "did it say the
intended words?", naturalness asks "did those words sound human?".

Mode 2 (adapted port): the paper's 10-dimension rubric and dimension-level
reporting are kept at full fidelity; the human-linguist rater and benchmark
suite are replaced by the repo's existing Gemini AudioJudge + metric-registry
plumbing, and the rubric prompt lives in a dedicated file rather than the
shared ``judge.yaml``.
"""

from eva.metrics.base import MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.speech_fidelity_base import SpeechFidelityBaseMetric
from eva.metrics.utils import build_per_category_rate_sub_metrics
from eva.metrics.versioning import _CURRENT_PROMPT_HASH, hash_prompt_template
from eva.models.config import PipelineType
from eva.models.results import MetricScore

# The 10 linguistically grounded perceptual dimensions of TTS naturalness
# (arXiv:2608.09930, annotation schema). Each becomes a per-dimension rate
# sub-metric: the fraction of rated assistant turns flagged for that dimension.
_NATURALNESS_DIMENSIONS = (
    # Word level
    "phonetic_accuracy",  # sounds fall within acceptable realisation for lexical targets
    "lexical_stress",  # primary stress on the correct syllable; unstressed vowel reduction
    # Prosodic level
    "intonation",  # pitch contour encodes utterance type and discourse structure
    "prosodic_stress",  # prominence driven by information structure
    "prosodic_boundary",  # phrasing / chunking aligned with syntactic structure
    "speech_rate",  # overall pace of delivery
    # Paralinguistic level
    "emotional_appropriateness",  # emotional tone matches the affective content
    "expressiveness",  # natural variation in delivery style and vocal energy
    "speaker_identity",  # stability of perceived vocal characteristics across the utterance
    "human_plausibility",  # vocal qualities within the range of human production
)

# The rubric prompt lives in its own namespace (configs/prompts/naturalness_rubric.yaml)
# rather than the shared judge.yaml, so the prompt stays self-contained with this metric
# and the change requires no edit to the existing judge prompt file.
_PROMPT_NAMESPACE = "naturalness_rubric"


@register_metric
class TTSNaturalnessMetric(SpeechFidelityBaseMetric):
    """Audio-based TTS naturalness metric for the agent using Gemini.

    Evaluates whether the agent's spoken audio sounds natural across 10
    linguistically grounded perceptual dimensions (phonetic, prosodic, and
    paralinguistic), instead of collapsing to a single MOS-style score.

    Rating scale: 0 (unnatural) or 1 (natural) per assistant turn. Each
    dimension with an audible deficiency is tagged and surfaced as a
    per-dimension rate sub-metric, making specific linguistic error types
    visible — the core recommendation of arXiv:2608.09930, which shows MOS
    predictors and Audio-LLM judges miss structured linguistic errors.
    """

    name = "tts_naturalness"
    version = "v0.1"
    description = "Diagnostic metric: TTS naturalness across 10 linguistic dimensions"
    category = "diagnostic"
    role = "assistant"
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE, PipelineType.AUDIO_LLM})
    rating_scale = (0, 1)

    def get_judge_prompt(self, prompt_key: str = "user_prompt", **variables) -> str:
        """Resolve the judge prompt from the naturalness rubric namespace.

        Overrides the base ``judge.{name}.{key}`` path so the 10-dimension
        rubric can live in a dedicated prompt file instead of the shared
        ``judge.yaml``, keeping this metric's prompt self-contained.
        """
        prompt_path = f"{_PROMPT_NAMESPACE}.{self.name}.{prompt_key}"
        _CURRENT_PROMPT_HASH.set(hash_prompt_template(self.prompt_manager.get_template(prompt_path)))
        return self.prompt_manager.get_prompt(prompt_path, **variables)

    def build_sub_metrics(
        self,
        context: MetricContext,
        per_turn_ratings: dict[int, int | None],
        per_turn_failure_modes: dict[int, list[str]],
    ) -> dict[str, MetricScore] | None:
        """Surface one sub-metric per naturalness dimension: rate = flagged turns / rated turns."""
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
