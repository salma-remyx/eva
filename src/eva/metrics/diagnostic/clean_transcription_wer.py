"""Disfluency-aware clean-transcription Word Error Rate metric.

Diagnostic metric adapted from the AgenticSR "audio-to-clean-text" formulation
(``AgenticASR: Refining Speech Recognition in Real-World Scenarios via an
Agentic Approach``, arXiv:2607.28175). Verbatim ASR transcripts retain fillers,
repetitions, and false starts that inflate Word Error Rate without reflecting a
real recognition error -- they obscure the speaker's final intent rather than
mis-recognizing it. This metric applies an intent-preserving disfluency cleaner
to *both* the reference and the hypothesis before the existing WER
normalization pipeline, so the error rate measures recognition accuracy against
the speaker's final intent instead of verbatim disfluency.

Adaptation note (Mode 2). The paper cleans transcripts with a *learned*
text->text "Refiner" model. This metric substitutes that learned component with
``clean_disfluencies`` -- a parameter-free, deterministic proxy (hesitation
filler removal + immediate-repetition collapse). That keeps the core mechanism
(clean-then-score) reproducible and side-effect-free for a diagnostic metric.
Resolution of *self-corrections* (the paper's third disfluency class) is
intentionally out of scope: a naive deterministic rule for it distorts meaning
(the abandoned fragment is not always a whole-utterance restart), so it is left
to the learned Refiner it would take a separate port to host.

The existing deterministic ``stt_wer`` metric is preserved unchanged as the
verbatim baseline; this metric reports the clean error rate alongside a
``verbatim_wer`` so the disfluency-inflation gap is visible.
"""

import re
from typing import Any

import jiwer

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import (
    aggregate_wer_errors,
    extract_wer_errors,
    make_rate_sub_metric,
    reverse_word_error_rate,
)
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.wer_normalization import normalize_text

_BRACKET_PATTERN = re.compile(r"\[.*?\]")

# Languages without word-level whitespace segmentation -- CER is the appropriate
# measure instead of WER (mirrors ``stt_wer``).
_CER_LANGUAGES = frozenset({"ja", "zh", "ko"})

# Pure hesitation / vocalized fillers with no semantic content. Deliberately
# conservative: discourse words (like, so, well, actually, right, okay) are
# excluded because they are context-dependent and dropping them can change
# meaning. Removed as standalone tokens only.
_EN_FILLERS = frozenset(
    {
        "um",
        "umm",
        "uh",
        "uhh",
        "uhm",
        "er",
        "err",
        "ah",
        "ahh",
        "mm",
        "mmm",
        "hmm",
        "hmmm",
        "hm",
        "ugh",
        "uh-huh",
        "uhhuh",
        "mm-hmm",
        "mmhmm",
        "mhm",
        "huh",
    }
)

# Match standalone filler tokens (case-insensitive). Longest-first so
# multi-word tokens like "uh-huh" win over their substrings.
_FILLER_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(_EN_FILLERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Collapse immediate token runs caused by stuttering / false-start repetition:
# "I I want" -> "I want", "the the the" -> "the". Applied to 2+ identical
# consecutive word-tokens. Symmetric on reference/hypothesis, so it cannot bias
# WER even when it collapses legitimate emphatic doubles.
_REPETITION_PATTERN = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)

# Trailing cutoff artifact left by interrupted speech ("I want to--", "go ...").
# Stripping a dangling dash/ellipsis tidies false starts without dropping words.
_TRAILING_CUTOFF_PATTERN = re.compile(r"[\-–—]{1,}\s*$|\.{2,}\s*$")

_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_disfluencies(text: str, language: str = "en") -> str:
    """Remove conversational disfluencies while preserving final intent.

    Parameter-free proxy for AgenticASR's learned Refiner. Targets the
    disfluency classes a deterministic rule can resolve correctly -- hesitation
    fillers and immediate repetitions -- leaving self-correction resolution to
    the learned component (see module docstring). Applied symmetrically to the
    reference and hypothesis, so it cannot bias the resulting error rate.

    Args:
        text: Raw transcript or reference text.
        language: Language code (e.g. ``"en"``, ``"fr-FR"``). Hesitation-filler
            removal is English-only; repetition collapse and cutoff tidying
            apply to every language.

    Returns:
        Text with targeted disfluencies removed.
    """
    if not text:
        return text

    cleaned = _TRAILING_CUTOFF_PATTERN.sub("", text)
    # Collapse repetitions first so the filler pass sees stable surrounding tokens.
    cleaned = _REPETITION_PATTERN.sub(r"\1", cleaned)
    if language.split("-")[0].lower() == "en":
        cleaned = _FILLER_PATTERN.sub("", cleaned)
    # A second pass catches repetitions exposed once fillers between them drop.
    cleaned = _REPETITION_PATTERN.sub(r"\1", cleaned)
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def _build_wer_component_sub_metrics(
    parent_name: str,
    substitutions: int,
    deletions: int,
    insertions: int,
    reference_words: int,
) -> dict[str, MetricScore]:
    """Build sub-metrics for substitution, deletion, and insertion rates.

    Mirrors the helper in ``stt_wer``: each rate = component count / reference
    word count. Returns an empty dict when there are no reference words.
    """
    if reference_words <= 0:
        return {}

    components = {
        "substitution_rate": substitutions,
        "deletion_rate": deletions,
        "insertion_rate": insertions,
    }
    return {
        key: make_rate_sub_metric(
            parent_name=parent_name,
            key=key,
            numerator=count,
            denominator=reference_words,
            details={"count": count, "reference_words": reference_words},
        )
        for key, count in components.items()
    }


@register_metric
class CleanTranscriptionWERMetric(CodeMetric):
    """Disfluency-aware Speech-to-Text Word Error Rate metric.

    Like ``stt_wer`` but cleans disfluencies (fillers, repetitions) from both
    the intended and transcribed user turns before scoring, so the error rate
    reflects recognition accuracy against the speaker's final intent rather than
    verbatim disfluency. The clean WER is the headline score; ``verbatim_wer``
    (without cleaning) is reported in ``details`` to surface the inflation gap.

    Lower WER is better; converted to accuracy via ``reverse_word_error_rate``.

    This is a diagnostic metric for diagnosing model performance issues. It is
    not directly used in final evaluation scores.
    """

    name = "clean_transcription_wer"
    version = "v0.1"
    description = "Debug metric: STT accuracy as WER on disfluency-cleaned transcripts"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the metric with language configuration."""
        super().__init__(config)
        self.language = self.config.get("language", "en")

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute disfluency-cleaned STT WER for user turns."""
        try:
            common_turn_ids = sorted(context.intended_user_turns.keys() & context.transcribed_user_turns.keys())

            evaluated_turn_ids: list[int] = []
            references: list[str] = []
            hypotheses: list[str] = []
            verbatim_references: list[str] = []
            verbatim_hypotheses: list[str] = []

            for turn_id in common_turn_ids:
                ref = _BRACKET_PATTERN.sub("", context.intended_user_turns[turn_id]).strip()
                hyp = _BRACKET_PATTERN.sub("", context.transcribed_user_turns[turn_id]).strip()
                if ref and hyp:
                    evaluated_turn_ids.append(turn_id)
                    references.append(clean_disfluencies(ref, self.language))
                    hypotheses.append(clean_disfluencies(hyp, self.language))
                    verbatim_references.append(ref)
                    verbatim_hypotheses.append(hyp)

            if not references:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user turns with both TTS text and transcript available",
                )

            references_clean = [normalize_text(r, self.language) for r in references]
            hypotheses_clean = [normalize_text(h, self.language) for h in hypotheses]
            verbatim_references_clean = [normalize_text(r, self.language) for r in verbatim_references]
            verbatim_hypotheses_clean = [normalize_text(h, self.language) for h in verbatim_hypotheses]

            use_cer = self.language in _CER_LANGUAGES
            rate_fn = jiwer.cer if use_cer else jiwer.wer
            process_fn = jiwer.process_characters if use_cer else jiwer.process_words

            error_rate = rate_fn(references_clean, hypotheses_clean)
            output = process_fn(references_clean, hypotheses_clean)
            verbatim_error_rate = rate_fn(verbatim_references_clean, verbatim_hypotheses_clean)

            accuracy = reverse_word_error_rate(error_rate)
            rate_key = "cer" if use_cer else "wer"

            per_turn_wer: dict[int, float] = {}
            per_turn_errors: dict[int, dict[str, list[Any]]] = {}
            for turn_id, ref_clean, hyp_clean in zip(evaluated_turn_ids, references_clean, hypotheses_clean):
                turn_rate = rate_fn(ref_clean, hyp_clean)
                turn_output = process_fn(ref_clean, hyp_clean)
                per_turn_wer[turn_id] = round(turn_rate, 3)
                per_turn_errors[turn_id] = extract_wer_errors(turn_output)

            error_summary = aggregate_wer_errors(output)

            reference_unit_count = (
                len("".join(references_clean)) if use_cer else sum(len(r.split()) for r in references_clean)
            )
            sub_metrics = _build_wer_component_sub_metrics(
                parent_name=self.name,
                substitutions=output.substitutions,
                deletions=output.deletions,
                insertions=output.insertions,
                reference_words=reference_unit_count,
            )

            return MetricScore(
                name=self.name,
                score=round(error_rate, 3),
                normalized_score=round(accuracy, 3),
                details={
                    rate_key: round(error_rate, 3),
                    f"verbatim_{rate_key}": round(verbatim_error_rate, 3),
                    "disfluency_inflation": round(max(0.0, verbatim_error_rate - error_rate), 3),
                    "accuracy": round(accuracy, 3),
                    "language": self.language,
                    "use_cer": use_cer,
                    "num_turns": len(references),
                    "per_turn_wer": per_turn_wer,
                    "per_turn_errors": per_turn_errors,
                    "error_summary": error_summary,
                    "total_substitutions": output.substitutions,
                    "total_deletions": output.deletions,
                    "total_insertions": output.insertions,
                    "reference_words": reference_unit_count,
                },
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
