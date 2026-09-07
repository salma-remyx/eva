"""STT reference-leakage diagnostic metric.

Debug metric for diagnosing model performance issues, not directly used in
final evaluation scores.

Adapted from "Towards Quantifying Benchmark Optimization in ASR Models"
(arXiv:2608.19936), which quantifies cases where an ASR model reproduces
reference transcript spans even though the audio cannot support them. EVA
knows the exact spoken ground truth (``tts_text_user``) and which turns were
truncated by a barge-in, so two of the paper's probe families port as
deterministic checks on ``intended_user_turns`` vs ``transcribed_user_turns``:

- ``directive_leakage`` (paper family: reference disagreement): bracketed
  non-speech directives in the intended text (e.g. ``[slow]``, ``[laughs]``)
  are never vocalized, so their words cannot be in the audio. Recovering one
  verbatim in the transcript means the span came from something other than
  the audio (e.g. STT context carryover), not from listening.
- ``truncation_recovery`` (paper family: masked-number recovery): when the
  assistant barges in on the user, the user's audio is cut mid-utterance, so
  the STT can only have heard a prefix. A transcript that still recovers
  every intended content word — especially long digit runs such as
  confirmation codes — is suspicious verbatim recovery.
"""

import re
from collections import Counter

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.log_processing import AnnotationLabel, normalize_for_comparison
from eva.utils.wer_normalization import normalize_text

_BRACKET_PATTERN = re.compile(r"\[([^\[\]]*)\]")

# Bracket contents that are framework bookkeeping (interruption labels, truncation
# markers) rather than user-simulator delivery directives. These are inserted by
# the log processor into both sides of the comparison and carry no probe signal.
# AnnotationLabel values include their brackets ("[pause]"), so strip them to
# compare against bracket contents.
_FRAMEWORK_BRACKETS = frozenset(str(label).strip("[] \t").lower() for label in AnnotationLabel) | {"truncated"}

_DIGIT_RUN_PATTERN = re.compile(r"\d+")

# Bracket contents shorter than this (after normalization) are ignored as directives —
# too short to match without false positives.
_MIN_DIRECTIVE_LEN = 2


def _is_probe_directive(bracket_content: str) -> bool:
    """Return True when a bracket span is a non-speech directive worth probing.

    Framework labels and truncation markers are excluded: the log processor may
    insert them into the transcript itself, which would mask real leakage.
    """
    return len(normalize_for_comparison(bracket_content)) >= _MIN_DIRECTIVE_LEN and (
        bracket_content.strip().lower() not in _FRAMEWORK_BRACKETS
    )


def _content_coverage(reference_clean: str, hypothesis_clean: str) -> float:
    """Return the fraction of reference content words present in the hypothesis.

    Multiset intersection over whitespace tokens, so repeated words are counted
    correctly and extra hypothesis words never inflate the score. Both inputs
    must already be WER-normalized (digits to words, casing folded, etc.).
    """
    ref_tokens = Counter(reference_clean.split())
    hyp_tokens = Counter(hypothesis_clean.split())
    total_ref = sum(ref_tokens.values())
    if total_ref == 0:
        return 0.0
    recovered = sum((ref_tokens & hyp_tokens).values())
    return recovered / total_ref


@register_metric
class STTReferenceLeakageMetric(CodeMetric):
    """Flags STT transcript spans that reproduce reference text the audio cannot support.

    Compares what the user simulator intended to say (``intended_user_turns``,
    which includes non-speech bracket directives and is truncated by barge-ins)
    against what STT transcribed (``transcribed_user_turns``). A turn is flagged
    when the transcript recovers content that was never in the audio:

    - directive leakage: a non-speech directive (e.g. ``[slow]``) appears verbatim
      in the transcript;
    - truncation recovery: on turns the assistant interrupted, every intended
      content word is nevertheless recovered.

    Score = flagged turns / evaluated turns. Lower is better (0.0 = no
    unsupported reference recovery observed).

    High rates do not prove benchmark optimization on their own — they surface
    turns for manual review of the STT pipeline (e.g. context carryover leaking
    reference text past a barge-in). This is a diagnostic metric used for
    diagnosing model performance issues. It is not directly used in final
    evaluation scores.
    """

    name = "stt_reference_leakage"
    version = "v0.1"
    description = "Debug metric: rate of STT turns recovering reference text the audio cannot support"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})
    higher_is_better = False

    def __init__(self, config: dict | None = None):
        """Initialize the metric with probe thresholds and language configuration."""
        super().__init__(config)
        self.language = self.config.get("language", "en")
        # Coverage at or above this on an interrupted turn counts as full recovery.
        self.full_recovery_threshold = float(self.config.get("full_recovery_threshold", 1.0))
        # Digit runs at least this long are highlighted in the evidence (codes, phone numbers).
        self.min_digit_run = int(self.config.get("min_digit_run", 4))
        # Turns with fewer normalized content words than this are not probed for
        # truncation recovery — trivially short utterances carry no signal.
        self.min_turn_words = int(self.config.get("min_turn_words", 3))

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute the reference-leakage rate over user turns."""
        try:
            common_turn_ids = sorted(context.intended_user_turns.keys() & context.transcribed_user_turns.keys())

            if not common_turn_ids:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user turns with both TTS text and transcript available",
                )

            directive_flagged: list[int] = []
            directive_probe_turns: list[int] = []
            truncation_eligible: list[int] = []
            truncation_flagged: list[int] = []
            per_turn_evidence: dict[int, dict] = {}

            for turn_id in common_turn_ids:
                intended_raw = context.intended_user_turns[turn_id]
                transcript_raw = context.transcribed_user_turns[turn_id]

                directives = [c for c in _BRACKET_PATTERN.findall(intended_raw) if _is_probe_directive(c)]
                if directives:
                    directive_probe_turns.append(turn_id)
                # Alphanumeric-normalized substring match, so punctuation and case
                # differences cannot hide a verbatim directive recovery.
                transcript_key = normalize_for_comparison(transcript_raw)
                leaked_directives = [c for c in directives if normalize_for_comparison(c) in transcript_key]
                if leaked_directives:
                    directive_flagged.append(turn_id)

                # Barge-in turns: the user's audio was cut, so only a prefix was audible.
                interrupted = turn_id in context.assistant_interrupted_turns
                recovered_everything = False
                coverage: float | None = None
                digit_runs: list[str] = []
                if interrupted:
                    intended_spoken = _BRACKET_PATTERN.sub("", intended_raw)
                    intended_clean = normalize_text(intended_spoken, self.language)
                    transcript_clean = normalize_text(_BRACKET_PATTERN.sub("", transcript_raw), self.language)
                    if len(intended_clean.split()) >= self.min_turn_words:
                        truncation_eligible.append(turn_id)
                        coverage = _content_coverage(intended_clean, transcript_clean)
                        digit_runs = self._long_digit_runs(intended_spoken)
                        recovered_everything = coverage >= self.full_recovery_threshold
                        if recovered_everything:
                            truncation_flagged.append(turn_id)

                if leaked_directives or recovered_everything:
                    per_turn_evidence[turn_id] = {
                        "leaked_directives": leaked_directives,
                        "interrupted": interrupted,
                        "coverage": round(coverage, 3) if coverage is not None else None,
                        "digit_runs": digit_runs,
                    }

            flagged_turn_ids = sorted(set(directive_flagged) | set(truncation_flagged))
            num_turns = len(common_turn_ids)
            rate = len(flagged_turn_ids) / num_turns
            # Nothing to probe: no directives in any intended turn and no barge-in turns.
            # Report as skipped rather than a misleading clean 0.0.
            skipped = not directive_probe_turns and not truncation_eligible

            sub_metrics = {
                "directive_leakage_rate": make_rate_sub_metric(
                    parent_name=self.name,
                    key="directive_leakage_rate",
                    numerator=len(directive_flagged),
                    denominator=num_turns,
                    details={"count": len(directive_flagged), "num_turns": num_turns, "turn_ids": directive_flagged},
                )
            }
            if truncation_eligible:
                sub_metrics["truncation_recovery_rate"] = make_rate_sub_metric(
                    parent_name=self.name,
                    key="truncation_recovery_rate",
                    numerator=len(truncation_flagged),
                    denominator=len(truncation_eligible),
                    details={
                        "count": len(truncation_flagged),
                        "num_eligible": len(truncation_eligible),
                        "turn_ids": truncation_flagged,
                    },
                )

            return MetricScore(
                name=self.name,
                score=None if skipped else round(rate, 3),
                normalized_score=None if skipped else round(rate, 3),
                details={
                    "language": self.language,
                    "full_recovery_threshold": self.full_recovery_threshold,
                    "num_turns": num_turns,
                    "num_flagged": len(flagged_turn_ids),
                    "flagged_turn_ids": flagged_turn_ids,
                    "num_directive_probe_turns": len(directive_probe_turns),
                    "num_truncation_eligible": len(truncation_eligible),
                    "per_turn_evidence": per_turn_evidence,
                },
                sub_metrics=sub_metrics,
                skipped=skipped,
            )

        except Exception as e:
            return self._handle_error(e, context)

    def _long_digit_runs(self, text: str) -> list[str]:
        """Return digit runs at least ``min_digit_run`` long (codes, phone numbers)."""
        return [run for run in _DIGIT_RUN_PATTERN.findall(text) if len(run) >= self.min_digit_run]
