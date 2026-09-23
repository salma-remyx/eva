"""Semantic-tier STT transcription quality metric.

Diagnostic metric separating character-level transcription drift from actual
meaning change, complementing ``stt_wer`` (word level) and
``transcription_accuracy_key_entities`` (entity level).

Adapted from "Rethinking Human-Aligned Evaluation: An Analysis of Semantic
Metrics Beyond WER" (arXiv:2609.21663), which finds that WER agrees least with
human judgment of ASR quality, recommends character error rate (CER) as the
primary low-cost error rate — for English as well as morphosyllabic writing
systems — and SemDist as a complementary semantic measure. The paper's
sentence-embedding SemDist is substituted here with a stopword-filtered
content-token cosine similarity so the metric stays deterministic and offline;
content-word substitutions, deletions and insertions still move the score,
while pure function-word drift ("a" vs "the", "I'd" vs "I would") does not.
"""

import math
import re
from collections import Counter

import jiwer

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.wer_normalization import normalize_text

_BRACKET_PATTERN = re.compile(r"\[.*?\]")

_TOKEN_PATTERN = re.compile(r"[^\W_]+")

# English function words ignored by the semantic comparison: they carry syntax,
# not meaning, so swapping one is lexical drift rather than a semantic error —
# exactly the deviation class WER charges at full price. Negations are
# deliberately NOT listed: dropping a "not" reverses meaning and must register
# as a semantic error. Other languages currently fall back to comparing all
# tokens (conservative: function-word drift then counts as semantic).
_ENGLISH_STOPWORDS = frozenset(
    """
    a an the this that these those
    am is are was were be been being
    do does did doing have has had having
    will would shall should can could may might must
    i you he she it we they me him her us them
    my your his its our their mine yours
    and but or so yet
    as at by for from in into of to with without
    if then than because while when where
    which who whom whose what how
    very just also too there here again please
    """.split()
)


def _content_tokens(text: str, language: str) -> list[str]:
    """Lowercase word tokens of ``text`` with function words removed.

    Only English has a curated stopword list; other languages keep every token.
    """
    tokens = _TOKEN_PATTERN.findall(text)
    if language != "en":
        return tokens
    return [token for token in tokens if token not in _ENGLISH_STOPWORDS]


def _semantic_similarity(reference: str, hypothesis: str, language: str) -> float:
    """Cosine similarity between content-token frequency vectors (1.0 = same meaning).

    Order-insensitive and tolerant of repeated function-word drift, unlike WER.
    Two texts with no content tokens at all are treated as identical (nothing
    meaning-bearing differs); exactly one empty side scores 0.0.
    """
    reference_counts = Counter(_content_tokens(reference, language))
    hypothesis_counts = Counter(_content_tokens(hypothesis, language))

    if not reference_counts and not hypothesis_counts:
        return 1.0
    if not reference_counts or not hypothesis_counts:
        return 0.0

    dot_product = sum(count * hypothesis_counts[token] for token, count in reference_counts.items())
    reference_norm = math.sqrt(sum(count**2 for count in reference_counts.values()))
    hypothesis_norm = math.sqrt(sum(count**2 for count in hypothesis_counts.values()))
    return dot_product / (reference_norm * hypothesis_norm)


@register_metric
class TranscriptionSemanticAccuracyMetric(CodeMetric):
    """Meaning preservation of STT transcription.

    Measures how well the meaning of what the user simulator intended to say
    (tts_text_user) survived transcription (transcript_user), reporting two
    signals per turn:

    - CER: character error rate over the normalized texts, computed for every
      language (``stt_wer`` only uses CER for whitespace-less writing systems).
      Cheap, interpretable, and closer to human judgment than WER.
    - SemDist: 1 - cosine similarity of content-token frequency vectors, the
      deterministic stand-in for the paper's embedding-based SemDist. High CER
      with low SemDist means lexical drift that preserved meaning; both high
      means the transcript changed what was said.

    The parent score is the mean per-turn SemDist (lower is better), with
    semantic accuracy (1 - SemDist) as the normalized score. CER is surfaced as
    a sub-metric and in ``details``.

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "transcription_semantic_accuracy"
    version = "v0.1"
    description = "Debug metric: STT meaning preservation via CER and content-token semantic distance"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})

    def __init__(self, config: dict | None = None):
        """Initialize the metric with language configuration."""
        super().__init__(config)
        # Get language from config (default: "en")
        self.language = self.config.get("language", "en")

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute CER and content-token semantic distance for user turns."""
        try:
            # Collect reference/hypothesis pairs for turns present in both dicts
            common_turn_ids = sorted(context.intended_user_turns.keys() & context.transcribed_user_turns.keys())

            evaluated_turn_ids = []
            references = []
            hypotheses = []

            for turn_id in common_turn_ids:
                ref = _BRACKET_PATTERN.sub("", context.intended_user_turns[turn_id]).strip()
                hyp = _BRACKET_PATTERN.sub("", context.transcribed_user_turns[turn_id]).strip()
                if ref and hyp:
                    evaluated_turn_ids.append(turn_id)
                    references.append(ref)
                    hypotheses.append(hyp)

            if not references:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No user turns with both TTS text and transcript available",
                )

            references_clean = [normalize_text(r, self.language) for r in references]
            hypotheses_clean = [normalize_text(h, self.language) for h in hypotheses]

            # Character error rate over the whole conversation (jiwer aggregates
            # across the reference corpus), plus its component counts.
            cer = jiwer.cer(references_clean, hypotheses_clean)
            char_output = jiwer.process_characters(references_clean, hypotheses_clean)

            per_turn_cer: dict[int, float] = {}
            per_turn_semdist: dict[int, float] = {}

            for turn_id, ref_clean, hyp_clean in zip(evaluated_turn_ids, references_clean, hypotheses_clean):
                per_turn_cer[turn_id] = round(jiwer.cer(ref_clean, hyp_clean), 3)
                per_turn_semdist[turn_id] = round(1.0 - _semantic_similarity(ref_clean, hyp_clean, self.language), 3)

            mean_semdist = sum(per_turn_semdist.values()) / len(per_turn_semdist)
            semantic_accuracy = 1.0 - mean_semdist

            reference_characters = len("".join(references_clean))
            char_errors = char_output.substitutions + char_output.deletions + char_output.insertions
            sub_metrics = {
                "character_error_rate": make_rate_sub_metric(
                    parent_name=self.name,
                    key="character_error_rate",
                    numerator=char_errors,
                    denominator=reference_characters,
                    details={"count": char_errors, "reference_characters": reference_characters},
                )
            }

            return MetricScore(
                name=self.name,
                score=round(mean_semdist, 3),
                normalized_score=round(semantic_accuracy, 3),
                details={
                    "cer": round(cer, 3),
                    "semdist": round(mean_semdist, 3),
                    "semantic_accuracy": round(semantic_accuracy, 3),
                    "language": self.language,
                    "num_turns": len(references),
                    "per_turn_cer": per_turn_cer,
                    "per_turn_semdist": per_turn_semdist,
                    "total_substitutions": char_output.substitutions,
                    "total_deletions": char_output.deletions,
                    "total_insertions": char_output.insertions,
                    "reference_characters": reference_characters,
                },
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
