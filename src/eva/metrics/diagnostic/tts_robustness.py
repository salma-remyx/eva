"""TTS complex-text robustness diagnostic (assistant side).

Deterministic counterpart to ``stt_wer``: where ``stt_wer`` compares what the
user simulator intended to say with the agent's transcript of it, this metric
compares the text the agent intended to speak (``intended_assistant_turns``)
with what came back through the TTS -> STT round trip
(``transcribed_assistant_turns``). The gap between the two is speech-synthesis
infidelity that text-only metrics never see.

Adapted from "Complex-Text Robustness Evaluation and Failure Diagnosis for
Low-Resource Multilingual Text-to-Speech" (arXiv:2609.11545). The paper's
diagnostic axes are kept; its trained language-identification model is
substituted with a parameter-free script + function-word heuristic so the
metric stays fully deterministic:

- Content consistency — character error rate (CER) between intended and
  transcribed assistant text, reported separately for complex and simple
  turns (the paper's CER on complex spans).
- Language consistency — per-turn language check on the transcript (the
  paper's LID accuracy), surfaced as a wrong-language rate.
- Text Risk Score (TRS) — a lightweight pre-synthesis risk estimate computed
  from interpretable features of the intended text alone (numbers, dates,
  named entities, code-switching, length, symbol clutter). Like the paper's
  TRS it needs no annotation and no model training; here it drives the
  complex/simple turn split.

This is a diagnostic metric used for diagnosing model performance issues.
It is not directly used in final evaluation scores.
"""

import re
from typing import Any

import jiwer

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.config import PipelineType
from eva.models.results import MetricScore
from eva.utils.wer_normalization import normalize_text

_BRACKET_PATTERN = re.compile(r"\[.*?\]")

# Text Risk Score: binary feature triggers with hand-set weights. TRS is the
# capped sum of triggered weights; per-turn feature flags are reported so the
# split can be re-weighted offline without re-running conversations.
_RISK_WEIGHTS: dict[str, float] = {
    "numbers": 0.25,  # digit groups ("402", "919-696-3901")
    "dates": 0.2,  # numeric dates and times ("12/09/2026", "18:45")
    "named_entities": 0.2,  # acronyms ("DJLPO") and mid-sentence proper nouns
    "code_switching": 0.25,  # letters from more than one script family
    "long_text": 0.2,  # far beyond a typical spoken utterance
    "symbols": 0.15,  # symbol clutter ("&", "%", "(", "/", "=")
}
_NUMBER_PATTERN = re.compile(r"\d+")
_DATE_PATTERN = re.compile(r"\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b|\b\d{1,2}:\d{2}\b")
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")
_CAPITALIZED_PATTERN = re.compile(r"\b[A-Z][a-zà-ÿ]+\b")
_SYMBOL_PATTERN = re.compile(r"[()[\]{}&%$#@*+=~<>|«»…—_/\\]")
_LONG_TURN_CHARS = 300

# Unicode script blocks used for code-switching detection and the
# language-consistency check (family name, block start, block end).
_SCRIPT_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("latin", 0x41, 0x5A),
    ("latin", 0x61, 0x7A),
    ("latin", 0xC0, 0x24F),
    ("cyrillic", 0x400, 0x4FF),
    ("han", 0x3400, 0x4DBF),
    ("han", 0x4E00, 0x9FFF),
    ("kana", 0x3040, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
    ("thai", 0xE00, 0xE7F),
    ("arabic", 0x600, 0x6FF),
    ("devanagari", 0x900, 0x97F),
)

# Per-language consistency profiles: expected script families, frequent
# function words, and signature diacritics. The lexical layer catches
# same-script drift (e.g. English output during a French run) that script
# detection alone cannot.
_LANGUAGE_PROFILES: dict[str, dict[str, Any]] = {
    "en": {"scripts": {"latin"}, "markers": {"the", "is", "are", "you", "your", "we", "will", "to", "for", "of"}},
    "fr": {
        "scripts": {"latin"},
        "markers": {"le", "la", "les", "des", "une", "est", "pas", "je", "vous", "nous"},
        "diacritics": set("àâäéèêëîïôöùûüç"),
    },
    "de": {
        "scripts": {"latin"},
        "markers": {"der", "die", "das", "und", "ist", "nicht", "ich", "sie", "wir"},
        "diacritics": set("äöüß"),
    },
    "es": {
        "scripts": {"latin"},
        "markers": {"el", "los", "las", "que", "no", "es", "usted", "para", "con"},
        "diacritics": set("áéíóúñü"),
    },
    "ja": {"scripts": {"kana", "han"}},
    "zh": {"scripts": {"han"}},
    "ko": {"scripts": {"hangul"}},
    "ru": {"scripts": {"cyrillic"}},
}
_SCRIPT_MATCH_THRESHOLD = 0.5
_MIN_WORDS_FOR_LEXICAL_CHECK = 8


def _script_of(char: str) -> str | None:
    """Return the script family of a character, or None if unrecognized."""
    code = ord(char)
    for family, start, end in _SCRIPT_BLOCKS:
        if start <= code <= end:
            return family
    return None


def _has_code_switching(text: str) -> bool:
    """Check whether at least two script families contribute >= 2 letters each."""
    counts: dict[str, int] = {}
    for char in text:
        if char.isalpha():
            family = _script_of(char)
            if family:
                counts[family] = counts.get(family, 0) + 1
    return sum(1 for count in counts.values() if count >= 2) > 1


def _named_entity_count(text: str) -> int:
    """Count acronym and proper-noun candidates, ignoring the turn-opening word."""
    # The first capitalized word is usually just the turn opening ("Bien sûr...").
    return len(_ACRONYM_PATTERN.findall(text)) + max(0, len(_CAPITALIZED_PATTERN.findall(text)) - 1)


def text_risk_score(text: str) -> tuple[float, dict[str, bool]]:
    """Compute the Text Risk Score of an intended-to-synthesize turn.

    The score estimates how likely the turn is to trip up TTS before any audio
    exists, from interpretable text features only (paper's TRS: no annotation,
    no training). Returns the capped weighted sum and the per-feature triggers.
    """
    triggered = {
        "numbers": bool(_NUMBER_PATTERN.search(text)),
        "dates": bool(_DATE_PATTERN.search(text)),
        "named_entities": _named_entity_count(text) > 0,
        "code_switching": _has_code_switching(text),
        "long_text": len(text) > _LONG_TURN_CHARS,
        "symbols": len(_SYMBOL_PATTERN.findall(text)) >= 2,
    }
    score = min(1.0, sum(_RISK_WEIGHTS[feature] for feature, on in triggered.items() if on))
    return score, triggered


def detect_wrong_language(text: str, language: str) -> str | None:
    """Check a transcript turn against the expected language profile.

    Returns a reason string when the turn looks like it left the target
    language ("wrong_script" when most letters are in another script,
    "no_target_language_markers" when a long turn contains none of the
    language's function words or signature diacritics), or None when the turn
    is consistent — or the language has no profile, in which case the check is
    skipped by the caller.
    """
    profile = _LANGUAGE_PROFILES.get(language)
    if profile is None:
        return None
    letters = [char for char in text.lower() if char.isalpha()]
    if not letters:
        return None
    expected_scripts = profile["scripts"]
    in_script = sum(1 for char in letters if _script_of(char) in expected_scripts)
    if in_script / len(letters) < _SCRIPT_MATCH_THRESHOLD:
        return "wrong_script"
    markers = profile.get("markers", set())
    if markers:
        words = set(re.findall(r"[^\W\d_]+", text.lower()))
        has_diacritics = bool(set(text.lower()) & profile.get("diacritics", set()))
        if len(words) >= _MIN_WORDS_FOR_LEXICAL_CHECK and not words & markers and not has_diacritics:
            return "no_target_language_markers"
    return None


@register_metric
class TTSRobustnessMetric(CodeMetric):
    """TTS complex-text robustness metric.

    Measures how faithfully the agent's spoken output survives the TTS -> STT
    round trip on assistant turns, and whether complex text (numbers, dates,
    named entities, code-switching) degrades more than simple text.

    Lower CER is better. Converted to accuracy in normalized_score.

    This is a diagnostic metric used for diagnosing model performance issues.
    It is not directly used in final evaluation scores.
    """

    name = "tts_robustness"
    version = "v0.1"
    description = "Debug metric: TTS complex-text robustness via character error rate, language consistency, text risk"
    category = "diagnostic"
    exclude_from_pass_at_k = True
    supported_pipeline_types = frozenset({PipelineType.CASCADE})
    higher_is_better = False  # Score is a character error rate — lower is better.

    def __init__(self, config: dict | None = None):
        """Initialize the metric with language and risk-threshold configuration."""
        super().__init__(config)
        self.language = self.config.get("language", "en")
        self.risk_threshold = float(self.config.get("risk_threshold", 0.5))

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute TTS robustness over assistant turns."""
        try:
            common_turn_ids = sorted(
                context.intended_assistant_turns.keys() & context.transcribed_assistant_turns.keys()
            )

            pairs: list[tuple[int, str, str]] = []
            for turn_id in common_turn_ids:
                ref = _BRACKET_PATTERN.sub("", context.intended_assistant_turns[turn_id]).strip()
                hyp = _BRACKET_PATTERN.sub("", context.transcribed_assistant_turns[turn_id]).strip()
                if ref and hyp:
                    pairs.append((turn_id, ref, hyp))

            if not pairs:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No assistant turns with both intended text and transcript available",
                )

            base_language = self.language.split("-")[0].lower()

            per_turn_cer: dict[int, float | None] = {}
            per_turn_risk: dict[int, float] = {}
            per_turn_features: dict[int, dict[str, bool]] = {}
            per_turn_language_flag: dict[int, str] = {}
            complex_ids: list[int] = []
            simple_ids: list[int] = []
            language_evaluated_ids: list[int] = []
            complex_edits = complex_chars = simple_edits = simple_chars = total_edits = total_chars = 0
            substitution_counts: dict[str, int] = {}

            for turn_id, ref, hyp in pairs:
                risk, features = text_risk_score(ref)
                per_turn_risk[turn_id] = round(risk, 3)
                per_turn_features[turn_id] = features
                if risk >= self.risk_threshold:
                    complex_ids.append(turn_id)
                else:
                    simple_ids.append(turn_id)

                if _LANGUAGE_PROFILES.get(base_language) and any(char.isalpha() for char in hyp):
                    language_evaluated_ids.append(turn_id)
                    reason = detect_wrong_language(hyp, base_language)
                    if reason:
                        per_turn_language_flag[turn_id] = reason

                ref_clean = normalize_text(ref, self.language)
                hyp_clean = normalize_text(hyp, self.language)
                if not ref_clean:
                    per_turn_cer[turn_id] = None
                    continue

                output = jiwer.process_characters(ref_clean, hyp_clean)
                edits = output.substitutions + output.deletions + output.insertions
                per_turn_cer[turn_id] = round(edits / len(ref_clean), 3)
                total_edits += edits
                total_chars += len(ref_clean)
                if risk >= self.risk_threshold:
                    complex_edits += edits
                    complex_chars += len(ref_clean)
                else:
                    simple_edits += edits
                    simple_chars += len(ref_clean)
                for chunk in output.alignments[0]:
                    if chunk.type == "substitute":
                        ref_span = ref_clean[chunk.ref_start_idx : chunk.ref_end_idx]
                        hyp_span = hyp_clean[chunk.hyp_start_idx : chunk.hyp_end_idx]
                        key = f"{ref_span} → {hyp_span}"
                        substitution_counts[key] = substitution_counts.get(key, 0) + 1

            overall_cer = total_edits / total_chars if total_chars else 0.0
            accuracy = 1 - min(1.0, overall_cer)

            sub_metrics: dict[str, MetricScore] = {}
            if language_evaluated_ids:
                flagged_ids = [turn_id for turn_id in language_evaluated_ids if turn_id in per_turn_language_flag]
                sub_metrics["wrong_language_rate"] = make_rate_sub_metric(
                    parent_name=self.name,
                    key="wrong_language_rate",
                    numerator=len(flagged_ids),
                    denominator=len(language_evaluated_ids),
                    details={
                        "count": len(flagged_ids),
                        "num_evaluated": len(language_evaluated_ids),
                        "turn_ids": flagged_ids,
                        "reasons": {str(tid): per_turn_language_flag[tid] for tid in flagged_ids},
                    },
                )
            for sub_key, bucket_edits, bucket_chars, bucket_ids in (
                ("complex_text_accuracy", complex_edits, complex_chars, complex_ids),
                ("simple_text_accuracy", simple_edits, simple_chars, simple_ids),
            ):
                if bucket_chars > 0:
                    bucket_accuracy = max(0.0, 1 - bucket_edits / bucket_chars)
                    sub_metrics[sub_key] = MetricScore(
                        name=f"{self.name}.{sub_key}",
                        score=round(bucket_accuracy, 3),
                        normalized_score=round(bucket_accuracy, 3),
                        details={
                            "edits": bucket_edits,
                            "reference_characters": bucket_chars,
                            "turn_ids": bucket_ids,
                            "mean_text_risk": round(sum(per_turn_risk[tid] for tid in bucket_ids) / len(bucket_ids), 3),
                        },
                    )

            top_substitutions = sorted(substitution_counts.items(), key=lambda item: item[1], reverse=True)[:10]
            feature_rates = {
                feature: round(sum(1 for flags in per_turn_features.values() if flags[feature]) / len(pairs), 3)
                for feature in _RISK_WEIGHTS
            }

            return MetricScore(
                name=self.name,
                score=round(overall_cer, 3),
                normalized_score=round(accuracy, 3),
                details={
                    "cer": round(overall_cer, 3),
                    "accuracy": round(accuracy, 3),
                    "language": self.language,
                    "num_turns": len(pairs),
                    "risk_threshold": self.risk_threshold,
                    "mean_text_risk": round(sum(per_turn_risk.values()) / len(per_turn_risk), 3),
                    "num_complex_turns": len(complex_ids),
                    "num_simple_turns": len(simple_ids),
                    "risk_feature_rate": feature_rates,
                    "per_turn_cer": per_turn_cer,
                    "per_turn_text_risk": per_turn_risk,
                    "per_turn_risk_features": per_turn_features,
                    "per_turn_language_flag": per_turn_language_flag,
                    "top_substitutions": [{"error": error, "count": count} for error, count in top_substitutions],
                    "reference_characters": total_chars,
                },
                sub_metrics=sub_metrics or None,
            )

        except Exception as e:
            return self._handle_error(e, context)
