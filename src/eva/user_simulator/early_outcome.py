"""Early outcome prediction: halt conversations whose result is already evident.

Adapted from "EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction"
(arXiv:2609.02783), which trains success/failure classifiers over partial agent
trajectories and halts a run the moment a calibrated confidence threshold is
crossed. EVA has no labelled trajectory corpus to train classifiers on, so the
LightGBM pair is replaced with a parameter-free evidence score computed from
the user simulator's own event stream plus the record's goal decision tree.
The core mechanism — per-turn outcome prediction with a confidence-threshold
halt — is preserved. Evidence weights are hand-set defaults: treat the scores
as monotone evidence strength, not calibrated probabilities.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from eva.models.config import EarlyOutcomeConfig
from eva.utils.conversation_checks import LLM_GENERIC_ERROR_MESSAGE
from eva.utils.logging import get_logger

logger = get_logger(__name__)

EARLY_HALT_REASON = "early_halt"
EARLY_HALT_ARTIFACT_FILENAME = "early_halt.json"

Outcome = Literal["success", "failure"]

# Fraction of a phrase's content words that must appear in the conversation
# before the phrase counts as expressed.
_PHRASE_MATCH_RATIO = 0.5

# Words too generic to signal goal progress when shared by a criterion and
# the transcript.
_STOPWORDS = frozenset(
    "the a an and or for with that this these those you your yours are was were "
    "be been being to of in on at by from as if no not none any all each most "
    "more less than then them they there their has have had having can could "
    "will would should must may might when while before after during about into "
    "over under again further once here where why how what who whom which".split()
)

# Frustration signals grouped by kind; every kind matched at least once in the
# user's speech counts as one unit of failure evidence.
_FRUSTRATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("escalation", re.compile(r"\b(supervisor|manager|representative|human agent|real person|someone else)\b")),
    ("dissatisfaction", re.compile(r"\b(unacceptable|ridiculous|useless|frustrat\w*|annoy\w*|worst|hate)\b")),
    ("giving_up", re.compile(r"\b(forget it|never ?mind|give up|cancel everything|call back later|wrong number)\b")),
    ("no_progress", re.compile(r"\b(already told|keep asking|same (?:thing|question)|repeatedly|going in circles)\b")),
)


def _content_words(text: str) -> set[str]:
    """Extract lowercase tokens that can carry goal-related meaning.

    Keeps alphabetic tokens of 3+ characters and any token containing digits —
    dates, times, and prices are the most discriminative criterion tokens.
    """
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _STOPWORDS:
            continue
        if any(char.isdigit() for char in token) or len(token) >= 3:
            tokens.add(token)
    return tokens


def _noisy_or(terms: list[float]) -> float:
    """Combine independent evidence terms into a confidence in [0, 1]."""
    remaining = 1.0
    for term in terms:
        remaining *= 1.0 - min(max(term, 0.0), 1.0)
    return 1.0 - remaining


@dataclass(frozen=True)
class EarlyOutcomeDecision:
    """Halt decision emitted once predicted confidence crosses the threshold."""

    outcome: Outcome
    confidence: float
    user_turns: int
    assistant_turns: int
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form used for the event log and the artifact file."""
        return {
            "outcome": self.outcome,
            "confidence": round(self.confidence, 4),
            "user_turns": self.user_turns,
            "assistant_turns": self.assistant_turns,
            "features": self.features,
        }


class EarlyOutcomeMonitor:
    """Predict the conversation outcome from partial simulator events.

    Feeds on the provider-neutral event stream (user/assistant speech, errors)
    and the record's goal decision tree. After every completed user turn it
    scores failure and success evidence; when the configured threshold is met
    it emits a single halt decision and writes ``early_halt.json``.
    """

    def __init__(self, config: EarlyOutcomeConfig, goal: dict[str, Any], output_dir: Path) -> None:
        """Initialize the monitor.

        Args:
            config: Threshold and halt-policy settings.
            goal: Resolved user goal containing a ``decision_tree`` of criteria.
            output_dir: Record output directory for the halt artifact.
        """
        self._config = config
        self._output_dir = Path(output_dir)
        decision_tree = (goal or {}).get("decision_tree") or {}
        criteria = decision_tree.get("must_have_criteria") or []
        self._criteria_tokens = [_content_words(str(criterion)) for criterion in criteria]
        self._resolution_tokens = _content_words(str(decision_tree.get("resolution_condition") or ""))
        self._failure_tokens = _content_words(str(decision_tree.get("failure_condition") or ""))
        self._transcript_tokens: set[str] = set()
        self._user_turns = 0
        self._assistant_turns = 0
        self._generic_error_count = 0
        self._error_event_count = 0
        self._max_repeated_assistant_turn = 1
        self._frustration_kinds: set[str] = set()
        self._last_assistant_text = ""
        self._current_repeat_run = 1
        self._halted = False

    @property
    def halted(self) -> bool:
        """Whether a halt decision has already been emitted."""
        return self._halted

    def on_event(self, event: dict[str, Any]) -> EarlyOutcomeDecision | None:
        """Ingest one simulator event and evaluate the halt policy.

        Args:
            event: Event as recorded by ``UserSimulatorEventLogger.log_event``.

        Returns:
            The halt decision, or ``None`` while the outcome is still uncertain.
        """
        if self._halted:
            return None
        event_type = event.get("type") or event.get("event_type")
        data = event.get("data") or {}
        if event_type == "user_speech":
            text = str(data.get("text") or "")
            self._user_turns += 1
            self._transcript_tokens |= _content_words(text)
            self._track_frustration(text)
        elif event_type == "assistant_speech":
            text = str(data.get("text") or "")
            self._assistant_turns += 1
            self._transcript_tokens |= _content_words(text)
            self._track_assistant_text(text)
        elif event_type == "error":
            self._error_event_count += 1
        else:
            return None
        # Only score at exchange boundaries: the user's reply to an assistant
        # turn is the most informative point to re-evaluate the outcome.
        if event_type != "user_speech":
            return None
        return self._evaluate()

    def predict(self) -> tuple[float, float]:
        """Score the trajectory observed so far.

        Returns:
            ``(success_confidence, failure_confidence)`` from goal-criteria
            coverage and behavioral failure evidence respectively.
        """
        return self._success_confidence(), self._failure_confidence()

    def features(self) -> dict[str, Any]:
        """Snapshot of the behavioral and textual features behind a prediction."""
        covered = sum(1 for tokens in self._criteria_tokens if self._phrase_expressed(tokens))
        return {
            "user_turns": self._user_turns,
            "assistant_turns": self._assistant_turns,
            "generic_error_count": self._generic_error_count,
            "error_event_count": self._error_event_count,
            "max_repeated_assistant_turn": self._max_repeated_assistant_turn,
            "frustration_kinds": sorted(self._frustration_kinds),
            "criteria_covered": covered,
            "criteria_total": len(self._criteria_tokens),
            "failure_condition_match": self._phrase_expressed(self._failure_tokens),
            "resolution_condition_match": self._phrase_expressed(self._resolution_tokens),
        }

    def _evaluate(self) -> EarlyOutcomeDecision | None:
        """Apply the halt policy at the current exchange boundary."""
        if self._user_turns < self._config.min_user_turns:
            return None
        success_confidence, failure_confidence = self.predict()
        halt_failure = failure_confidence >= self._config.confidence_threshold
        halt_success = self._config.halt_on == "any" and success_confidence >= self._config.confidence_threshold
        if not (halt_failure or halt_success):
            return None
        self._halted = True
        if halt_failure:
            outcome: Outcome = "failure"
            confidence = failure_confidence
        else:
            outcome = "success"
            confidence = success_confidence
        decision = EarlyOutcomeDecision(
            outcome=outcome,
            confidence=confidence,
            user_turns=self._user_turns,
            assistant_turns=self._assistant_turns,
            features=self.features(),
        )
        self._save_artifact(decision)
        return decision

    def _failure_confidence(self) -> float:
        """Noisy-OR over behavioral and textual failure evidence."""
        repeat_run = self._max_repeated_assistant_turn
        terms = [
            0.5 if self._generic_error_count >= 1 else 0.0,
            0.4 if self._generic_error_count >= 3 else 0.0,
            0.0 if repeat_run < 2 else (0.2 if repeat_run == 2 else 0.35),
            0.15 if self._error_event_count == 1 else (0.3 if self._error_event_count >= 2 else 0.0),
            min(0.2 * len(self._frustration_kinds), 0.6) if self._frustration_kinds else 0.0,
            0.6 if self._phrase_expressed(self._failure_tokens) else 0.0,
        ]
        return _noisy_or(terms)

    def _success_confidence(self) -> float:
        """Goal-progress score: must-have criteria plus the resolution condition."""
        if not self._criteria_tokens:
            coverage = 0.0
        else:
            covered = sum(1 for tokens in self._criteria_tokens if self._phrase_expressed(tokens))
            coverage = covered / len(self._criteria_tokens)
        resolution = 0.25 if self._phrase_expressed(self._resolution_tokens) else 0.0
        return min(0.95, 0.75 * coverage + resolution)

    def _phrase_expressed(self, tokens: set[str]) -> bool:
        """Check whether enough of a phrase's content words appeared in the conversation."""
        if not tokens:
            return False
        overlap = len(tokens & self._transcript_tokens) / len(tokens)
        return overlap >= _PHRASE_MATCH_RATIO

    def _track_assistant_text(self, text: str) -> None:
        """Update behavioral features from one assistant utterance."""
        if LLM_GENERIC_ERROR_MESSAGE in text:
            self._generic_error_count += 1
        if text == self._last_assistant_text:
            self._current_repeat_run += 1
        else:
            self._current_repeat_run = 1
            self._last_assistant_text = text
        self._max_repeated_assistant_turn = max(self._max_repeated_assistant_turn, self._current_repeat_run)

    def _track_frustration(self, text: str) -> None:
        """Flag frustration markers in one user utterance."""
        lowered = text.lower()
        for kind, pattern in _FRUSTRATION_PATTERNS:
            if pattern.search(lowered):
                self._frustration_kinds.add(kind)

    def _save_artifact(self, decision: EarlyOutcomeDecision) -> None:
        """Persist the halt decision next to the record's other artifacts."""
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self._output_dir / EARLY_HALT_ARTIFACT_FILENAME
            artifact_path.write_text(json.dumps(decision.to_dict(), indent=2) + "\n")
        except OSError:
            logger.warning(f"Could not write early-halt artifact to {self._output_dir}", exc_info=True)


def is_early_halted(record_dir: Path) -> bool:
    """Check whether a record's conversation was halted by early outcome prediction."""
    return (Path(record_dir) / EARLY_HALT_ARTIFACT_FILENAME).exists()
