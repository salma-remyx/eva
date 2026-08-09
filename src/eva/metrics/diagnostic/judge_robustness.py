"""Diagnostic metric: adversarial robustness of reference-free LLM-judge metrics.

Adapted from "Measuring the Robustness of Reference-Free Dialogue Evaluation
Systems" (https://arxiv.org/abs/2501.06728v1), which benchmarks how four
families of adversarial perturbations -- speaker-tag prefixes, static
responses, ungrammatical responses, and repeated conversational context --
shift the scores of reference-free dialogue metrics (DialogRPT, UniEval,
PromptEval).

This metric ports those four attack categories as a diagnostic that wraps an
existing EVA LLM-judge metric (selected by name from the registry) and reports
the normalized-score shift between the clean transcript and each perturbed
transcript. A small mean shift means the wrapped judge is robust to the
attacks; a large shift means it is easily fooled.

The paper's standalone benchmark suite and its learned metric estimators are
intentionally out of scope -- EVA's native judges stand in as the
reference-free metric under test, and evaluation of the resulting robustness
numbers belongs in a downstream analysis.
"""

from copy import copy
from typing import Any

from eva.metrics.base import BaseMetric, MetricContext
from eva.metrics.registry import get_global_registry, register_metric
from eva.models.results import MetricScore

# The four adversarial attack categories from the paper.
SPEAKER_TAG_PREFIX = "speaker_tag_prefix"
STATIC_RESPONSE = "static_response"
UNGRAMMATICAL_RESPONSE = "ungrammatical_response"
REPEATED_CONTEXT = "repeated_context"

DEFAULT_ATTACKS: tuple[str, ...] = (
    SPEAKER_TAG_PREFIX,
    STATIC_RESPONSE,
    UNGRAMMATICAL_RESPONSE,
    REPEATED_CONTEXT,
)

# Generic, non-substantive boilerplate used for the static-response attack.
_STATIC_RESPONSE_TEXT = "I understand. How can I help you today?"
_LEADING_ARTICLES = frozenset({"the", "a", "an"})


def _corrupt_grammar(text: str) -> str:
    """Return a deterministically ungrammatical version of ``text``.

    Drops leading articles and duplicates the first remaining word, e.g.
    "The flight is booked." -> "flight flight is booked.".
    """
    words = text.split()
    while words and words[0].lower().rstrip(".,!?;:") in _LEADING_ARTICLES:
        words.pop(0)
    if words:
        words[0] = f"{words[0]} {words[0]}"
    return " ".join(words)


def _transform_assistant_content(content: str, attack: str) -> str:
    """Apply a single content attack to one assistant response string."""
    if not content:
        return content
    if attack == SPEAKER_TAG_PREFIX:
        return f"Agent: {content}"
    if attack == STATIC_RESPONSE:
        return _STATIC_RESPONSE_TEXT
    if attack == UNGRAMMATICAL_RESPONSE:
        return _corrupt_grammar(content)
    return content


def _max_turn_id(trace: list[dict[str, Any]]) -> int:
    """Return the largest turn_id in ``trace`` (0 when empty)."""
    return max((entry.get("turn_id", 0) for entry in trace), default=0)


def perturb_conversation_trace(trace: list[dict[str, Any]], attack: str) -> list[dict[str, Any]]:
    """Return a copy of ``trace`` with ``attack`` applied.

    ``REPEATED_CONTEXT`` appends a re-numbered duplicate of the whole
    conversation; the other attacks rewrite assistant response content in
    place. User turns, tool calls, and tool responses are preserved.
    """
    if attack == REPEATED_CONTEXT:
        offset = _max_turn_id(trace)
        duplicate: list[dict[str, Any]] = []
        for entry in trace:
            new_entry = dict(entry)
            if "turn_id" in new_entry:
                new_entry["turn_id"] = new_entry["turn_id"] + offset
            duplicate.append(new_entry)
        return [*trace, *duplicate]

    perturbed: list[dict[str, Any]] = []
    for entry in trace:
        new_entry = dict(entry)
        if entry.get("role") == "assistant" and entry.get("content"):
            new_entry["content"] = _transform_assistant_content(entry["content"], attack)
        perturbed.append(new_entry)
    return perturbed


def perturb_assistant_turns(turns: dict[int, str], attack: str) -> dict[int, str]:
    """Mirror an attack onto an ``intended_assistant_turns`` mapping.

    Lets the wrapper stay correct for judges that read intended turns
    (e.g. ``speakability``) rather than the full ``conversation_trace``.
    """
    if attack == REPEATED_CONTEXT:
        if not turns:
            return {}
        offset = max(turns)
        duplicate = {turn_id + offset: content for turn_id, content in turns.items()}
        return {**turns, **duplicate}

    return {turn_id: _transform_assistant_content(content, attack) for turn_id, content in turns.items()}


@register_metric
class JudgeRobustnessDiagnosticMetric(BaseMetric):
    """Diagnostic: how stable is a reference-free judge under adversarial attacks?

    Wraps an existing LLM-judge metric (by default ``conciseness``) and
    re-scores the conversation after applying each of the four perturbations
    from arXiv:2501.06728. The reported score is the mean absolute
    normalized-score shift across the enabled attacks -- lower means the judge
    is more robust.

    Opt-in and expensive (it runs the wrapped judge once per attack plus a
    clean baseline), so it is excluded from the default metric set and from
    pass@k.
    """

    name = "judge_robustness"
    description = "Diagnostic: adversarial score-shift robustness of a reference-free LLM-judge metric"
    category = "diagnostic"
    higher_is_better = False  # lower mean shift == more robust judge
    exclude_from_pass_at_k = True
    exclude_from_default_metrics = True

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.judge_metric: str = self.config.get("judge_metric") or "conciseness"
        configured_attacks = self.config.get("attacks", DEFAULT_ATTACKS)
        self.attacks: tuple[str, ...] = tuple(configured_attacks)

    def _build_inner_judge(self) -> BaseMetric | None:
        """Resolve and instantiate the wrapped judge metric from the registry."""
        judge_cls = get_global_registry().get(self.judge_metric)
        if judge_cls is None:
            return None
        return judge_cls(config=self.config.get("judge_config") or {})

    def _score_value(self, metric_score: MetricScore) -> float | None:
        """Extract the comparable score from a wrapped judge's MetricScore."""
        if metric_score.error:
            return None
        value = metric_score.normalized_score
        if value is None:
            value = metric_score.score
        return float(value) if value is not None else None

    def _perturbed_context(self, context: MetricContext, attack: str) -> MetricContext:
        """Shallow-copy ``context`` with ``attack`` applied to its transcript."""
        perturbed = copy(context)
        perturbed.conversation_trace = perturb_conversation_trace(context.conversation_trace, attack)
        perturbed.intended_assistant_turns = perturb_assistant_turns(context.intended_assistant_turns, attack)
        return perturbed

    async def compute(self, context: MetricContext) -> MetricScore:
        """Run the wrapped judge on the clean plus each perturbed transcript."""
        judge = self._build_inner_judge()
        if judge is None:
            return MetricScore(
                name=self.name,
                score=0.0,
                normalized_score=0.0,
                error=f"Unknown wrapped judge metric: {self.judge_metric!r}",
            )
        if not context.conversation_trace:
            return MetricScore(
                name=self.name,
                score=0.0,
                normalized_score=0.0,
                error="No transcript available",
            )

        try:
            baseline_score = await judge.compute(context)
        except Exception as error:
            return self._handle_error(error, context)

        baseline_value = self._score_value(baseline_score)
        if baseline_value is None:
            return MetricScore(
                name=self.name,
                score=0.0,
                normalized_score=0.0,
                error=f"Wrapped judge {self.judge_metric!r} failed on baseline: {baseline_score.error}",
            )

        per_attack: dict[str, dict[str, Any]] = {}
        sub_metrics: dict[str, MetricScore] = {}
        shifts: list[float] = []

        for attack in self.attacks:
            perturbed_context = self._perturbed_context(context, attack)
            try:
                perturbed_score = await judge.compute(perturbed_context)
            except Exception as error:
                self.logger.warning(f"[{context.record_id}] attack {attack} raised: {error}")
                per_attack[attack] = {"error": str(error)}
                continue

            perturbed_value = self._score_value(perturbed_score)
            if perturbed_value is None:
                per_attack[attack] = {"score": None, "error": perturbed_score.error}
                continue

            shift = abs(perturbed_value - baseline_value)
            shifts.append(shift)
            per_attack[attack] = {
                "score": round(perturbed_value, 4),
                "shift": round(shift, 4),
            }
            sub_metrics[f"{attack}_shift"] = MetricScore(
                name=f"{self.name}.{attack}_shift",
                score=round(shift, 4),
                normalized_score=round(shift, 4),
                details={"attack": attack, "baseline_score": round(baseline_value, 4)},
            )

        mean_shift = round(sum(shifts) / len(shifts), 4) if shifts else 0.0

        return MetricScore(
            name=self.name,
            score=mean_shift,
            normalized_score=mean_shift,
            details={
                "judge_metric": self.judge_metric,
                "baseline_score": round(baseline_value, 4),
                "attacks": self.attacks,
                "per_attack": per_attack,
            },
            sub_metrics=sub_metrics or None,
        )
