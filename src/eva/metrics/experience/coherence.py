"""Turn-level dialogue coherence metric using LLM-as-judge.

Adapted from *ECoh: Turn-level Coherence Evaluation for Multilingual Dialogues*
(Passali et al., 2024 — arXiv:2407.11660). The paper's core contribution — a
lightweight, multilingual, turn-level coherence evaluator that judges whether
each assistant response is locally coherent with the preceding dialogue — is
kept at full fidelity. The paper's fine-tuned evaluator model and GenResCoh
training dataset are substituted with EVA's existing configurable judge LLM
(zero-shot via the prompt template), matching how the other LLM-judge metrics in
this package operate.
"""

from typing import Any

from eva.metrics.base import MetricContext, PerTurnConversationJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_per_category_rate_sub_metrics
from eva.models.results import MetricScore

# Local coherence failure modes — the kinds of turn-level incoherence the
# evaluator looks for. Distinct from conversation_progression (goal advancement),
# faithfulness (policy / grounding), and conciseness (length).
_COHERENCE_FAILURE_MODES = (
    "non_sequitur",
    "contradicts_context",
    "topic_drift",
    "ignores_user_input",
)


@register_metric
class CoherenceJudgeMetric(PerTurnConversationJudgeMetric):
    """LLM-based turn-level dialogue coherence metric.

    Evaluates whether each assistant response is locally coherent with the
    preceding dialogue context — it follows from what was just said, does not
    contradict established facts, stays on the user's current concern, and
    actually addresses the user's most recent turn. Multilingual by construction:
    the judge evaluates in the conversation's language.

    Rating scale: 3 (coherent), 2 (minor coherence issue), 1 (incoherent)
    Normalized: 3->1.0, 2->0.5, 1->0.0
    """

    name = "coherence"
    version = "v0.1"
    description = "LLM judge evaluation of turn-level dialogue coherence (multilingual)"
    category = "experience"
    rating_scale = (1, 3)

    def get_expected_turn_ids(self, context: MetricContext) -> list[int]:
        """Return unique turn IDs from conversation trace, preserving order."""
        return list(dict.fromkeys(e.get("turn_id") for e in context.conversation_trace if e.get("turn_id") is not None))

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "conversation_turns": transcript_text,
            "language_display_name": context.language_display_name,
        }

    def process_turn_item(self, item: dict, turn_id: int, rating: int | None, context: MetricContext) -> dict[str, Any]:
        """Extract and validate failure_modes from the judge response item."""
        if rating is None:
            return {"failure_modes": []}

        failure_modes = item.get("failure_modes", [])
        if isinstance(failure_modes, str):
            failure_modes = [failure_modes] if failure_modes else []
        elif isinstance(failure_modes, list):
            failure_modes = [str(fm) for fm in failure_modes if fm]
        else:
            failure_modes = []

        if rating == self.rating_scale[1] and failure_modes:
            self.logger.warning(
                f"[{context.record_id}] Turn {turn_id}: rating={rating} but failure_modes={failure_modes}; clearing"
            )
            failure_modes = []

        return {"failure_modes": failure_modes}

    def build_sub_metrics(
        self,
        context: MetricContext,
        per_turn_ratings: dict[int, int | None],
        per_turn_extra: dict[int, dict[str, Any]],
    ) -> dict[str, MetricScore] | None:
        """Surface one sub-metric per coherence failure mode, rate = flagged turns / rated turns."""
        rated_turn_ids = [tid for tid, r in per_turn_ratings.items() if r is not None]
        per_turn_failure_modes = {tid: extra.get("failure_modes") or [] for tid, extra in per_turn_extra.items()}
        return (
            build_per_category_rate_sub_metrics(
                parent_name=self.name,
                categories=_COHERENCE_FAILURE_MODES,
                rated_turn_ids=rated_turn_ids,
                per_turn_categories=per_turn_failure_modes,
            )
            or None
        )
