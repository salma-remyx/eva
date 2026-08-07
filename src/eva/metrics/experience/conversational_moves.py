"""Conversational-moves metric using LLM-as-judge (per-turn).

Evaluates each assistant turn along the discrete "conversational move"
dimension and rates whether the move the assistant realized was the
appropriate selection for the dialogue context.

The move taxonomy and the framing of dialogue management as *move selection*
are adapted from Latent-IM: Latent Interaction Management for Speech LLMs
(arXiv:2607.26928). Latent-IM recovers move selection and realization inside
an LLM and steers generation toward a chosen move. EVA cannot steer a model it
is benchmarking, so this metric instead *measures* whether the assistant's
chosen moves align with what a reference selection policy would pick — using
the existing LLM-judge path as that reference policy in place of Latent-IM's
learned latent-state estimator. The realization/steering half of Latent-IM is
out of scope here: there is no generation to control during evaluation.

For every assistant turn the judge (1) classifies the move it realized and
(2) rates the appropriateness of that move given the dialogue context. The
per-turn appropriateness ratings aggregate into the parent score; the move
classifications surface as a per-move distribution sub-metric.
"""

from typing import Any

from eva.metrics.base import MetricContext, PerTurnConversationJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_per_category_rate_sub_metrics
from eva.models.results import MetricScore

# Discrete dialogue moves a turn can realize. The five core moves are taken
# from Latent-IM (arXiv:2607.26928); `other` is a catch-all so the judge is
# never forced to mislabel a turn that does not fit the taxonomy.
CONVERSATIONAL_MOVES: tuple[str, ...] = (
    "acknowledging",
    "checking",
    "querying",
    "explaining",
    "replying",
    "other",
)


@register_metric
class ConversationalMovesJudgeMetric(PerTurnConversationJudgeMetric):
    """LLM-based conversational-moves metric (per-turn).

    For every assistant turn the judge classifies the move the assistant
    realized and rates whether that move was the appropriate selection for the
    dialogue context. Per-turn appropriateness ratings aggregate into the
    parent score; move classifications surface as a per-move distribution
    sub-metric (share of rated turns per move type).

    Rating scale: 3 (move well chosen), 2 (acceptable), 1 (wrong move)
    Normalized: 3 -> 1.0, 2 -> 0.5, 1 -> 0.0
    """

    name = "conversational_moves"
    version = "v0.1"
    description = "LLM judge of whether each assistant turn realized an appropriate conversational move"
    category = "experience"
    rating_scale = (1, 3)

    def get_expected_turn_ids(self, context: MetricContext) -> list[int]:
        """Return unique turn IDs from the conversation trace, preserving order."""
        return list(dict.fromkeys(e.get("turn_id") for e in context.conversation_trace if e.get("turn_id") is not None))

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {"conversation_turns": transcript_text}

    def process_turn_item(self, item: dict, turn_id: int, rating: int | None, context: MetricContext) -> dict[str, Any]:
        """Extract and validate the move label from the judge response item.

        Coerces any missing or unrecognized label to ``other`` so the
        distribution sub-metrics always have a valid denominator.
        """
        move = item.get("move")
        if isinstance(move, str):
            move = move.strip().lower()
        if move not in CONVERSATIONAL_MOVES:
            move = "other"
        return {"move": move}

    def build_sub_metrics(
        self,
        context: MetricContext,
        per_turn_ratings: dict[int, int | None],
        per_turn_extra: dict[int, dict[str, Any]],
    ) -> dict[str, MetricScore] | None:
        """Surface one sub-metric per move type: share of rated turns classified as that move."""
        rated_turn_ids = [tid for tid, r in per_turn_ratings.items() if r is not None]
        per_turn_moves = {tid: [extra["move"]] for tid, extra in per_turn_extra.items() if extra.get("move")}
        return (
            build_per_category_rate_sub_metrics(
                parent_name=self.name,
                categories=CONVERSATIONAL_MOVES,
                rated_turn_ids=rated_turn_ids,
                per_turn_categories=per_turn_moves,
            )
            or None
        )
