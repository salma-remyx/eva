"""Long-term memory recall metric using LLM-as-judge (whole conversation).

Adapted from RUMBA (Russian User Memory BenchmArk, arXiv:2607.21447), which
introduces a fine-grained taxonomy for evaluating long-term conversational
memory along semantic type, session scope, temporal reasoning, and the
explicitness of recalled information.

EVA has no QA-pair memory dataset, so this is an adapted port (Mode 2): we keep
RUMBA's core contribution — a fine-grained *taxonomy of memory failure modes*
scored by an LLM judge — and substitute RUMBA's Russian/English QA-pair
benchmark instrument with EVA's existing conversation-trace judge contract
(same I/O as ``faithfulness.py``). The judge reads the full multi-turn
transcript and rates whether the assistant recalls, combines, and reasons
consistently over information established in earlier turns.
"""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import (
    get_assistant_turns_disclaimer,
    get_user_turns_disclaimer,
)
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

# Memory-failure dimensions adapted from RUMBA's memory-question taxonomy
# (retrieval/explicitness, cross-turn combination+reasoning, temporal
# reasoning, and persistent session-scope constraints). Each is surfaced as a
# binary "issue occurred" sub-metric; the ``_rate`` suffix tells the aggregator
# these are lower-is-better issue frequencies.
_MEMORY_DIMENSION_KEYS = (
    "forgetting_established_facts",
    "inconsistent_cross_turn_reasoning",
    "temporal_reasoning_error",
    "ignoring_standing_constraints",
)


@register_metric
class MemoryRecallMetric(ConversationTextJudgeMetric):
    """LLM-based long-term memory recall metric (whole conversation).

    Evaluates whether the assistant maintains and consistently uses information
    established earlier in the conversation — recalling stated facts and
    preferences, combining information across turns, reasoning about dates and
    times, and honoring standing constraints.

    Rating scale: 1 (clear memory failure), 2 (minor/ambiguous lapse), 3 (consistent recall).
    The overall rating is the minimum across the four memory dimensions.
    """

    name = "memory_recall"
    version = "v0.1"
    description = (
        "LLM judge evaluation of long-term memory: whether the assistant recalls, "
        "combines, and reasons consistently over information established in earlier turns"
    )
    category = "accuracy"
    default_model = "us.anthropic.claude-opus-4-6-v1"
    default_params = {"max_tokens": 100000}  # Drop the OpenAI-only flex tier inherited from TextJudgeMetric.
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "agent_instructions": context.agent_instructions,
            "agent_role": context.agent_role,
            "conversation_trace": transcript_text,
            "current_date_time": context.current_date_time,
            "user_turns_disclaimer": get_user_turns_disclaimer(context.is_audio_native),
            "assistant_turns_disclaimer": get_assistant_turns_disclaimer(context.is_audio_native),
        }

    def build_metric_score(
        self,
        rating: int,
        normalized: float,
        response: dict,
        prompt: str,
        context: MetricContext,
        raw_response: str | None = None,
    ) -> MetricScore:
        """Build MetricScore with per-dimension memory-failure sub-metrics."""
        dimensions = response.get("dimensions", {}) if isinstance(response, dict) else {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_MEMORY_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        analysis = {"dimensions": dimensions}
        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": analysis,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )
