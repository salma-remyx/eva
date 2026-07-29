"""Information-acquisition metric — diagnosis-decoupled elicited-history scoring.

Evaluates whether the assistant elicited the information required to address
the user's goal, measured over the *elicited conversation history* and
deliberately decoupled from the quality of the final answer. This removes the
confound MedDDC-Eval names: in coupled evaluation, strong final-answer
generation can compensate for a thin history while weak generation can obscure
a rich one.

Adapted (Mode 2) from "MedDDC-Eval: Diagnosis-Decoupled Evaluation of
Multi-Turn Medical Consultation Agents" (arXiv:2607.18999). The paper's core
two-step measurement is ported at full fidelity — an LLM judge performs
*directional semantic coverage* of the required information items against the
elicited history, and the metric applies the paper's *deterministic one-to-one
assignment* (at most one credit per required item) to produce a coverage ratio.
Auxiliary components are substituted with target-native equivalents: the
medical D/T/E harness and frozen reader are replaced by EVA's existing
``ConversationTextJudgeMetric`` LLM-judge path, and the paper's separate
held-out benchmark suite and GRPO post-training are out of scope (EVA is an
evaluation framework, not a trainer).
"""

import json
from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import (
    get_assistant_turns_disclaimer,
    get_user_turns_disclaimer,
)
from eva.metrics.registry import register_metric
from eva.models.results import MetricScore


def coverage_ratio(required_information: list[Any]) -> tuple[float, int, int]:
    """Deterministic one-to-one coverage assignment over required items.

    Mirrors MedDDC-Eval's deterministic assignment step: each required
    information item is credited **at most once** when the judge marked it
    covered by the elicited history (directional coverage from the reference
    items into the transcript). Returns ``(ratio, num_covered, num_required)``;
    ``(0.0, 0, 0)`` when the judge identified no required items.
    """
    items = [i for i in (required_information or []) if isinstance(i, dict)]
    total = len(items)
    if total == 0:
        return 0.0, 0, 0
    covered = sum(1 for i in items if i.get("covered") is True)
    return covered / total, covered, total


@register_metric
class InformationAcquisitionMetric(ConversationTextJudgeMetric):
    """Diagnosis-decoupled information-acquisition judge (whole conversation).

    Scores how completely the assistant *elicited* the information needed to
    address the user's goal — independent of how well the final answer was
    phrased. The judge lists the required information items and marks which
    were captured in the elicited history; the metric deterministically
    aggregates them into a coverage ratio (recall of required items).

    Rating scale: 3 (adequate acquisition), 2 (partial), 1 (insufficient).
    The headline ``normalized_score`` is the deterministic coverage ratio in
    ``[0, 1]``; ``rating`` is a coarser human-readable summary kept in details.
    """

    name = "information_acquisition"
    version = "v0.1"
    description = (
        "LLM judge of whether the assistant elicited the information required to address the "
        "user's goal, measured over the elicited history and decoupled from final-answer quality"
    )
    category = "accuracy"
    default_model = "us.anthropic.claude-opus-4-6-v1"
    default_params = {"max_tokens": 100000}  # Drop the OpenAI-only flex tier inherited from TextJudgeMetric.
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "user_goal": context.user_goal,
            "user_persona": context.user_persona,
            "agent_instructions": context.agent_instructions,
            "agent_role": context.agent_role,
            "available_tools": json.dumps(context.agent_tools, indent=4),
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
        """Build MetricScore with the deterministic coverage ratio as the headline normalized score."""
        required = response.get("required_information", []) if isinstance(response, dict) else []
        ratio, covered, total = coverage_ratio(required)

        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=ratio,
            details={
                "rating": rating,
                "explanation": response.get("explanation", ""),
                "required_information": required,
                "coverage_ratio": ratio,
                "items_covered": covered,
                "items_required": total,
                "num_turns": len(context.conversation_trace),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
        )
