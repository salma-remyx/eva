"""Theory-of-Mind metric using LLM-as-judge (whole conversation).

Adapted from *MeetingToM: Evaluating Multimodal LLMs on Theory-of-Mind
Reasoning in Multi-Party Meetings* (arXiv:2607.19235). The paper's portable
contribution is a Theory-of-Mind (ToM) rubric: does the model infer the
other party's latent beliefs, intentions, and states of knowledge from
dialogue cues, rather than taking utterances at face value? Its signature
phenomenon is **pseudo-consensus** — apparent agreement that masks private
dissent under social pressure.

Mode-2 adaptation to EVA's two-party user/assistant voice setting:
  * The paper annotates latent social states against a multi-party meeting
    benchmark and reasons over non-verbal/video cues. Here the ground-truth
    latent state is taken from the existing ``user_goal`` / ``user_persona``
    fields (the oracle the assistant should have inferred) and the dialogue
    signal is the text transcript (``conversation_trace``).
  * Subject-level mental-state prediction and knowledge-state reasoning port
    directly. Pseudo-consensus is reframed for two parties: the user's
    superficial "sure / I guess that's fine" should not be mistaken for
    genuine buy-in.
  * Intentionally out of scope (they collapse in a two-party setting): the
    paper's group-level consensus reasoning, dyadic addressee disambiguation,
    and non-verbal cue integration.
"""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import get_assistant_turns_disclaimer, get_user_turns_disclaimer
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.models.results import MetricScore

# Ordered keys of the judge-response ``dimensions`` block. Must match the
# prompt's dimension names so sub-metrics are emitted deterministically.
_THEORY_OF_MIND_DIMENSION_KEYS = (
    "missed_latent_intent",
    "pseudo_consensus_unrecognized",
    "knowledge_state_misread",
)


@register_metric
class TheoryOfMindJudgeMetric(ConversationTextJudgeMetric):
    """LLM-based Theory-of-Mind metric (conversation-level).

    Judges whether the assistant modeled the user's mind across the
    conversation — inferring the underlying goal, recognizing pseudo-consensus
    (surface agreement masking dissent), and tracking what the user does and
    does not know — instead of acting on literal utterances alone.

    Rating scale: 3 (strong ToM), 2 (partial), 1 (poor)
    Normalized: 3 -> 1.0, 2 -> 0.5, 1 -> 0.0
    """

    name = "theory_of_mind"
    version = "v0.1"
    description = (
        "LLM judge of whether the assistant inferred the user's latent intent, pseudo-consensus, and knowledge state"
    )
    category = "experience"
    rating_scale = (1, 3)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting.

        ``user_goal`` and ``user_persona`` are the ground-truth latent state
        (the beliefs/intentions the assistant should have inferred) used as the
        oracle the judge compares the assistant's behavior against.
        """
        return {
            "conversation_trace": transcript_text,
            "user_goal": context.user_goal,
            "user_persona": context.user_persona,
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
        """Build MetricScore with per-dimension issue-flag sub-metrics.

        Each sub-metric reads as the fraction of records where that ToM failure
        mode occurred (1.0 when flagged, 0.0 otherwise); lower is better.
        """
        dimensions = response.get("dimensions", {}) or {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=dimensions,
            entry_keys=_THEORY_OF_MIND_DIMENSION_KEYS,
            flag_field="flagged",
            detail_fields=("rating", "evidence"),
        )

        analysis = {
            "dimensions": dimensions,
            "flags_count": response.get("flags_count", ""),
        }
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
