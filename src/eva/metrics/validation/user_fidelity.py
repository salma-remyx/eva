"""User Fidelity Score (UFS) metric for user-simulator evaluation.

Scores how faithfully the simulated user executed its assigned role —
persona, private information, decision tree, and end-of-call rules — using
a task-grounded rubric that is judged independently of agent success.
Adapted from "UserProxyBench: Evaluating LLM User Simulators for Agent
Benchmarks and Training" (arXiv:2609.38043), which found that
user-specification violations (dominated by premature disclosure: users
revealing private information before it is requested) are frequent even
among successful episodes and are invisible to reward-based scoring.

Unlike ``user_behavioral_fidelity`` — a corruption gate that only flags
user behavior which changed database state — this metric flags role
violations even when the final state is untouched, because the interaction
the agent was evaluated on still changed.
"""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.registry import register_metric
from eva.metrics.utils import build_binary_flag_sub_metrics
from eva.metrics.validation.user_behavioral_fidelity import _render_user_simulator_instructions
from eva.metrics.versioning import _CURRENT_PROMPT_HASH, hash_prompt_template
from eva.models.results import MetricScore

_USER_FIDELITY_CRITERIA_KEYS = (
    "premature_disclosure",
    "withheld_information",
    "persona_inconsistency",
    "protocol_violation",
)

_CONVERSATION_EVIDENCE = (
    "### Conversation (includes tool calls)\n"
    "{conversation_trace}\n\n"
    "### User-Side Text (ground truth for what the user said)\n"
    "{intended_user_turns}\n"
    "Evaluate what the user said against this text: in cascade pipelines the user turns in the "
    "transcript above are the agent's transcriptions and may mishear names, numbers, or codes, "
    "while this is what the user simulator actually said."
)


@register_metric
class UserFidelityMetric(ConversationTextJudgeMetric):
    """User Fidelity Score (UFS) validation metric for the simulated user.

    Judges the user simulator's adherence to its private instructions using
    four rubric criteria, each scored independently of agent success:

    - ``premature_disclosure``: revealed private information before the agent
      asked for it (the dominant failure family in UserProxyBench).
    - ``withheld_information``: failed to provide information the agent
      explicitly requested and the user instructions contain.
    - ``persona_inconsistency``: manner was pervasively inconsistent with the
      assigned persona.
    - ``protocol_violation``: violated an explicit decision-tree or
      end-of-call instruction.

    The parent score is the fraction of satisfied criteria (the UFS itself);
    per-criterion ``_rate`` sub-metrics report violation frequency
    (1.0 when violated, lower is better).

    Opt-in diagnostic: excluded from default metric runs and resolvable by
    name for explicit selection. Score existing runs with
    ``scripts/compute_user_fidelity.py``.

    ``version`` is intentionally unset for now: enrolling this metric in the
    versioned set requires regenerating ``tests/fixtures/metric_signatures.json``
    (``scripts/regen_metric_signatures.py``), which should happen together
    with wiring the module into ``eva.metrics.validation`` imports when the
    metric is promoted out of opt-in.
    """

    name = "user_fidelity"
    description = "User Fidelity Score: simulated-user adherence to its private instructions"
    category = "validation"
    rating_scale = (0, 1)
    default_model = "gpt-5.2-medium"
    exclude_from_default_metrics = True

    def get_judge_prompt(self, prompt_key: str = "user_prompt", **variables: Any) -> str:
        """Resolve the judge prompt from this metric's own top-level namespace.

        The PromptManager merges prompt files shallowly by top-level key, so a
        new file cannot extend the ``judge`` namespace owned by judge.yaml
        without clobbering it; this metric's template therefore ships under
        ``user_fidelity`` in configs/prompts/user_fidelity.yaml. Stamps the
        template hash like the base implementation. Move the template into
        judge.yaml (as ``judge.user_fidelity``) and drop this override when
        the metric is promoted out of opt-in.
        """
        prompt_path = f"{self.name}.{prompt_key}"
        _CURRENT_PROMPT_HASH.set(hash_prompt_template(self.prompt_manager.get_template(prompt_path)))
        return self.prompt_manager.get_prompt(prompt_path, **variables)

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        conversation_evidence = _CONVERSATION_EVIDENCE.format(
            conversation_trace=transcript_text,
            intended_user_turns=context.intended_user_turns,
        )
        return {
            "conversation_evidence": conversation_evidence,
            # Reuse the corruption metric's renderer so both user-evaluation
            # judges ground their rubric in the exact user-simulator system
            # prompt the simulator was given for this record.
            "user_simulator_instructions": _render_user_simulator_instructions(context),
            "language_display_name": context.language_display_name,
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
        """Build MetricScore with the rubric-derived UFS and per-criterion violation sub-metrics."""
        criteria_analysis = response.get("criteria_analysis", {}) or {}
        sub_metrics = build_binary_flag_sub_metrics(
            parent_name=self.name,
            entries=criteria_analysis,
            entry_keys=_USER_FIDELITY_CRITERIA_KEYS,
            flag_field="violated",
            detail_fields=("analysis",),
        )

        # UFS = fraction of rubric criteria the user satisfied. Computed from
        # the per-criterion verdicts rather than the holistic rating so the
        # score stays rubric-grounded; falls back to the rating when the judge
        # returned no usable per-criterion verdicts.
        num_violated = sum(int(sm.score) for sm in sub_metrics.values())
        if sub_metrics:
            ufs = 1.0 - num_violated / len(sub_metrics)
        else:
            ufs = float(normalized)

        return MetricScore(
            name=self.name,
            score=round(ufs, 3),
            normalized_score=round(ufs, 3),
            details={
                "rating": rating,
                "user_fidelity_score": round(ufs, 3),
                "num_criteria_violated": num_violated,
                "criteria_analysis": criteria_analysis,
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )
