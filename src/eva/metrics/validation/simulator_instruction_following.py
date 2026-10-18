"""Graded instruction-following validation metric for the simulated user."""

from typing import Any

from eva.metrics.base import ConversationTextJudgeMetric, MetricContext
from eva.metrics.pipeline_prompts import get_user_turns_disclaimer
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.results import MetricScore


@register_metric
class SimulatorInstructionFollowingMetric(ConversationTextJudgeMetric):
    """Graded scenario-adherence and ordered goal-completion validation metric.

    Adapted from SimIF-Bench (SteerablePlex: Can We Steer Full-Duplex Models?,
    arXiv:2610.12201), which grades a conversational model on staying within a
    prescribed scenario and completing multiple goals in the required order —
    properties full-duplex user simulators lose as the conversation history
    grows. The prescribed scenario here is the record's ``user_goal`` decision
    tree (must-have criteria as the ordered goal sequence), and the subject
    under test is eva's user simulator (e.g., the OpenAI Realtime beta), so a
    drifting simulator is caught before it silently invalidates agent scores.

    Complements ``user_behavioral_fidelity``: instead of that metric's binary
    corruption verdict, this returns graded per-goal completion, in-order
    progression, scenario adherence, and an early-vs-late goal split that
    surfaces late-conversation control loss.

    Rating scale: 3 (faithful), 2 (minor drift), 1 (significant drift), 0 (lost control)
    """

    name = "simulator_instruction_following"
    version = "v0.1"
    description = "Graded validation of simulator scenario adherence and ordered goal completion"
    category = "validation"
    rating_scale = (0, 3)
    default_model = "gpt-5.2-medium"

    def get_prompt_variables(self, context: MetricContext, transcript_text: str) -> dict[str, Any]:
        """Return variables for prompt formatting."""
        return {
            "conversation_trace": transcript_text,
            "user_turns_disclaimer": get_user_turns_disclaimer(context.is_audio_native),
            "prescribed_scenario": _render_prescribed_scenario(context),
            "prescribed_goals": _render_goals(context.user_goal),
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
        """Build MetricScore with graded goal-progression and adherence sub-metrics."""
        goal_entries = _ordered_goal_entries(response.get("goal_analysis"))
        scenario_adherence = response.get("scenario_adherence")
        if not isinstance(scenario_adherence, dict):
            scenario_adherence = {}
        violations = scenario_adherence.get("violations")
        if not isinstance(violations, list):
            violations = []

        progression = _analyze_goal_progression(goal_entries)
        num_user_turns = sum(1 for entry in context.conversation_trace if entry.get("role") == "user")
        violation_turns = _distinct_violation_turns(violations)

        sub_metrics: dict[str, MetricScore] = {}
        if progression["num_goals"]:
            sub_metrics["goal_completion_accuracy"] = make_rate_sub_metric(
                parent_name=self.name,
                key="goal_completion_accuracy",
                numerator=progression["num_goals_completed"],
                denominator=progression["num_goals"],
                details={k: progression[k] for k in ("num_goals", "num_goals_completed")},
            )
            sub_metrics["ordered_goal_completion_accuracy"] = make_rate_sub_metric(
                parent_name=self.name,
                key="ordered_goal_completion_accuracy",
                numerator=progression["in_order_prefix"],
                denominator=progression["num_goals"],
                details={k: progression[k] for k in ("num_goals", "in_order_prefix")},
            )
            # Early-vs-late split (SimIF-Bench's degradation signal): completion of
            # goals in the first vs second half of the prescribed order. Only
            # emitted when both halves are non-empty.
            if progression["early_goals_total"] and progression["late_goals_total"]:
                sub_metrics["early_goal_completion_accuracy"] = make_rate_sub_metric(
                    parent_name=self.name,
                    key="early_goal_completion_accuracy",
                    numerator=progression["early_goals_completed"],
                    denominator=progression["early_goals_total"],
                    details={k: progression[k] for k in ("early_goals_total", "early_goals_completed")},
                )
                sub_metrics["late_goal_completion_accuracy"] = make_rate_sub_metric(
                    parent_name=self.name,
                    key="late_goal_completion_accuracy",
                    numerator=progression["late_goals_completed"],
                    denominator=progression["late_goals_total"],
                    details={k: progression[k] for k in ("late_goals_total", "late_goals_completed")},
                )
        sub_metrics["scenario_violation_rate"] = make_rate_sub_metric(
            parent_name=self.name,
            key="scenario_violation_rate",
            numerator=len(violation_turns),
            denominator=num_user_turns,
            details={
                "violating_turns": sorted(violation_turns),
                "num_user_turns": num_user_turns,
                "violations": violations,
            },
        )

        early_rate = _safe_rate(progression["early_goals_completed"], progression["early_goals_total"])
        late_rate = _safe_rate(progression["late_goals_completed"], progression["late_goals_total"])
        return MetricScore(
            name=self.name,
            score=float(rating),
            normalized_score=normalized,
            details={
                "rating": rating,
                "explanation": scenario_adherence.get("analysis", ""),
                "goal_analysis": goal_entries,
                "scenario_adherence": scenario_adherence,
                "goal_progression": progression,
                # Positive gap = goals late in the scenario completed less often —
                # the late-conversation control loss SimIF-Bench reports.
                "early_late_completion_gap": round(early_rate - late_rate, 3),
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            },
            sub_metrics=sub_metrics or None,
        )


def _render_prescribed_scenario(context: MetricContext) -> str:
    """Render the scenario context the simulator was given, minus the ordered goals.

    The decision-tree fields other than must-have criteria (resolution/failure
    conditions, negotiation and escalation behavior, edge cases) define what
    "staying within the scenario" means for the adherence analysis.
    """
    user_goal = context.user_goal
    lines = [f"User persona: {context.user_persona}"]
    if isinstance(user_goal, dict):
        decision_tree = user_goal.get("decision_tree")
        if not isinstance(decision_tree, dict):
            decision_tree = {}
        lines.extend(
            [
                f"High-level goal: {user_goal.get('high_level_user_goal', '')}",
                f"Resolution condition: {decision_tree.get('resolution_condition', '')}",
                f"Failure condition: {decision_tree.get('failure_condition', '')}",
                f"Negotiation behavior: {decision_tree.get('negotiation_behavior', [])}",
                f"Escalation behavior: {decision_tree.get('escalation_behavior', '')}",
                f"Edge cases: {decision_tree.get('edge_cases', [])}",
                f"Nice-to-have criteria (optional — not goals): {decision_tree.get('nice_to_have_criteria', [])}",
            ]
        )
    else:
        lines.append(f"High-level goal: {user_goal}")
    return "\n".join(lines)


def _render_goals(user_goal: Any) -> str:
    """Render the prescribed ordered goal list for the judge prompt.

    Must-have criteria are the simulator's goals in the order they must be
    satisfied. When they are prose (or absent), the judge is asked to derive the
    distinct checkable goals from the scenario instead, so the output shape is
    the same either way.
    """
    criteria = None
    if isinstance(user_goal, dict):
        decision_tree = user_goal.get("decision_tree")
        if isinstance(decision_tree, dict):
            criteria = decision_tree.get("must_have_criteria")

    if isinstance(criteria, list) and criteria:
        lines = ["The simulator's must-have criteria, in the order they must be satisfied:"]
        lines.extend(f"{i}. {criterion}" for i, criterion in enumerate(criteria, start=1))
        return "\n".join(lines)
    if isinstance(criteria, str) and criteria.strip():
        return (
            f"The simulator's must-have criteria are: \"{criteria.strip()}\"\n"
            "Identify the distinct checkable goals within them, in the prescribed order, and report "
            "them as goal_id 1..N."
        )
    return (
        "No explicit must-have criteria were provided. Identify the distinct checkable goals in the "
        "prescribed scenario above, in the prescribed order, and report them as goal_id 1..N."
    )


def _optional_int(value: Any) -> int | None:
    """Return ``value`` as an int, or None when it is not a usable turn/goal index."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _ordered_goal_entries(raw: Any) -> list[dict]:
    """Normalize the judge's goal_analysis into ordered, well-formed entries.

    Non-dict entries are dropped. When every entry carries a unique integer
    goal_id the list is sorted by it; otherwise the judge's list order (the
    prescribed order, per the prompt) is kept as-is.
    """
    if not isinstance(raw, list):
        return []
    entries = [entry for entry in raw if isinstance(entry, dict)]
    goal_ids = [entry.get("goal_id") for entry in entries]
    ids_are_usable = all(isinstance(index, int) and not isinstance(index, bool) for index in goal_ids)
    if ids_are_usable and len(set(goal_ids)) == len(goal_ids):
        entries.sort(key=lambda entry: entry["goal_id"])
    return entries


def _distinct_violation_turns(violations: list[Any]) -> set[int]:
    """Collect the distinct user turn indices flagged as scenario violations."""
    turns: set[int] = set()
    for violation in violations:
        if isinstance(violation, dict):
            turn = _optional_int(violation.get("turn_index"))
            if turn is not None:
                turns.add(turn)
    return turns


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return numerator/denominator, or 0.0 when the denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


def _analyze_goal_progression(goal_entries: list[dict]) -> dict[str, int]:
    """Compute graded progression statistics over the ordered goal entries.

    - completion: how many prescribed goals the simulator satisfied.
    - in-order prefix: the longest run of goals satisfied in the prescribed
      order (turn indices, when provided, must not go backwards; a goal without
      a turn index neither proves nor breaks the order).
    - early/late split: completion of goals in the first vs second half of the
      prescribed order — the per-record proxy for SimIF-Bench's finding that
      simulator control degrades as the conversation history grows. The split is
      only populated when at least two goals were analyzed.
    """
    total = len(goal_entries)
    completed = [bool(entry.get("completed")) for entry in goal_entries]
    turn_indices = [_optional_int(entry.get("turn_index")) for entry in goal_entries]

    in_order_prefix = 0
    last_turn: int | None = None
    for is_completed, turn_index in zip(completed, turn_indices):
        if not is_completed:
            break
        if turn_index is not None and last_turn is not None and turn_index < last_turn:
            break
        if turn_index is not None:
            last_turn = turn_index
        in_order_prefix += 1

    early_total = (total + 1) // 2 if total >= 2 else 0
    late_total = total - early_total if total >= 2 else 0
    return {
        "num_goals": total,
        "num_goals_completed": sum(completed),
        "in_order_prefix": in_order_prefix,
        "early_goals_total": early_total,
        "early_goals_completed": sum(completed[:early_total]) if early_total else 0,
        "late_goals_total": late_total,
        "late_goals_completed": sum(completed[early_total:]) if late_total else 0,
    }
