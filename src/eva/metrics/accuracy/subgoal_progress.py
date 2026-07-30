"""Subgoal-progress metric — per-turn task progress with automated error analysis.

Adapted from "Talk, Evaluate, Diagnose: User-aware Agent Evaluation with
Automated Error Analysis" (TED, arXiv:2603.15483).

TED's core mechanism is kept at full fidelity: an LLM judge reads the agent
trace and, per *subgoal*, marks the turn by which it was accomplished; those
per-subgoal completions are turned into a per-turn **progress curve** and
aggregated into TED's headline quantities:

  * **progress-curve AUC** — normalized area under the per-turn progress curve
    (rewards agents that complete subgoals *early*, not just eventually);
  * **final_progress** — terminal fraction of subgoals accomplished (a granular
    analog of the binary ``task_completion`` hash check);
  * **Progress-Per-Turn (PPT)** — average progress made per assistant turn
    (``final_progress / num_assistant_turns``);
  * **automated error categorization** — which subgoals were never reached and
    whether progress stalled (plateaued) late in the conversation.

Mode-2 substitutions (target-native auxiliaries in place of the paper's):
  * TED's task-specific subgoal lists are replaced with subgoals derived from
    EVA's own ground-truth scenario change — the field-level diff between
    ``initial_scenario_db`` and ``expected_scenario_db`` (the same ground truth
    ``task_completion`` hashes). When that diff is empty, a single coarse
    subgoal is synthesized from the user goal so the metric still runs.
  * TED's bespoke judge/eval harness is replaced by EVA's ``TextJudgeMetric``
    LLM client + prompt manager + metric registry, so the result plugs directly
    into ``MetricsRunner`` and its pass@k / bootstrap-CI aggregation.
"""

from typing import Any

from eva.metrics.base import MetricContext, MetricType, TextJudgeMetric
from eva.metrics.registry import register_metric
from eva.metrics.utils import format_transcript_with_tools
from eva.models.results import MetricScore
from eva.utils.hash_utils import compute_db_diff


def _flatten_diff_subgoals(diff: dict[str, Any]) -> list[dict[str, str]]:
    """Turn a scenario-DB diff into a flat list of ``{id, description}`` subgoals.

    Each distinct required change (table added/removed, record added/removed, or
    a field modification) becomes one checkable subgoal grounded in the data.
    """
    subgoals: list[dict[str, str]] = []

    for table in diff.get("tables_added", []) or []:
        subgoals.append({"description": f"Remove unexpected table '{table}'"})
    for table in diff.get("tables_removed", []) or []:
        subgoals.append({"description": f"Add required table '{table}'"})

    for table, table_diff in (diff.get("tables_modified", {}) or {}).items():
        # Non-dict tables (e.g. lists) carry their own expected/actual values.
        if not isinstance(table_diff, dict) or "records_added" not in table_diff:
            subgoals.append({"description": f"Update table '{table}' to its expected value"})
            continue
        for record in table_diff.get("records_added", []) or []:
            subgoals.append({"description": f"Remove record '{record}' from table '{table}'"})
        for record in table_diff.get("records_removed", []) or []:
            subgoals.append({"description": f"Add record '{record}' to table '{table}'"})
        for record, record_diff in (table_diff.get("records_modified", {}) or {}).items():
            subgoals.append({"description": _record_subgoal_description(table, record, record_diff)})

    for index, subgoal in enumerate(subgoals, start=1):
        subgoal["id"] = f"s{index}"
    return subgoals


def _record_subgoal_description(table: str, record: str, record_diff: Any) -> str:
    """Describe a single modified-record subgoal, drilling into changed fields."""
    if not isinstance(record_diff, dict):
        return f"Update record '{record}' in table '{table}'"

    modified_fields = list((record_diff.get("fields_modified", {}) or {}).keys())
    if modified_fields:
        fields = ", ".join(modified_fields)
        return f"Set field(s) [{fields}] on record '{record}' in table '{table}'"

    parts: list[str] = []
    added = record_diff.get("fields_added", []) or []
    removed = record_diff.get("fields_removed", []) or []
    if added:
        parts.append(f"add field(s) {added}")
    if removed:
        parts.append(f"remove field(s) {removed}")
    if parts:
        return f"Update record '{record}' in table '{table}': {', '.join(parts)}"
    return f"Update record '{record}' in table '{table}'"


def _derive_subgoals(context: MetricContext) -> list[dict[str, str]]:
    """Derive the subgoal list for a record from its ground-truth scenario change.

    Falls back to a single user-goal subgoal when the scenario DB does not need
    to change, so the metric always has at least one objective to track.
    """
    # compute_db_diff(expected, actual): what differs between the goal state and
    # the initial state is precisely what the agent must accomplish.
    diff = compute_db_diff(
        expected_db=context.expected_scenario_db,
        actual_db=context.initial_scenario_db,
    )
    subgoals = _flatten_diff_subgoals(diff)
    if subgoals:
        return subgoals
    return [{"id": "s1", "description": f"Resolve the user's request: {context.user_goal}"}]


def _build_progress_curve(
    completion_turns: dict[str, int | None],
    total: int,
    num_turns: int,
) -> list[float]:
    """Per-turn fraction of subgoals accomplished, for turns 1..num_turns."""
    curve: list[float] = []
    for turn in range(1, num_turns + 1):
        achieved = sum(1 for ct in completion_turns.values() if ct is not None and ct <= turn)
        curve.append(achieved / total if total else 0.0)
    return curve


def _diagnose(
    curve: list[float],
    subgoal_status: list[dict[str, Any]],
) -> dict[str, Any]:
    """TED-style automated error categorization from the progress curve."""
    unachieved = [s for s in subgoal_status if not s["achieved"]]
    final_progress = curve[-1] if curve else 0.0

    # "Stalled": the task was left incomplete AND progress did not advance over
    # the trailing third of the turns — the agent kept talking without making
    # further task progress. Only meaningful when the goal was not reached.
    stalled = False
    if final_progress < 1.0 and len(curve) >= 2:
        tail = max(1, len(curve) // 3)
        if len(curve) > tail:
            stalled = curve[-1] == curve[-1 - tail]

    return {
        "achieved_count": len(subgoal_status) - len(unachieved),
        "unachieved_count": len(unachieved),
        "unachieved_subgoals": [s["description"] for s in unachieved],
        "incomplete": final_progress < 1.0,
        "stalled": stalled,
    }


@register_metric
class SubgoalProgressMetric(TextJudgeMetric):
    """Per-turn subgoal progress with PPT, progress-curve AUC, and error analysis.

    The headline ``normalized_score`` is the progress-curve AUC (rewards early
    completion); ``score`` is the terminal ``final_progress`` (fraction of
    subgoals accomplished). Both are on a 0–1 scale where higher is better.
    """

    name = "subgoal_progress"
    version = "v0.1"
    description = (
        "TED-style per-turn subgoal progress: progress-curve AUC, final progress, PPT, and error categorization"
    )
    category = "accuracy"
    metric_type = MetricType.TEXT_JUDGE
    pass_at_k_threshold = 0.5

    async def compute(self, context: MetricContext) -> MetricScore:
        """Score per-turn task progress for a single conversation record."""
        try:
            num_turns = context.num_assistant_turns
            if num_turns == 0:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No assistant turns to evaluate",
                )

            transcript_text = format_transcript_with_tools(context.conversation_trace)
            if not transcript_text:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="No transcript available",
                )

            subgoals = _derive_subgoals(context)
            subgoal_list = "\n".join(f"- {s['id']}: {s['description']}" for s in subgoals)
            prompt = self.get_judge_prompt(
                user_goal=context.user_goal,
                num_assistant_turns=num_turns,
                subgoal_list=subgoal_list or "(no subgoals)",
                conversation_trace=transcript_text,
            )

            response, raw_response = await self.call_judge(prompt, context)
            if response is None:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    normalized_score=0.0,
                    error="Failed to parse judge response",
                    details={"judge_prompt": prompt, "judge_raw_response": raw_response},
                )

            curve, subgoal_status, final_progress, progress_auc, ppt = self._aggregate(response, subgoals, num_turns)
            error_analysis = _diagnose(curve, subgoal_status)

            details: dict[str, Any] = {
                "num_turns": num_turns,
                "num_evaluated": num_turns,
                "num_subgoals": len(subgoals),
                "num_assistant_turns": num_turns,
                "progress_curve": curve,
                "progress_auc": progress_auc,
                "final_progress": final_progress,
                "progress_per_turn": ppt,
                "subgoal_status": subgoal_status,
                "error_analysis": error_analysis,
                "judge_prompt": prompt,
                "judge_raw_response": raw_response,
            }

            return MetricScore(
                name=self.name,
                score=final_progress,
                normalized_score=progress_auc,
                details=details,
                sub_metrics=self._build_sub_metrics(progress_auc, final_progress, ppt),
            )
        except Exception as e:  # noqa: BLE001  (standard metric error envelope)
            return self._handle_error(e, context)

    def _aggregate(
        self,
        response: dict[str, Any],
        subgoals: list[dict[str, str]],
        num_turns: int,
    ) -> tuple[list[float], list[dict[str, Any]], float, float, float]:
        """Map the judge's per-subgoal completions to the progress aggregates."""
        by_id: dict[str, int | None] = {s["id"]: None for s in subgoals}
        for item in response.get("subgoals", []) or []:
            if not isinstance(item, dict):
                continue
            sub_id = item.get("id")
            if sub_id not in by_id:
                continue
            raw_turn = item.get("completion_turn")
            completion: int | None
            if isinstance(raw_turn, bool) or raw_turn is None:
                completion = None
            else:
                try:
                    completion = int(raw_turn)
                except (TypeError, ValueError):
                    completion = None
            # Clamp out-of-range turns to the last turn (judge said "completed").
            if completion is not None and completion < 1:
                completion = None
            by_id[sub_id] = completion

        total = len(subgoals)
        curve = _build_progress_curve(by_id, total, num_turns)
        final_progress = curve[-1] if curve else 0.0
        progress_auc = round(sum(curve) / len(curve), 4) if curve else 0.0
        ppt = round(final_progress / num_turns, 4) if num_turns else 0.0

        subgoal_status: list[dict[str, Any]] = []
        for subgoal in subgoals:
            ct = by_id[subgoal["id"]]
            subgoal_status.append(
                {
                    "id": subgoal["id"],
                    "description": subgoal["description"],
                    "completion_turn": ct,
                    "achieved": ct is not None,
                }
            )
        return curve, subgoal_status, final_progress, progress_auc, ppt

    def _build_sub_metrics(
        self,
        progress_auc: float,
        final_progress: float,
        ppt: float,
    ) -> dict[str, MetricScore]:
        """Surface TED's headline aggregates as sub-metrics for per-metric breakdowns."""
        entries = {
            "progress_auc": progress_auc,
            "final_progress": final_progress,
            "progress_per_turn": ppt,
        }
        return {
            key: MetricScore(name=f"{self.name}.{key}", score=value, normalized_score=value)
            for key, value in entries.items()
        }
