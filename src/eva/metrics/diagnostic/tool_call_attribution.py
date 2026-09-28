"""Attribute tool-call protocol outcomes to the serving stack vs. the model.

Serving-layer failures (request rejected before inference, retry exhaustion,
parser artifacts) surface in the conversation trace as the generic LLM error
message rather than as tool calls, so a fidelity metric that only looks at
``tool_responses`` reads them as "the model made no calls" — or, worse, as a
perfect score when no calls exist at all. This metric separates the two so a
diagnostic report can say *which layer* failed.

Outcome taxonomy (per conversation record):

- ``valid_call``        — tool call executed without a format/parse error
- ``parse_failure``     — call emitted but not executable (bad name/params)
- ``serving_failure``   — assistant turn produced by the error path, i.e. the
                          serving layer never returned a model completion
- ``model_non_call``    — real assistant reply that simply made no tool call

Both per-instance (per call) and turn-pooled fidelity are reported because the
two estimates diverge when failures cluster inside turns.

Debug metric for diagnosing model vs. infrastructure performance, not directly
used in final evaluation scores.

Adapted from "Measuring the Serving Stack Instead of the Model: Hidden
Confounds in Local Tool-Use Evaluation" (arXiv:2609.26693). The paper probes
live serving stacks (Ollama, llama.cpp, vLLM, SGLang); this port reads the
same outcome taxonomy off eva's recorded conversation traces instead.
"""

from eva.metrics.base import CodeMetric, MetricContext
from eva.metrics.diagnostic.tool_call_validity import CALL_ERROR_TYPES
from eva.metrics.registry import register_metric
from eva.metrics.utils import make_rate_sub_metric
from eva.models.results import MetricScore
from eva.utils.conversation_checks import LLM_GENERIC_ERROR_MESSAGE


def _collect_call_outcomes(conversation_trace: list[dict]) -> dict[int, list[bool]]:
    """Map turn_id -> per-call validity flags, pairing each call with its response.

    A call is *valid* when the tool response that follows it carries no
    ``error_type`` from the harness-side parse/validation taxonomy
    (``CALL_ERROR_TYPES``, shared with ``tool_call_validity``).
    """
    calls_per_turn: dict[int, list[bool]] = {}
    pending_call_turns: list[int] = []

    for entry in conversation_trace:
        entry_type = entry.get("type")
        turn_id = entry.get("turn_id")
        if entry_type == "tool_call":
            pending_call_turns.append(turn_id)
        elif entry_type == "tool_response":
            tool_response = entry.get("tool_response")
            error_type = tool_response.get("error_type", "") if isinstance(tool_response, dict) else ""
            is_valid = error_type not in CALL_ERROR_TYPES
            # Responses follow their calls in trace order; unpaired trailing
            # responses (interrupted calls) are ignored rather than guessed at.
            if pending_call_turns:
                call_turn = pending_call_turns.pop(0)
                calls_per_turn.setdefault(call_turn, []).append(is_valid)

    return calls_per_turn


@register_metric
class ToolCallAttributionMetric(CodeMetric):
    """Separates serving-layer failures from model behavior in tool-call outcomes.

    The parent score is tool-selection fidelity over the attempts the model
    actually got to make (valid calls / parseable calls). Serving failures are
    excluded from the denominator and reported separately, so a record whose
    serving stack rejected every request is marked ``skipped`` — not scored as
    0% (model never called) or 100% (no calls to invalidate) fidelity.

    This is a diagnostic metric used for diagnosing model vs. infrastructure
    performance. It is not directly used in final evaluation scores.
    """

    name = "tool_call_attribution"
    version = "v0.1"
    description = "Debug metric: attribute tool-call failures to the serving layer vs. the model"
    category = "diagnostic"
    exclude_from_pass_at_k = True

    async def compute(self, context: MetricContext) -> MetricScore:
        """Compute per-layer tool-call outcome counts and dual-granularity fidelity."""
        try:
            calls_per_turn = _collect_call_outcomes(context.conversation_trace or [])
            total_calls = sum(len(flags) for flags in calls_per_turn.values())
            valid_calls = sum(1 for flags in calls_per_turn.values() for ok in flags if ok)
            parse_failures = total_calls - valid_calls

            # Assistant turns whose only content came from the error path are the
            # trace-level fingerprint of a serving-layer failure: the pipeline
            # yields the generic apology instead of a model completion.
            serving_failure_turns = {
                turn
                for turn, text in (context.intended_assistant_turns or {}).items()
                if LLM_GENERIC_ERROR_MESSAGE in text
            }
            assistant_turns = set((context.intended_assistant_turns or {}).keys())
            conversation_ended_in_error = context.conversation_ended_reason == "error"
            tool_call_turns = set(calls_per_turn.keys())
            non_call_turns = assistant_turns - tool_call_turns - serving_failure_turns

            # Unmeasurable rather than wrong: serving failures crowded out every
            # attempt the fidelity score could have been computed over.
            confounded = total_calls == 0 and (bool(serving_failure_turns) or conversation_ended_in_error)

            per_instance = valid_calls / total_calls if total_calls else None
            turns_with_calls = [flags for flags in calls_per_turn.values() if flags]
            valid_turns = sum(1 for flags in turns_with_calls if all(flags))
            turn_pooled = valid_turns / len(turns_with_calls) if turns_with_calls else None
            aggregation_gap: float | None = None
            if per_instance is not None and turn_pooled is not None:
                aggregation_gap = abs(turn_pooled - per_instance)

            sub_metrics = self._build_sub_metrics(
                self.name,
                total_calls,
                valid_calls,
                parse_failures,
                len(assistant_turns),
                len(serving_failure_turns),
                turn_pooled,
            )

            details = {
                "total_tool_calls": total_calls,
                "outcome_counts": {
                    "valid_call": valid_calls,
                    "parse_failure": parse_failures,
                    "serving_failure": len(serving_failure_turns),
                    "model_non_call": len(non_call_turns),
                },
                "attribution": {
                    "serving_layer": {
                        "failed_turns": len(serving_failure_turns),
                        "failed_turn_ids": sorted(serving_failure_turns),
                        "conversation_ended_in_error": conversation_ended_in_error,
                    },
                    "model": {"parse_failures": parse_failures, "non_call_turns": len(non_call_turns)},
                },
                "per_instance_fidelity": round(per_instance, 4) if per_instance is not None else None,
                "turn_pooled_fidelity": round(turn_pooled, 4) if turn_pooled is not None else None,
                "aggregation_gap": round(aggregation_gap, 4) if aggregation_gap is not None else None,
                "confounded": confounded,
            }

            if total_calls == 0:
                if confounded:
                    details["note"] = "Serving layer failed before any model attempt; fidelity not measurable."
                else:
                    details["note"] = "No tool calls and no serving failures — the model made no attempts."
                return MetricScore(
                    name=self.name,
                    score=None,
                    normalized_score=None,
                    skipped=True,
                    details=details,
                    sub_metrics=sub_metrics,
                )

            return MetricScore(
                name=self.name,
                score=round(per_instance, 4),
                normalized_score=round(per_instance, 4),
                details=details,
                sub_metrics=sub_metrics,
            )

        except Exception as e:
            return self._handle_error(e, context)

    @staticmethod
    def _build_sub_metrics(
        parent_name: str,
        total_calls: int,
        valid_calls: int,
        parse_failures: int,
        total_assistant_turns: int,
        serving_failure_turns: int,
        turn_pooled: float | None,
    ) -> dict[str, MetricScore]:
        """Build call-granularity rates plus the turn-pooled fidelity estimate.

        ``num_tool_calls`` is a count (normalized_score=None); rate sub-metrics
        use ``make_rate_sub_metric`` so zero denominators never divide.
        """
        sub_metrics: dict[str, MetricScore] = {
            "num_tool_calls": MetricScore(
                name=f"{parent_name}.num_tool_calls",
                score=float(total_calls),
                normalized_score=None,
                details={},
            ),
            "valid_call_rate": make_rate_sub_metric(
                parent_name=parent_name,
                key="valid_call_rate",
                numerator=valid_calls,
                denominator=total_calls,
                details={"count": valid_calls, "total_tool_calls": total_calls},
                precision=4,
            ),
            "parse_failure_rate": make_rate_sub_metric(
                parent_name=parent_name,
                key="parse_failure_rate",
                numerator=parse_failures,
                denominator=total_calls,
                details={"count": parse_failures, "total_tool_calls": total_calls},
                precision=4,
            ),
            "serving_failure_rate": make_rate_sub_metric(
                parent_name=parent_name,
                key="serving_failure_rate",
                numerator=serving_failure_turns,
                denominator=total_assistant_turns,
                details={"count": serving_failure_turns, "total_assistant_turns": total_assistant_turns},
                precision=4,
            ),
        }
        if turn_pooled is not None:
            sub_metrics["turn_pooled_fidelity"] = MetricScore(
                name=f"{parent_name}.turn_pooled_fidelity",
                score=round(turn_pooled, 4),
                normalized_score=round(turn_pooled, 4),
                details={"note": "Fraction of calling turns whose calls are all valid"},
            )
        return sub_metrics
