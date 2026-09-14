"""Reliability-inclusive scoring for EVA composite metrics.

The default aggregation in :mod:`eva.metrics.aggregation` is
reliability-*exclusive*: when a composite's component metric errored, the
composite collapses to ``None`` and the record silently leaves the run-level
denominator. Route failures therefore shrink the EVA-A / EVA-X denominators,
which can flip model orderings without any trace in the reported means.

This module implements the reliability-*inclusive* first-pass scoring rule of
IB2 ("IBIB: A Protocol for Measuring Enterprise AI Systems by Serving Route,
Not Model Identifier", arXiv:2609.10494): keep failure in the score while
keeping unsupported capability out. Mapped onto EVA's ``MetricScore``:

- ``error`` set     -> failure (component scores 0.0, record stays in the
  composite denominator instead of being dropped);
- ``skipped`` set or no score -> unsupported (excluded from the composite);
- metric absent     -> unsupported (e.g. pipeline-incompatible metrics are
  never recorded for a record).

The exclusive aggregates stay the source of truth for ``aggregate_metrics``;
the inclusive variants are reported alongside them so readers can see whether
reliability inclusion changes a conclusion rather than just its wording.
"""

import math
from typing import Any, Literal

from eva.metrics.aggregation import EVA_COMPOSITES, EVACompositeDefinition, _check_threshold
from eva.models.results import MetricScore, RecordMetrics
from eva.utils.bootstrap import mean_ci_fields
from eva.utils.pass_at_k import parse_trial_record_id

ComponentOutcome = Literal["supported", "failed", "unsupported"]


def classify_component(metric: MetricScore | None) -> ComponentOutcome:
    """Classify a composite component as supported, failed, or unsupported.

    The classification table from IB2 adapted to ``MetricScore``: a metric the
    route attempted but errored on is a *failure*; a metric with no applicable
    data (``skipped``) or not recorded for the record is *unsupported*.
    """
    if metric is None:
        return "unsupported"
    if metric.error is not None:
        return "failed"
    if metric.skipped or (metric.normalized_score is None and metric.score is None):
        return "unsupported"
    return "supported"


def _component_value(metric: MetricScore) -> float:
    """Normalized score with fallback to raw score (mirrors ``get_score`` for supported metrics)."""
    value = metric.normalized_score if metric.normalized_score is not None else metric.score
    return float(value)


def reliability_inclusive_aggregates(
    record_metrics: RecordMetrics,
    composites: list[EVACompositeDefinition] | None = None,
) -> dict[str, float | None]:
    """Compute EVA composite scores with route failures kept in the score.

    Unlike :func:`eva.metrics.aggregation.compute_record_aggregates`, a
    component with ``error`` set does not collapse the composite to ``None``:
    the record counts as a failure (0.0) and stays in run-level denominators.
    Skipped and absent components remain excluded (unsupported capability out).

    Returns:
        Dict mapping composite name to score (1.0/0.0 for pass, float for
        mean, None when the composite has no measurable component).
    """
    composites = composites or EVA_COMPOSITES
    results: dict[str, float | None] = {}

    for comp in composites:
        if comp.aggregation_type == "pass":
            has_failure = False
            checks: list[tuple[float, str, float]] = []
            for metric_name in comp.component_metrics:
                metric = record_metrics.metrics.get(metric_name)
                outcome = classify_component(metric)
                if outcome == "failed":
                    has_failure = True
                elif outcome == "supported" and metric is not None:
                    op, threshold = comp.thresholds[metric_name]
                    checks.append((_component_value(metric), op, threshold))

            if has_failure:
                # A required component errored: the failure stays in the score.
                results[comp.name] = 0.0
            elif not checks:
                results[comp.name] = None
            else:
                all_pass = all(_check_threshold(v, op, th) for v, op, th in checks)
                results[comp.name] = 1.0 if all_pass else 0.0

        elif comp.aggregation_type == "mean":
            total = 0.0
            n_components = 0
            for metric_name in comp.component_metrics:
                outcome = classify_component(record_metrics.metrics.get(metric_name))
                if outcome == "failed":
                    # Failure contributes 0.0 but keeps the denominator.
                    n_components += 1
                elif outcome == "unsupported":
                    continue
                else:
                    total += _component_value(record_metrics.metrics[metric_name])
                    n_components += 1
            results[comp.name] = total / n_components if n_components else None

        else:  # derived
            prereqs = [results.get(name) for name in comp.derived_from]
            if any(p is None for p in prereqs):
                results[comp.name] = None
            else:
                results[comp.name] = 1.0 if all(math.isclose(p, 1.0, abs_tol=1e-9) for p in prereqs) else 0.0

    return results


def _scenario_means(per_record_values: dict[str, float | None]) -> list[float]:
    """Group per-record values by base scenario id and return per-scenario means.

    Same trial-collapsing as ``eva.metrics.aggregation._scenario_means``;
    scenarios with no measurable records are dropped.
    """
    grouped: dict[str, list[float]] = {}
    for record_id, val in per_record_values.items():
        if val is None:
            continue
        base_id, _ = parse_trial_record_id(record_id)
        grouped.setdefault(base_id, []).append(float(val))
    return [sum(values) / len(values) for values in grouped.values()]


def compute_reliability_report(
    all_metrics: dict[str, RecordMetrics],
    composites: list[EVACompositeDefinition] | None = None,
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare reliability-inclusive scoring against the default exclusive aggregates.

    For each composite this reports the inclusive mean (metric failures kept
    in the denominator) next to the exclusive mean already stored in
    ``aggregate_metrics``, how many failures the exclusive mode silently
    dropped from the denominator, per-component failure/unsupported counts,
    and a bootstrap CI on the inclusive scenario means (same seeding scheme as
    the composite CIs in ``compute_run_level_aggregates``).

    Args:
        all_metrics: Dict mapping record ID to RecordMetrics (with
            ``aggregate_metrics`` populated by the exclusive path).
        composites: Custom composite definitions. Defaults to EVA_COMPOSITES.
        seed: Bootstrap seed for CI computation.

    Returns:
        Dict mapping composite name to its reliability entry, ready for
        serialization into ``metrics_summary.json``.
    """
    composites = composites or EVA_COMPOSITES
    inclusive_by_record = {
        record_id: reliability_inclusive_aggregates(rm, composites) for record_id, rm in all_metrics.items()
    }

    report: dict[str, Any] = {}
    for comp in composites:
        failed_counts: dict[str, int] = {}
        unsupported_counts: dict[str, int] = {}
        for record_metrics in all_metrics.values():
            for metric_name in comp.component_metrics:
                outcome = classify_component(record_metrics.metrics.get(metric_name))
                if outcome == "failed":
                    failed_counts[metric_name] = failed_counts.get(metric_name, 0) + 1
                elif outcome == "unsupported":
                    unsupported_counts[metric_name] = unsupported_counts.get(metric_name, 0) + 1

        inclusive_values = {rid: values[comp.name] for rid, values in inclusive_by_record.items()}
        inclusive = [v for v in inclusive_values.values() if v is not None]
        exclusive = [v for v in (rm.aggregate_metrics.get(comp.name) for rm in all_metrics.values()) if v is not None]
        inclusive_mean = sum(inclusive) / len(inclusive) if inclusive else None
        exclusive_mean = sum(exclusive) / len(exclusive) if exclusive else None

        # Records the exclusive mode dropped (composite None) that inclusive
        # scoring counts as failures — the silent denominator shrinkage.
        silently_dropped = sum(
            1
            for rid, v in inclusive_values.items()
            if v == 0.0 and all_metrics[rid].aggregate_metrics.get(comp.name) is None
        )

        entry: dict[str, Any] = {
            "total_records": len(all_metrics),
            "inclusive_mean": round(inclusive_mean, 4) if inclusive_mean is not None else None,
            "exclusive_mean": round(exclusive_mean, 4) if exclusive_mean is not None else None,
            "delta": (
                round(inclusive_mean - exclusive_mean, 4)
                if inclusive_mean is not None and exclusive_mean is not None
                else None
            ),
            "silently_dropped_failures": silently_dropped,
            "failed_component_records": dict(sorted(failed_counts.items())),
            "unsupported_component_records": dict(sorted(unsupported_counts.items())),
        }
        entry.update(mean_ci_fields(_scenario_means(inclusive_values), seed=seed))
        report[comp.name] = entry

    return report
