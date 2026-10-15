"""Per-group fairness summaries: group scores, worst group, and the group gap.

Implements the reporting protocol from "Pruning for Efficiency, Paying in Fairness:
Demographic Disparities in Pruned Speech-LLMs" (arXiv:2609.38106): an aggregate
score can hide how a change affects individual speaker groups, so group-level
scores, the worst-performing group, and the best-to-worst gap belong next to the
aggregate — with the worst group's score usable as an explicit selection criterion.

Applied to EVA: given the scenario-level metric values of a run plus a group label
per row (accent, speaker gender, language, ...), produce per-group bootstrap CIs
and a per-condition worst-group/gap summary. CIs come from eva.utils.bootstrap so
they match the other reported intervals and stay deterministic in the run seed.

Pure computation: takes DataFrames, returns DataFrames. No file I/O, no plotting.
"""

import numpy as np
import pandas as pd

from eva.utils.bootstrap import bootstrap_ci, run_seed

# Metric-name fragments that count errors instead of quality. Mirrors the sub-metric
# convention in eva.metrics.utils.direction_for_sub_metric ("_rate" = lower-is-better).
LOWER_IS_BETTER_TOKENS: tuple[str, ...] = ("wer", "cer", "error_rate")

PER_GROUP_COLUMNS: list[str] = [
    "model_label",
    "metric",
    "condition",
    "group",
    "point",
    "ci_lower",
    "ci_upper",
    "n",
]

SUMMARY_COLUMNS: list[str] = [
    "model_label",
    "metric",
    "condition",
    "n_groups",
    "n_scenarios",
    "lower_is_better",
    "aggregate_point",
    "best_group",
    "best_group_point",
    "worst_group",
    "worst_group_point",
    "worst_group_ci_lower",
    "worst_group_ci_upper",
    "group_gap",
    "gap_ci_lower",
    "gap_ci_upper",
    "gap_ci_excludes_zero",
    "worst_group_share",
]


def metric_lower_is_better(metric: str, config: dict | None = None) -> bool:
    """Return True when a lower value of ``metric`` is better.

    Error-rate metrics (stt_wer, sub-metrics ending in ``_rate``) improve downward,
    so their worst-performing group is the one with the *highest* value. Every other
    EVA metric is a quality score whose worst group has the lowest value.

    Args:
        metric: Metric name as it appears in the metric column of the input table.
        config: Optional parsed perturbations config. ``lower_is_better_metrics`` and
            ``higher_is_better_metrics`` override name inference by exact match, in
            that order.
    """
    config = config or {}
    if metric in config.get("higher_is_better_metrics", []):
        return False
    if metric in config.get("lower_is_better_metrics", []):
        return True
    lowered = metric.lower()
    return lowered.endswith("_rate") or any(token in lowered for token in LOWER_IS_BETTER_TOKENS)


def _as_tuple(group_vals: object) -> tuple:
    """Normalize a pandas groupby key to a tuple (single-column groups yield a scalar)."""
    if isinstance(group_vals, tuple):
        return group_vals
    return (group_vals,)


def _labelled(values_long: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Filter to rows that carry a group label. The clean baseline carries none."""
    if group_col not in values_long.columns:
        return values_long.iloc[0:0]
    return values_long[values_long[group_col].notna()]


def group_value_cis(values_long: pd.DataFrame, config: dict, group_col: str = "group") -> pd.DataFrame:
    """Bootstrap CIs on scenario-level metric values per (model, metric, condition, group).

    Pooling matches ``stats_perturbations.metric_value_cis``: a group's values are
    concatenated across domains (scenario-level weighting) and the interval comes
    from ``eva.utils.bootstrap.bootstrap_ci``.

    Args:
        values_long: Scenario-level metric values (the ``scenario_metricvalues.csv``
            shape) plus a ``group_col`` column holding each row's group label.
        config: Parsed perturbations_config.yaml with alpha, n_bootstrap, random_seed.
        group_col: Column holding the group label; rows without one are dropped.

    Returns:
        DataFrame with columns model_label, metric, condition, group, point,
        ci_lower, ci_upper, n.
    """
    alpha: float = config["alpha"]
    n_boot: int = config["n_bootstrap"]
    seed: int = config["random_seed"]

    labelled = _labelled(values_long, group_col)
    if labelled.empty:
        return pd.DataFrame(columns=PER_GROUP_COLUMNS)

    group_keys = ["model_label", "metric", "condition", group_col]
    rows: list[dict] = []
    for group_vals, g in labelled.groupby(group_keys, sort=False):
        model, metric, condition, group = _as_tuple(group_vals)
        values = g["value"].to_numpy(dtype=float)
        cell_seed = run_seed(f"{seed}:gf:{model}:{metric}:{condition}:{group}")
        ci_lower, ci_upper = bootstrap_ci(values, n_boot=n_boot, seed=cell_seed, alpha=alpha)
        rows.append(
            {
                "model_label": model,
                "metric": metric,
                "condition": condition,
                "group": group,
                "point": float(values.mean()),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(values),
            }
        )
    return pd.DataFrame(rows, columns=PER_GROUP_COLUMNS)


def gap_bootstrap_ci(
    group_values: dict[str, np.ndarray],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    lower_is_better: bool,
    worst_group: str,
) -> tuple[float, float, float]:
    """Bootstrap CI on the best-to-worst group gap.

    Each replicate resamples scenarios within every group independently and
    recomputes the gap between that replicate's best and worst group, so the
    interval covers group re-identification — a group can stop being the worst
    across resamples — rather than the spread of two fixed groups.

    Args:
        group_values: Scenario-level values per group name.
        n_boot: Number of bootstrap replicates.
        seed: Deterministic seed for this cell.
        alpha: CI level (0.05 → 95% interval).
        lower_is_better: Metric direction; decides which tail counts as worst.
        worst_group: Point-estimate worst group, used for the stability share.

    Returns:
        (ci_lower, ci_upper, worst_group_share) where worst_group_share is the
        fraction of replicates in which ``worst_group`` is still the worst.
    """
    non_empty = {name: np.asarray(v, dtype=float) for name, v in group_values.items() if len(v) > 0}
    if len(non_empty) < 2:
        return 0.0, 0.0, 1.0

    names = list(non_empty)
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, len(names)), dtype=float)
    for j, name in enumerate(names):
        values = non_empty[name]
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        means[:, j] = values[idx].mean(axis=1)

    worst_j = means.argmax(axis=1) if lower_is_better else means.argmin(axis=1)
    best_j = means.argmin(axis=1) if lower_is_better else means.argmax(axis=1)
    replicate = np.arange(n_boot)
    gaps = np.abs(means[replicate, best_j] - means[replicate, worst_j])

    ci_lower = float(np.percentile(gaps, 100 * alpha / 2))
    ci_upper = float(np.percentile(gaps, 100 * (1 - alpha / 2)))
    worst_group_share = float(np.mean(worst_j == names.index(worst_group)))
    return ci_lower, ci_upper, worst_group_share


def group_fairness_summary(
    values_long: pd.DataFrame,
    config: dict,
    group_col: str = "group",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group scores plus worst-group and group-gap summaries for each condition.

    One row per (model, metric, condition) that has at least two labelled groups,
    pairing every group's point estimate and CI with the worst-performing group, the
    best-to-worst gap, and the pooled aggregate the gap is hidden behind. This is the
    paper's deployment criterion made mechanical: rank systems by their worst group,
    not by the aggregate.

    Args:
        values_long: Scenario-level metric values (the ``scenario_metricvalues.csv``
            shape) plus a ``group_col`` column holding each row's group label. Rows
            without a label (e.g. the clean baseline) are excluded.
        config: Parsed perturbations_config.yaml with alpha, n_bootstrap, random_seed.
        group_col: Column holding the group label.

    Returns:
        (per_group_df, summary_df). ``per_group_df`` has columns
        model_label, metric, condition, group, point, ci_lower, ci_upper, n.
        ``summary_df`` has columns model_label, metric, condition, n_groups,
        n_scenarios, lower_is_better, aggregate_point, best_group, best_group_point,
        worst_group, worst_group_point, worst_group_ci_lower, worst_group_ci_upper,
        group_gap, gap_ci_lower, gap_ci_upper, gap_ci_excludes_zero, worst_group_share.
    """
    per_group = group_value_cis(values_long, config, group_col=group_col)
    if per_group.empty:
        return per_group, pd.DataFrame(columns=SUMMARY_COLUMNS)

    alpha: float = config["alpha"]
    n_boot: int = config["n_bootstrap"]
    seed: int = config["random_seed"]

    labelled = _labelled(values_long, group_col)
    per_group_indexed = per_group.set_index(["model_label", "metric", "condition", "group"])
    rows: list[dict] = []
    for (model, metric, condition), g in labelled.groupby(["model_label", "metric", "condition"], sort=False):
        cell_values = {name: vals.to_numpy(dtype=float) for name, vals in g.groupby(group_col, sort=False)["value"]}
        if len(cell_values) < 2:
            continue  # a single labelled group has no disparity to report

        points = {name: float(values.mean()) for name, values in cell_values.items()}
        lower_better = metric_lower_is_better(metric, config)
        best_group = min(points, key=points.get) if lower_better else max(points, key=points.get)
        worst_group = max(points, key=points.get) if lower_better else min(points, key=points.get)

        cell_seed = run_seed(f"{seed}:gf:{model}:{metric}:{condition}:gap")
        gap_ci_lower, gap_ci_upper, worst_group_share = gap_bootstrap_ci(
            cell_values,
            n_boot=n_boot,
            seed=cell_seed,
            alpha=alpha,
            lower_is_better=lower_better,
            worst_group=worst_group,
        )

        worst = per_group_indexed.loc[(model, metric, condition, worst_group)]

        rows.append(
            {
                "model_label": model,
                "metric": metric,
                "condition": condition,
                "n_groups": len(points),
                "n_scenarios": int(sum(len(values) for values in cell_values.values())),
                "lower_is_better": lower_better,
                # the aggregate a deployment decision would otherwise stop at
                "aggregate_point": float(g["value"].to_numpy(dtype=float).mean()),
                "best_group": best_group,
                "best_group_point": points[best_group],
                "worst_group": worst_group,
                "worst_group_point": points[worst_group],
                "worst_group_ci_lower": float(worst["ci_lower"]),
                "worst_group_ci_upper": float(worst["ci_upper"]),
                "group_gap": abs(points[best_group] - points[worst_group]),
                "gap_ci_lower": gap_ci_lower,
                "gap_ci_upper": gap_ci_upper,
                "gap_ci_excludes_zero": bool(gap_ci_lower > 0.0),
                "worst_group_share": worst_group_share,
            }
        )

    return per_group, pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
