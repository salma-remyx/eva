"""Cross-condition diagnostics for disclosure-sharding runs.

Consumes the per-trial long-format scores written by an evaluation run over a
disclosure-condition dataset (see ``make_condition_dataset.py``) and reports,
per system and metric:

- per-condition means with bootstrap confidence intervals, and
- paired cross-condition deltas, chief among them ``sharded - concat`` — the
  incremental-disclosure cost measured by SCB (arXiv:2609.40198) — plus
``full - concat``, which isolates sensitivity to goal reformulation.

Confidence intervals reuse ``eva.utils.bootstrap`` (percentile bootstrap on
scenario-level means, seeded via ``run_seed``) so results match the
framework's other reported statistics and are reproducible across runs.

Run from project root:
    uv run python analysis/disclosure_sharding/condition_deltas.py \
        --input output/disclosure/trial_scores.csv --output-dir output_processed/disclosure
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eva.utils.bootstrap import N_BOOT, ALPHA, bootstrap_ci, run_seed

try:
    from analysis.disclosure_sharding.disclosure_sharding import CONDITION_SEP, parse_condition
except ImportError:  # executed directly from analysis/disclosure_sharding/
    from disclosure_sharding import CONDITION_SEP, parse_condition

DEFAULT_PAIRS = (("sharded", "concat"), ("full", "concat"))
"""Condition pairs to compare: (delivered, baseline)."""

_ID_COLUMN_CANDIDATES = ("scenario_id", "record_id", "id")
_ALIAS_COLUMN_CANDIDATES = ("system_alias", "alias", "system", "model")

SUMMARY_COLUMNS = [
    "system_alias",
    "metric",
    "condition",
    "n_scenarios",
    "n_trials",
    "mean",
    "ci_low",
    "ci_high",
]
DELTA_COLUMNS = [
    "system_alias",
    "metric",
    "condition_a",
    "condition_b",
    "n_pairs",
    "mean_delta",
    "delta_ci_low",
    "delta_ci_high",
]


def normalize_rows(
    records: list[dict[str, Any]],
    *,
    id_column: str | None = None,
    alias_column: str | None = None,
) -> list[dict[str, Any]]:
    """Map heterogeneous trial-score rows onto the canonical keys.

    Args:
        records: Rows as read from a ``trial_scores`` CSV (any dict-like
            records with per-trial metric values).
        id_column: Column holding the record id; auto-detected when omitted.
        alias_column: Column holding the system alias; auto-detected when
            omitted.

    Returns:
        Rows with ``system_alias``, ``scenario_id``, ``metric``, ``trial`` and
        ``value`` keys and coerced types. Rows whose id carries no known
        ``::condition`` suffix are dropped (they are not part of a
        disclosure-sharding run).

    Raises:
        ValueError: If no candidate id/alias column is present.
    """
    if not records:
        return []

    def pick(candidates: tuple[str, ...], explicit: str | None, label: str) -> str:
        """Return the explicit column name or the first candidate present."""
        if explicit:
            return explicit
        for column in candidates:
            if column in records[0]:
                return column
        raise ValueError(f"No {label} column found; expected one of {candidates}")

    id_col = pick(_ID_COLUMN_CANDIDATES, id_column, "record id")
    alias_col = pick(_ALIAS_COLUMN_CANDIDATES, alias_column, "system alias")

    rows: list[dict[str, Any]] = []
    for record in records:
        scenario_id = str(record[id_col])
        _, condition = parse_condition(scenario_id)
        if condition is None:
            continue
        rows.append(
            {
                "system_alias": str(record[alias_col]),
                "scenario_id": scenario_id,
                "condition": condition,
                "metric": str(record["metric"]),
                "trial": int(record["trial"]),
                "value": float(record["value"]),
            }
        )
    return rows


def _scenario_means(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], float]:
    """Mean metric value per (alias, metric, base record, condition) over trials."""
    trials: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        base_id, condition = parse_condition(row["scenario_id"])
        trials[(row["system_alias"], row["metric"], base_id, condition)].append(row["value"])
    return {key: sum(values) / len(values) for key, values in trials.items()}


def _filter_metrics(rows: list[dict[str, Any]], metrics: list[str] | None) -> list[dict[str, Any]]:
    """Keep only rows for the requested metrics, or all rows when unset."""
    if not metrics:
        return rows
    wanted = set(metrics)
    return [row for row in rows if row["metric"] in wanted]


def summarize_conditions(
    rows: list[dict[str, Any]],
    *,
    metrics: list[str] | None = None,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
) -> list[dict[str, Any]]:
    """Per-condition means with bootstrap CIs over scenario-level means.

    Args:
        rows: Canonical rows from :func:`normalize_rows`.
        metrics: Restrict the summary to these metric names.
        n_boot: Number of bootstrap resamples.
        alpha: CI significance level.

    Returns:
        One row per (system_alias, metric, condition) with columns
        :data:`SUMMARY_COLUMNS`.
    """
    scenario_means = _scenario_means(_filter_metrics(rows, metrics))
    trials_per_condition: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in _filter_metrics(rows, metrics):
        trials_per_condition[(row["system_alias"], row["metric"], row["condition"])] += 1

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (alias, metric, _base, condition), mean in scenario_means.items():
        grouped[(alias, metric, condition)].append(mean)

    summary: list[dict[str, Any]] = []
    for (alias, metric, condition) in sorted(grouped):
        values = grouped[(alias, metric, condition)]
        ci_low, ci_high = bootstrap_ci(
            np.asarray(values, dtype=float),
            n_boot=n_boot,
            seed=run_seed(f"disclosure:{alias}:{metric}:{condition}"),
            alpha=alpha,
        )
        summary.append(
            {
                "system_alias": alias,
                "metric": metric,
                "condition": condition,
                "n_scenarios": len(values),
                "n_trials": trials_per_condition[(alias, metric, condition)],
                "mean": sum(values) / len(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return summary


def cross_condition_deltas(
    rows: list[dict[str, Any]],
    *,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
    metrics: list[str] | None = None,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
) -> list[dict[str, Any]]:
    """Paired cross-condition deltas over shared scenario records.

    For each pair ``(a, b)`` and each scenario record scored under both
    conditions, the delta is ``mean_a - mean_b`` over trials; the reported
    delta is the mean over scenarios with a bootstrap CI. This is the paired
    design the SCB protocol uses to attribute the ``sharded`` accuracy drop to
    incremental disclosure rather than to problem reformulation.

    Args:
        rows: Canonical rows from :func:`normalize_rows`.
        pairs: Condition pairs to compare as ``(delivered, baseline)``.
        metrics: Restrict the deltas to these metric names.
        n_boot: Number of bootstrap resamples.
        alpha: CI significance level.

    Returns:
        One row per (system_alias, metric, pair) with columns
        :data:`DELTA_COLUMNS`.
    """
    scenario_means = _scenario_means(_filter_metrics(rows, metrics))

    deltas: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (alias, metric, base, condition), mean in scenario_means.items():
        for condition_a, condition_b in pairs:
            baseline = scenario_means.get((alias, metric, base, condition_b))
            if condition == condition_a and baseline is not None:
                deltas[(alias, metric, condition_a, condition_b)].append(mean - baseline)

    results: list[dict[str, Any]] = []
    for (alias, metric, condition_a, condition_b) in sorted(deltas):
        values = deltas[(alias, metric, condition_a, condition_b)]
        ci_low, ci_high = bootstrap_ci(
            np.asarray(values, dtype=float),
            n_boot=n_boot,
            seed=run_seed(f"disclosure-delta:{alias}:{metric}:{condition_a}:{condition_b}"),
            alpha=alpha,
        )
        results.append(
            {
                "system_alias": alias,
                "metric": metric,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_pairs": len(values),
                "mean_delta": sum(values) / len(values),
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
            }
        )
    return results


def _print_table(title: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Print rows as an aligned table with a header."""
    print(f"\n{title}")
    if not rows:
        print("  (no rows)")
        return
    widths = {column: max(len(column), *(len(f"{row[column]}") for row in rows)) for column in columns}
    print("  " + "  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print("  " + "  ".join(f"{row[column]}".ljust(widths[column]) for column in columns))


def main() -> None:
    """Read a trial-scores CSV and write per-condition and cross-condition tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="trial_scores.csv from a condition run")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for the output CSVs")
    parser.add_argument("--metrics", nargs="*", default=None, help="Restrict to these metric names")
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="Bootstrap resample count")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="CI significance level")
    args = parser.parse_args()

    import pandas as pd  # local: keeps the analysis core importable without the apps extra

    raw = pd.read_csv(args.input).to_dict("records")
    rows = normalize_rows(raw)
    dropped = len(raw) - len(rows)
    if dropped:
        print(f"Skipped {dropped} row(s) without a known '::{CONDITION_SEP}' condition suffix")

    summary = summarize_conditions(rows, metrics=args.metrics, n_boot=args.n_boot, alpha=args.alpha)
    deltas = cross_condition_deltas(rows, metrics=args.metrics, n_boot=args.n_boot, alpha=args.alpha)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary, columns=SUMMARY_COLUMNS).to_csv(args.output_dir / "condition_summary.csv", index=False)
    pd.DataFrame(deltas, columns=DELTA_COLUMNS).to_csv(args.output_dir / "condition_deltas.csv", index=False)

    _print_table("Per-condition means (bootstrap CI)", SUMMARY_COLUMNS, summary)
    _print_table("Cross-condition deltas (bootstrap CI)", DELTA_COLUMNS, deltas)
    print(f"\nWrote condition_summary.csv and condition_deltas.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
