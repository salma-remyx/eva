# Config: local/rank_stability/rank_stability_config.yaml
#
# metric: EVA-overall_mean
# n_bootstrap: 2000
# random_seed: 42
# output_dir: output_processed/rank_stability
# runs:
#   <display_label>: <run dir containing records/<record_id>/metrics.json>

"""Rank-stability audit of a set of eva runs.

Loads each run's per-record ``metrics.json``, rebuilds the per-scenario
composite means that feed the run-level bootstrap CI, and answers the question
that CI does not: how often does each system hold its leaderboard rank when the
shared scenario set is jointly resampled? Rows are reported with the mean CI
beside the rank-stability columns, plus per-cell provenance (scenario counts,
missing scenarios, measurement date).

Adapted from "How Reproducible Are Evaluation Conclusions? A Self-Audit of
LLM-Inferred Prompt Structure" (arXiv:2609.30074). Pure computation lives in
eva.utils.rank_stability; this module only loads run artifacts, joins them with
the run-level aggregates, and writes the CSV.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from eva.metrics.aggregation import compute_run_level_aggregates
from eva.models.results import RecordMetrics
from eva.utils.bootstrap import N_BOOT, run_seed
from eva.utils.pass_at_k import parse_trial_record_id
from eva.utils.rank_stability import FIRM_RANK_HOLD, rank_stability_table, stability_summary

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "rank_stability" / "rank_stability_config.yaml"

CSV_COLUMNS = [
    "system",
    "mean",
    "mean_ci_lower",
    "mean_ci_upper",
    "rank",
    "rank_hold_share",
    "share_rank_1",
    "share_at_or_better",
    "mean_bootstrap_rank",
    "rank_ci_lower",
    "rank_ci_upper",
    "n_units",
    "n_missing_units",
    "firm",
    "measurement_date",
]


def load_run_metrics(run_dir: Path) -> dict[str, RecordMetrics]:
    """Read ``<run_dir>/records/<record_id>/metrics.json`` into RecordMetrics."""
    records_dir = run_dir / "records"
    if not records_dir.is_dir():
        raise FileNotFoundError(f"Records directory not found: {records_dir}")
    run_metrics: dict[str, RecordMetrics] = {}
    for record_dir in sorted(records_dir.iterdir()):
        metrics_path = record_dir / "metrics.json"
        if metrics_path.is_file():
            run_metrics[record_dir.name] = RecordMetrics.model_validate_json(metrics_path.read_text())
    return run_metrics


def per_scenario_composite_values(run_metrics: dict[str, RecordMetrics], composite: str) -> dict[str, float]:
    """Per-scenario mean over trials of ``composite``, keyed by base scenario id.

    Same grouping the run-level aggregates use: these are the per-scenario means
    whose mean the leaderboard ranks and whose CI eva.utils.bootstrap reports.
    """
    grouped: dict[str, list[float]] = {}
    for record_id, record_metrics in run_metrics.items():
        base_id, _ = parse_trial_record_id(record_id)
        value = record_metrics.aggregate_metrics.get(composite)
        if value is not None:
            grouped.setdefault(base_id, []).append(float(value))
    return {base_id: sum(values) / len(values) for base_id, values in grouped.items()}


def measurement_date(run_dir: Path) -> str:
    """Date the run's metrics were last (re)computed, as the paper recommends reporting.

    Falls back to the run directory's own mtime when no metrics_summary.json exists.
    """
    summary_path = run_dir / "metrics_summary.json"
    stamp = summary_path.stat().st_mtime if summary_path.is_file() else run_dir.stat().st_mtime
    return datetime.fromtimestamp(stamp).date().isoformat()


def build_stability_rows(
    runs: dict[str, Path],
    metric: str,
    *,
    n_boot: int = N_BOOT,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per system: observed rank/mean/CI beside rank stability and provenance.

    The mean CI comes from compute_run_level_aggregates so the two confidence
    statements are read side by side — a CI on the mean does not say whether the
    ranking itself is firm.

    Args:
        runs: display label -> run dir with per-record metrics.json artifacts.
        metric: Composite to rank on (e.g. EVA-overall_mean).
        n_boot: Number of joint bootstrap replicates.
        seed: Seed for the joint bootstrap.

    Returns:
        (DataFrame with CSV_COLUMNS rows ordered by observed rank, summary dict
        from stability_summary).
    """
    per_unit_values: dict[str, dict[str, float]] = {}
    run_level: dict[str, dict[str, Any]] = {}
    dates: dict[str, str] = {}
    for label, run_dir in runs.items():
        run_metrics = load_run_metrics(Path(run_dir))
        per_unit_values[label] = per_scenario_composite_values(run_metrics, metric)
        aggregates = compute_run_level_aggregates(run_metrics, seed=run_seed(Path(run_dir).name))
        run_level[label] = aggregates.get(metric, {})
        dates[label] = measurement_date(Path(run_dir))

    table = rank_stability_table(per_unit_values, seed=seed, n_boot=n_boot)
    rows = [
        {
            "system": label,
            **row,
            "mean_ci_lower": run_level[label].get("mean_ci_lower"),
            "mean_ci_upper": run_level[label].get("mean_ci_upper"),
            "measurement_date": dates[label],
        }
        for label, row in table.items()
    ]
    return pd.DataFrame(rows, columns=CSV_COLUMNS), stability_summary(table)


def main(config_path: Path = CONFIG_PATH) -> None:
    """Load the config, audit the configured runs, and write the stability CSV."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}. See this file's header for the expected shape.")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    metric: str = config["metric"]
    runs: dict[str, str] = config["runs"]
    seed: int = config.get("random_seed", 42)
    n_boot: int = config.get("n_bootstrap", N_BOOT)

    results, summary = build_stability_rows(
        {label: Path(run_dir) for label, run_dir in runs.items()},
        metric,
        n_boot=n_boot,
        seed=seed,
    )

    output_dir = PROJECT_ROOT / config.get("output_dir", "output_processed/rank_stability")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank_stability_{metric}.csv"
    results.to_csv(output_path, index=False)

    print(f"Rank stability of {metric}: {summary['n_systems']} systems, {summary['n_units']} scenarios")
    firm_rows = f"{summary['n_firm_rows']}/{summary['n_systems']}"
    print(f"  firm rows (hold rank in >= {FIRM_RANK_HOLD:.0%} of replicates): {firm_rows}")
    print(f"  best  {summary['best_system']}: firm={summary['best_system_firm']}")
    print(f"  worst {summary['worst_system']}: firm={summary['worst_system_firm']}")
    print(f"Wrote {len(results)} rows -> {output_path}")


if __name__ == "__main__":
    main()
