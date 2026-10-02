"""Compute the opt-in User Fidelity Score on existing benchmark runs.

The ``user_fidelity`` metric grades how faithfully the simulated user
executed its assigned role, independently of agent success (see
docs/metrics/user_fidelity.md). It is not part of the default metric set:
this script registers it, scores every record in the given run directory
(merging into each record's ``metrics.json``), and prints the aggregate
score plus per-criterion violation rates.

Pass ``--run-dir`` once per run to compare user simulators (e.g. an
ElevenLabs Agents run vs an OpenAI Realtime run) on the same dataset —
the summary lines make the fidelity gap between simulators directly
comparable.

Usage:
    PYTHONPATH=src python scripts/compute_user_fidelity.py \\
        --run-dir output/<run_id> \\
        [--run-dir output/<other_run_id>] \\
        [--judge-model gpt-5.2-medium]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import eva.metrics.validation.user_fidelity  # noqa: F401  -- registers the opt-in metric
from eva.metrics.runner import MetricsRunner
from eva.models.record import EvaluationRecord
from eva.utils import router

METRIC_NAME = "user_fidelity"


def _load_dataset(run_dir: Path) -> list[EvaluationRecord]:
    """Load the evaluation dataset a run was produced from."""
    config = json.loads((run_dir / "config.json").read_text())
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    if "EVA_LANGUAGE" not in os.environ and config.get("language"):
        os.environ["EVA_LANGUAGE"] = config["language"]
        print(f"Set EVA_LANGUAGE={os.environ['EVA_LANGUAGE']} from run config")
    return EvaluationRecord.load_dataset(dataset_path)


def _summarize_run(run_dir: Path) -> dict[str, float]:
    """Aggregate user_fidelity scores across a run's records.

    Returns {metric_name: mean} for the parent UFS (key ``user_fidelity``)
    and for each per-criterion ``_rate`` sub-metric (fraction of scored
    records where that criterion was violated). Records whose score carries
    an error are skipped.
    """
    ufs_scores: list[float] = []
    violation_counts: dict[str, int] = {}

    for metrics_path in sorted((run_dir / "records").glob("*/metrics.json")):
        try:
            data = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"Skipping unreadable {metrics_path}")
            continue
        score = (data.get("metrics") or {}).get(METRIC_NAME)
        if not score or score.get("error"):
            continue
        ufs_scores.append(float(score["normalized_score"]))
        for sub_key, sub in (score.get("sub_metrics") or {}).items():
            violation_counts[sub_key] = violation_counts.get(sub_key, 0) + int(sub["score"])

    if not ufs_scores:
        return {}

    summary: dict[str, float] = {METRIC_NAME: sum(ufs_scores) / len(ufs_scores)}
    num_records = len(ufs_scores)
    for sub_key, count in violation_counts.items():
        summary[sub_key] = count / num_records
    return summary


async def amain(args: argparse.Namespace) -> int:
    metric_configs: dict[str, dict[str, Any]] | None = None
    if args.judge_model:
        metric_configs = {METRIC_NAME: {"judge_model": args.judge_model}}

    for run_dir_arg in args.run_dir:
        run_dir = Path(run_dir_arg).resolve()
        records = _load_dataset(run_dir)
        runner_obj = MetricsRunner(
            run_dir=run_dir,
            dataset=records,
            metric_names=[METRIC_NAME],
            metric_configs=metric_configs,
            force_rerun=True,
        )
        result = await runner_obj.run()
        print(f"\n{run_dir}: {result.total_records} record(s) scored, {result.total_metric_failures} metric failure(s)")

        summary = _summarize_run(run_dir)
        if not summary:
            print("  No user_fidelity scores found on disk.")
            continue
        print(f"  User Fidelity Score: {summary.pop(METRIC_NAME):.3f}")
        for sub_key, rate in sorted(summary.items()):
            print(f"  {sub_key}: {rate:5.1%}")

    return 0


def main() -> int:
    load_dotenv()
    model_list_env = os.getenv("EVA_MODEL_LIST")
    if model_list_env:
        router.init(json.loads(model_list_env))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, action="append", help="Path to output/<run_id> (repeatable)")
    ap.add_argument("--judge-model", help="Override the judge model for this run")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
