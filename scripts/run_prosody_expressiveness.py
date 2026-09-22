"""Run the opt-in prosody_expressiveness metric on an existing benchmark run.

``prosody_expressiveness`` is a diagnostic audio-judge metric excluded from the
default metric set. Importing its module registers it in the global metric
registry (the same mechanism every stock metric uses); this script then runs it
through the standard MetricsRunner, so results land in each record's
metrics.json and in the run's metrics_summary.json with the usual aggregation,
sub-metrics, and bootstrap confidence intervals.

Usage:
    PYTHONPATH=src python scripts/run_prosody_expressiveness.py \\
        --run-dir output/<run_id> [--records 1.1.1,2.1.3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import eva.metrics  # noqa: F401  -- triggers registration of the stock metrics
from eva.metrics.diagnostic.prosody_expressiveness import ProsodyExpressivenessMetric
from eva.metrics.runner import MetricsRunner
from eva.models.record import EvaluationRecord
from eva.utils import router

METRIC_NAME = ProsodyExpressivenessMetric.name


async def amain(args: argparse.Namespace) -> int:
    """Load the run's dataset, run the metric, and print a per-record summary."""
    run_dir = Path(args.run_dir).resolve()

    config = json.loads((run_dir / "config.json").read_text())
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    if "EVA_LANGUAGE" not in os.environ and config.get("language"):
        os.environ["EVA_LANGUAGE"] = config["language"]
        print(f"Set EVA_LANGUAGE={os.environ['EVA_LANGUAGE']} from run config")

    records = EvaluationRecord.load_dataset(dataset_path)

    record_ids = args.records.split(",") if args.records else None
    runner_obj = MetricsRunner(
        run_dir=run_dir,
        dataset=records,
        metric_names=[METRIC_NAME],
        record_ids=record_ids,
        force_rerun=True,
    )
    result = await runner_obj.run()

    print(f"\nDone. {result.total_records} record(s) evaluated.")
    collapse_key = f"{METRIC_NAME}.dimension_collapse_rate"
    for record_id, record_metrics in sorted(result.all_metrics.items()):
        score = record_metrics.metrics.get(METRIC_NAME)
        if score is None:
            print(f"  {record_id}: not run")
            continue
        if score.error is not None:
            print(f"  {record_id}: error: {score.error}")
            continue
        collapse = (score.sub_metrics or {}).get(collapse_key)
        collapse_str = f"{collapse.score:.2f}" if collapse and collapse.score is not None else "n/a"
        print(f"  {record_id}: prosody={score.normalized_score} dimension_collapse_rate={collapse_str}")
    print(f"Metrics written under: {run_dir / 'records'}")
    return 0


def main() -> int:
    """Parse arguments and run the metric."""
    load_dotenv()
    model_list_env = os.getenv("EVA_MODEL_LIST")
    if model_list_env:
        router.init(json.loads(model_list_env))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Path to output/<run_id>")
    ap.add_argument("--records", help="Comma-separated record IDs to evaluate (default: all)")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
