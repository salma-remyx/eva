"""Build a disclosure-condition dataset from an existing evaluation dataset.

Reads a dataset of ``EvaluationRecord``s, shards each record's
``high_level_user_goal``, and writes one variant per disclosure condition
(``full``/``concat``/``sharded``) with ``::<condition>`` id suffixes. The
output is a drop-in dataset for the regular benchmark run: same personas,
decision trees, and ground truths, so per-condition scores are directly
comparable (see ``condition_deltas.py`` for the cross-condition analysis).

Run from project root:
    uv run python analysis/disclosure_sharding/make_condition_dataset.py \
        --input data/airline_dataset.json --output output/disclosure/airline_conditions.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from eva.models.record import EvaluationRecord

try:
    from analysis.disclosure_sharding.disclosure_sharding import (
        CONDITIONS,
        DEFAULT_MAX_SHARDS,
        condition_variants,
        goal_shards,
    )
except ImportError:  # executed directly from analysis/disclosure_sharding/
    from disclosure_sharding import (
        CONDITIONS,
        DEFAULT_MAX_SHARDS,
        condition_variants,
        goal_shards,
    )


def build_condition_dataset(
    records: list[EvaluationRecord],
    *,
    conditions: tuple[str, ...] = CONDITIONS,
    max_shards: int = DEFAULT_MAX_SHARDS,
) -> tuple[list[EvaluationRecord], Counter[int], list[str]]:
    """Expand records into disclosure-condition variants.

    Args:
        records: Source dataset records.
        conditions: Which conditions to emit per record.
        max_shards: Passed to :func:`disclosure_sharding.split_into_shards`.

    Returns:
        A tuple of the expanded records, the shard-count histogram, and the
        ids of records that yielded a single shard (and therefore only got a
        ``full`` variant).
    """
    expanded: list[EvaluationRecord] = []
    shard_counts: Counter[int] = Counter()
    single_shard_ids: list[str] = []
    for record in records:
        shards = goal_shards(record, max_shards=max_shards)
        shard_counts[len(shards)] += 1
        if len(shards) < 2:
            single_shard_ids.append(record.id)
        expanded.extend(condition_variants(record, conditions=conditions, max_shards=max_shards))
    return expanded, shard_counts, single_shard_ids


def main() -> None:
    """Expand a dataset into disclosure-condition variants and save them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="Source dataset (.json array of records)")
    parser.add_argument("--output", required=True, type=Path, help="Destination dataset path")
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated disclosure conditions to emit (default: full,concat,sharded)",
    )
    parser.add_argument("--max-shards", type=int, default=DEFAULT_MAX_SHARDS, help="Maximum shards per goal")
    parser.add_argument("--limit", type=int, default=None, help="Only expand the first N records")
    args = parser.parse_args()

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    records = EvaluationRecord.load_dataset(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    expanded, shard_counts, single_shard_ids = build_condition_dataset(
        records, conditions=conditions, max_shards=args.max_shards
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    EvaluationRecord.save_dataset(expanded, args.output)

    histogram = ", ".join(f"{count} shard(s): {n} records" for count, n in sorted(shard_counts.items()))
    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Shard histogram: {histogram or 'none'}")
    if single_shard_ids:
        print(f"WARNING: {len(single_shard_ids)} record(s) yielded a single shard and only got a 'full' variant:")
        print("  " + ", ".join(single_shard_ids))
    print(f"Wrote {len(expanded)} condition records to {args.output}")


if __name__ == "__main__":
    main()
