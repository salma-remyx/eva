# Rank-Stability Audit

This pipeline audits how much confidence a ranked comparison of eva runs
deserves. The leaderboard (and any cross-run table) ranks systems by run-level
aggregate means with bootstrap CIs on the mean; those CIs say where a score
lies, not whether the ranking survives resampling the scenarios it was measured
on. Here the shared scenario set is jointly bootstrapped (one draw per replicate
shared by every system, preserving the paired design) and each row is scored by
how often it holds its observed rank.

Adapted from ["How Reproducible Are Evaluation Conclusions? A Self-Audit of
LLM-Inferred Prompt Structure"](https://arxiv.org/abs/2609.30074), which
recommends reporting rank stability, per-cell provenance, and a measurement
date alongside any ranking. Its own audit found the bottom of a ranking firm
(86-99% of replicates) while the middle held rank in only 27-48% — a table can
identify the worst system reliably without reliably identifying the best.

## Files

| File | Role |
|------|------|
| `run_rank_stability.py` | Loads each run's `records/<id>/metrics.json`, rebuilds the per-scenario composite means (same grouping as the run-level aggregates), runs the joint bootstrap, and writes the audit CSV. |

Pure computation lives in `src/eva/utils/rank_stability.py`; CIs and seeds come
from `eva.utils.bootstrap`, so results match the leaderboard metrics and are
deterministic per seed.

## How to run

```bash
uv run python analysis/rank_stability/run_rank_stability.py
```

## Configuration

The script is driven by a local, gitignored YAML config at
`local/rank_stability/rank_stability_config.yaml`:

```yaml
metric: EVA-overall_mean   # composite to rank on
n_bootstrap: 2000
random_seed: 42
output_dir: output_processed/rank_stability
runs:                      # display label -> run dir with records/<id>/metrics.json
  system-a: output/system-a
  system-b: output/system-b
```

Output columns: observed `mean` with its run-level CI, `rank`, `rank_hold_share`
(share of replicates holding that rank), `share_rank_1`, `share_at_or_better`,
bootstrap rank CI, and per-cell provenance (`n_units`, `n_missing_units`,
`measurement_date`).
