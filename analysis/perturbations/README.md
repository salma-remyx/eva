# Perturbation Analysis Pipeline

This pipeline computes perturbation deltas (perturbed vs. clean) and per-condition
metric-value confidence intervals from per-trial scores, then writes the
`perturbation_delta` and `metric_values` blocks into
`website/src/data/leaderboardStats.json` for the leaderboard's perturbation charts.

## Files

| File | Role |
|------|------|
| `data_perturbations.py` | Computes per-scenario means, clean-vs-perturbation deltas, and per-condition metric values from raw trial scores. |
| `stats_perturbations.py` | Runs bootstrap CIs and paired sign-flip permutation tests with Holm-Bonferroni correction. CIs use `eva.utils.bootstrap` so they match the leaderboard metrics; both CIs and permutation tests are deterministic via a derived `run_seed`. |
| `run_perturbations.py` | End-to-end driver: calls `data_perturbations` then `stats_perturbations` in sequence. |
| `regenerate_perturbation_blocks.py` | Reads the results CSVs and writes `perturbation_delta` + `metric_values` blocks into `leaderboardStats.json`. Additive and idempotent (only writes new or changed entries). Dry-run by default; pass `--write` to apply. |

## How to run

```bash
# 1. Compute deltas and statistics
uv run python analysis/perturbations/run_perturbations.py

# 2. Preview changes to leaderboardStats.json (dry-run, writes a .regen file)
uv run python analysis/perturbations/regenerate_perturbation_blocks.py

# 2b. Apply changes
uv run python analysis/perturbations/regenerate_perturbation_blocks.py --write
```

## Configuration

The scripts are driven by local, gitignored YAML configs. `perturbations_config.yaml`
controls the analysis run: which models/aliases to include, which metrics to compute,
bootstrap/CI settings, and where to find the per-trial input scores.
`regenerate_perturbation_blocks.yaml` controls the regen step: which metrics to write,
the leaderboard JSON and results paths, and any display-name overrides.

## Rank stability audit

`stats_perturbations.py` also writes `results_rank_stability.csv` (via
`eva.utils.rank_stability`): for each (metric, condition), every model gets its observed
rank, how often it kept that rank under a joint cluster bootstrap over (domain, scenario)
units (the same resampled scenarios applied to every model), a percentile band on its
replicate ranks, and its rank under domain-weighted instead of scenario-weighted pooling
— `rank_changed` flags rows an equally defensible aggregation rule would move. Read the
per-model CIs and this table together: a model can sit atop the point estimates yet hold
that top rank in only half the replicates. Adapted from *How Reproducible Are Evaluation
Conclusions? A Self-Audit of LLM-Inferred Prompt Structure* (arXiv:2609.30074), which
recommends reporting rank stability and executed sensitivity comparisons alongside any
ranking.

