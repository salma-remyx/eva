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
| `src/eva/utils/group_fairness.py` | Per-group fairness summaries (per-group CIs, worst-performing group, best-to-worst gap) used by `stats_perturbations.py` when `group_col` is configured. |

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

## Group fairness

Setting an optional `group_col` (e.g. `accent`) in `perturbations_config.yaml` labels
each trial-score row with a demographic group. `data_perturbations.py` carries the
label into `scenario_metricvalues.csv` — computing trial means within group instead
of pooling them — and `stats_perturbations.py` then writes
`results_groupfairness_per_group.csv` (per-group score + bootstrap CI) and
`results_groupfairness_summary.csv` (worst-performing group, best-to-worst gap with
its own CI, and the pooled aggregate the gap sits behind).

An aggregate score can hide how a condition affects individual speaker groups, so
comparisons between models or conditions should read the worst group's score as a
criterion in its own right, not only the aggregate. Summaries are computed by
`eva.utils.group_fairness` (adapted from *Pruning for Efficiency, Paying in
Fairness: Demographic Disparities in Pruned Speech-LLMs*, arXiv:2609.38106), and
error-rate metrics such as `stt_wer` are oriented automatically (lower is better).
The automatic orientation can be overridden per metric via the optional
`lower_is_better_metrics` and `higher_is_better_metrics` config keys.
