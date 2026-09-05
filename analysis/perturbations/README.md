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
| `transition_audits.py` | Audits the per-scenario gain/lose ledger against a measured null built from clean replicate trials (see "Transition audits" below). |
| `run_perturbations.py` | End-to-end driver: calls `data_perturbations`, then `stats_perturbations`, then `transition_audits` in sequence. |
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

## Transition audits (measured null)

`transition_audits.py` audits the per-scenario gain/lose ledger against a
separately measured null, adapted from *Phantom Gains: Auditing
Self-Improvement Against a Measured Null* (arXiv:2608.20290). A sign-based
ledger reports a transition whenever a scenario's delta is non-zero, which
pure trial noise already guarantees; the audit instead builds the noise floor
for free from the clean replicate trials the run already contains (split-half
deltas per scenario), tests each observed delta against that pooled empirical
null, and Benjamini-Hochberg-corrects across scenarios within each
(model, metric, condition) family. `null_floor.csv` additionally reports how
often the naive and audited ledgers call a transition on pure-noise
comparisons — the floor any real condition's transition rate should be read
against. Pools smaller than the resolution needed to compute a p-value are
flagged `insufficient_null` rather than guessed at.

It reads `trial_scores.csv` and the `scenario_deltas.csv` written by
`data_perturbations.py`, and writes `transition_ledger.csv` and
`null_floor.csv` to the configured `output_dir`. It runs as the final step of
`run_perturbations.py` (data → stats → audit), and can also be run on its own:

```bash
uv run python analysis/perturbations/transition_audits.py
```

Optional config key (in `perturbations_config.yaml`): `n_null_splits`
(default 8), the number of split-half comparisons drawn per scenario when
building the null.
