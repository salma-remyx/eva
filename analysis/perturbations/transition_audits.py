# Config: local/perturbations/perturbations_config.yaml (shared with the other steps)
#
# Additional keys used here:
#   n_null_splits: 8   # split-half comparisons drawn per scenario for the null

"""Per-scenario gain/lose transition audits against a measured null.

Adapted from "Phantom Gains: Auditing Self-Improvement Against a Measured Null"
(arXiv:2608.20290): a ledger of per-problem gains/losses differences two noisy
estimates, so it manufactures phantom transitions. Every transition statistic
must instead be validated against a separately measured null — built from the
baseline replicates the study already owns — with per-problem exact tests under
false-discovery-rate control.

Mapped onto the perturbation pipeline (the clean replicate trials are the
baseline replicates a multi-condition run already contains):
  1. build_null_deltas  — split-half deltas between disjoint halves of each
                          scenario's *clean* trials: the measured noise floor
                          of any per-scenario delta, at zero extra experiments
  2. audit_transitions  — per-scenario exact p-value against the pooled null
                          (p = (1 + #{|null| >= |obs|}) / (1 + n)), then
                          Benjamini-Hochberg FDR across scenarios within each
                          (model, metric, condition) family
  3. null_floor         — runs the same audit on the null itself and reports
                          how often a naive sign ledger and the audited
                          procedure call a transition on pure-noise comparisons

Pure computation: takes DataFrames, returns DataFrames. No file I/O, no plotting.

Run from project root after data_perturbations.py:
    uv run python analysis/perturbations/transition_audits.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from data_perturbations import load_trial_scores
from eva.utils.bootstrap import run_seed

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "perturbations" / "perturbations_config.yaml"

NULL_COLUMNS: list[str] = ["model_label", "domain", "scenario_id", "metric", "split", "null_delta"]

LEDGER_COLUMNS: list[str] = [
    "model_label",
    "metric",
    "domain",
    "perturbation_condition",
    "scenario_id",
    "delta",
    "naive_transition",
    "p_value",
    "q_value",
    "audited_transition",
]

FLOOR_COLUMNS: list[str] = [
    "model_label",
    "metric",
    "n_null",
    "n_scenarios",
    "naive_transition_rate",
    "audited_reject_rate",
]


def build_null_deltas(
    clean_trials: pd.DataFrame,
    model_label: str,
    *,
    n_splits: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Split-half null deltas from clean replicate trials.

    For each (domain, scenario, metric) cell with >= 2 clean trials, draws
    ``n_splits`` random splits of the trials into two disjoint halves and
    records the difference of half means. Each split is a per-scenario delta
    measured with no perturbation at all, so its distribution is the noise
    floor any real perturbation delta must be judged against. Cells with a
    single trial cannot support a split and are skipped.

    Args:
        clean_trials: Rows of trial_scores.csv filtered to one model's clean
            condition (columns: domain, scenario_id, metric, trial, value).
        model_label: Display label attached to every null row.
        n_splits: Random splits to draw per cell (more splits -> finer p-value
            resolution, at the cost of correlated null samples).
        seed: Base seed; each cell derives its own via eva.utils.bootstrap.run_seed
            so nulls are deterministic across runs.

    Returns:
        DataFrame with columns NULL_COLUMNS (one row per cell x split).
    """
    rows: list[dict] = []
    group_keys = ["domain", "scenario_id", "metric"]

    for keys, group in clean_trials.groupby(group_keys, sort=False):
        meta = dict(zip(group_keys, _as_tuple(keys)))
        trials = group.sort_values("trial")["value"].to_numpy(dtype=float)
        if len(trials) < 2:
            continue

        rng = np.random.default_rng(run_seed(f"{seed}:{model_label}:{keys}"))
        half = len(trials) // 2
        for split in range(n_splits):
            shuffled = rng.permutation(trials)
            rows.append(
                {
                    "model_label": model_label,
                    **meta,
                    "split": split,
                    "null_delta": float(shuffled[half:].mean() - shuffled[:half].mean()),
                }
            )

    return pd.DataFrame(rows, columns=NULL_COLUMNS)


def _exact_p_vs_pool(observed: np.ndarray, null_pool: np.ndarray) -> np.ndarray:
    """Two-sided exact p-values: p = (1 + #{|null| >= |obs|}) / (1 + n_null)."""
    exceed = np.abs(null_pool)[None, :] >= np.abs(observed)[:, None]
    return (1 + exceed.sum(axis=1)) / (1 + len(null_pool))


def audit_transitions(
    deltas_df: pd.DataFrame,
    nulls_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    min_null_size: int = 20,
) -> pd.DataFrame:
    """Per-scenario gain/lose ledger validated against the measured null.

    The naive ledger signs each observed delta (perturb - clean); the audited
    ledger keeps that call only when the delta clears the noise floor: its
    exact p-value against the pooled null, Benjamini-Hochberg-corrected across
    all scenarios of the same (model, metric, condition) family. The null pool
    spans scenarios and domains within (model, metric) — pooling is what gives
    the exact test resolution, and within-scenario differencing already cancels
    per-scenario difficulty levels.

    Args:
        deltas_df: scenario_deltas.csv contract (model_label, domain,
            perturbation_condition, scenario_id, metric, delta, ...).
        nulls_df: Output of build_null_deltas, may span several models.
        alpha: FDR level for the Benjamini-Hochberg correction.
        min_null_size: Pools smaller than this cannot resolve a p-value; those
            families are flagged "insufficient_null" instead of guessing.

    Returns:
        DataFrame with columns LEDGER_COLUMNS (one row per scenario x condition).
    """
    rows: list[dict] = []

    family_keys = ["model_label", "metric", "perturbation_condition"]
    for keys, family in deltas_df.groupby(family_keys, sort=False):
        model_label, metric, condition = _as_tuple(keys)
        pool = nulls_df[(nulls_df["model_label"] == model_label) & (nulls_df["metric"] == metric)]
        pool_deltas = pool["null_delta"].to_numpy(dtype=float)

        deltas = family["delta"].to_numpy(dtype=float)
        naive = np.where(deltas > 0, "gain", np.where(deltas < 0, "lose", "no_change"))

        if len(pool_deltas) < min_null_size:
            p_values = np.full(len(deltas), np.nan)
            q_values = np.full(len(deltas), np.nan)
            audited = np.full(len(deltas), "insufficient_null", dtype=object)
        else:
            p_values = _exact_p_vs_pool(deltas, pool_deltas)
            reject, q_values, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
            audited = np.where(reject & (deltas > 0), "gain", np.where(reject & (deltas < 0), "lose", "no_change"))

        cell = zip(family["domain"], family["scenario_id"], deltas, naive, p_values, q_values, audited)
        for domain, scenario_id, delta, naive_call, p_val, q_val, audit_call in cell:
            rows.append(
                {
                    "model_label": model_label,
                    "metric": metric,
                    "domain": domain,
                    "perturbation_condition": condition,
                    "scenario_id": scenario_id,
                    "delta": float(delta),
                    "naive_transition": str(naive_call),
                    "p_value": float(p_val),
                    "q_value": float(q_val),
                    "audited_transition": str(audit_call),
                }
            )

    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def null_floor(
    nulls_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    min_null_size: int = 20,
) -> pd.DataFrame:
    """Measured phantom-transition floor: audit the null against itself.

    Applies the exact-test + FDR procedure to the null deltas alone (each
    compared against the pool with itself left out, so the p-values are not
    biased by self-inclusion). ``naive_transition_rate`` is how often a sign
    ledger reports a transition on pure-noise comparisons; ``audited_reject_rate``
    is the same for the audited procedure. This is the floor against which any
    real condition's transition rate should be read.
    """
    rows: list[dict] = []

    for keys, group in nulls_df.groupby(["model_label", "metric"], sort=False):
        model_label, metric = _as_tuple(keys)
        d = group["null_delta"].to_numpy(dtype=float)

        if len(d) < min_null_size:
            naive_rate, reject_rate = np.nan, np.nan
        else:
            naive_rate = float(np.mean(d != 0))
            exceed = np.abs(d)[None, :] >= np.abs(d)[:, None]  # (n, n); self always counted
            p_values = exceed.sum(axis=1) / len(d)  # leave-one-out: (1 + others) / n
            reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
            reject_rate = float(np.mean(reject))

        rows.append(
            {
                "model_label": model_label,
                "metric": metric,
                "n_null": len(d),
                "n_scenarios": int(group["scenario_id"].nunique()),
                "naive_transition_rate": naive_rate,
                "audited_reject_rate": reject_rate,
            }
        )

    return pd.DataFrame(rows, columns=FLOOR_COLUMNS)


def _as_tuple(group_vals: object) -> tuple:
    """Normalize a pandas groupby key to a tuple (single-column groups yield a scalar)."""
    if isinstance(group_vals, tuple):
        return group_vals
    return (group_vals,)


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    output_dir = project_root / config["output_dir"]

    deltas_path = output_dir / "scenario_deltas.csv"
    if not deltas_path.exists():
        raise FileNotFoundError(f"scenario_deltas.csv not found at {deltas_path}. Run data_perturbations.py first.")

    if "trial_scores_path" in config:
        trial_scores_path = project_root / config["trial_scores_path"]
    else:
        data_dir = project_root / config["trial_scores_dir"]
        subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in {data_dir}")
        trial_scores_path = subdirs[-1] / "trial_scores.csv"

    print(f"Loading deltas from {deltas_path} ...")
    deltas_df = pd.read_csv(deltas_path)
    print(f"  {len(deltas_df):,} rows loaded")

    print(f"Loading trial scores from {trial_scores_path} ...")
    trial_scores = load_trial_scores(trial_scores_path)

    metrics: list[str] = config["metrics"]
    n_splits: int = config.get("n_null_splits", 8)
    seed: int = config["random_seed"]
    alpha: float = config["alpha"]

    model_nulls: list[pd.DataFrame] = []
    for model_label, model_cfg in config["models"].items():
        clean = trial_scores[
            (trial_scores["system_alias"] == model_cfg["alias"])
            & (trial_scores["metric"].isin(metrics))
            & (trial_scores["perturbation_category"] == "clean")
        ]
        model_nulls.append(build_null_deltas(clean, model_label, n_splits=n_splits, seed=seed))

    nulls_df = pd.concat(model_nulls, ignore_index=True) if model_nulls else pd.DataFrame(columns=NULL_COLUMNS)
    # Keep the floor table aligned with the delta analysis (only complete models made it in).
    nulls_df = nulls_df[nulls_df["model_label"].isin(deltas_df["model_label"].unique())]
    print(f"Built {len(nulls_df):,} null deltas from clean replicate trials")

    ledger = audit_transitions(deltas_df, nulls_df, alpha=alpha)
    floor = null_floor(nulls_df, alpha=alpha)

    ledger_path = output_dir / "transition_ledger.csv"
    floor_path = output_dir / "null_floor.csv"
    ledger.to_csv(ledger_path, index=False)
    floor.to_csv(floor_path, index=False)
    print(f"Wrote {len(ledger):,} ledger rows -> {ledger_path}")
    print(f"Wrote {len(floor):,} floor rows -> {floor_path}")

    for _, row in floor.iterrows():
        print(
            f"  [{row['model_label']}/{row['metric']}] naive floor "
            f"{row['naive_transition_rate']:.3f}, audited floor {row['audited_reject_rate']:.3f} "
            f"({row['n_null']} nulls over {row['n_scenarios']} scenarios)"
        )


if __name__ == "__main__":
    main()
