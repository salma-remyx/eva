# Config: local/perturbations/perturbations_config.yaml (same file as stats_perturbations.py)
#
# Reads trial_scores.csv (per-trial values, incl. the repeat clean trials) and the
# results_*.csv files written by stats_perturbations.py, and writes:
#   results_measured_null.csv       — null p-values from clean-replicate pairs
#   results_null_audit_*.csv        — run_analysis output annotated with the null's verdict
#   results_scenario_transitions.csv — per-scenario exact tests vs the pooled baseline

"""Measured-null audit for the perturbation significance pipeline.

Adapted from "Phantom Gains: Auditing Self-Improvement Against a Measured Null"
(arXiv:2608.20290): transition-style statistics (deltas between two noisy runs)
must be audited against a null that was *measured* on frozen-control replicates,
not assumed. The repeat clean trials that trial_scores.csv already holds are
those replicates, so the null costs no new experiments.

Pipeline (pure computation, mirroring stats_perturbations.py):
  1. replicate_deltas     — pure-noise deltas from every pair of clean trials
  2. null_pvalues         — push those deltas through the pipeline's own
                            permutation_test and record where the assumed
                            sign-flip null manufactures rejections (phantom gains)
  3. audit_results        — annotate run_analysis() output with the measured
                            false-rejection rate and a threshold recalibrated to
                            the measured null; rejections that survive are verified
  4. scenario_exact_tests — per-scenario exact binomial tests of each perturbation
                            against the pooled clean baseline under Benjamini-Hochberg
                            FDR control (the paper's replacement for thresholded
                            win/loss ledgers), for binary metrics such as task_completion
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from data_perturbations import load_trial_scores
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from stats_perturbations import permutation_test, run_seed

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "local" / "perturbations" / "perturbations_config.yaml"

NULL_DELTA_COLUMNS = ["domain", "scenario_id", "metric", "trial_a", "trial_b", "delta"]
NULL_PVALUE_COLUMNS = ["model_label", "metric", "domain", "trial_a", "trial_b", "phantom_mean_delta", "p_value"]


def replicate_deltas(
    trial_scores: pd.DataFrame,
    alias: str,
    metrics: list[str],
    condition: str = "clean",
) -> pd.DataFrame:
    """Pure-noise deltas from every pair of frozen-control trials.

    Differencing two trials of the *same* system on the *same* scenario measures
    only run-to-run noise: this is the measured null. One row per
    (domain, scenario_id, metric, trial pair) with delta = value(trial_a) - value(trial_b),
    trial_a < trial_b. Scenarios with a single trial contribute nothing.

    Returns a DataFrame with NULL_DELTA_COLUMNS.
    """
    filtered = trial_scores[
        (trial_scores["system_alias"] == alias)
        & (trial_scores["perturbation_category"] == condition)
        & (trial_scores["metric"].isin(metrics))
    ]
    if filtered.empty:
        return pd.DataFrame(columns=NULL_DELTA_COLUMNS)

    rows: list[dict] = []
    for (domain, scenario_id, metric), g in filtered.groupby(["domain", "scenario_id", "metric"], sort=False):
        g = g.sort_values("trial")
        trials = g["trial"].to_numpy()
        values = g["value"].to_numpy(dtype=float)
        for i in range(len(trials)):
            for j in range(i + 1, len(trials)):
                rows.append(
                    {
                        "domain": domain,
                        "scenario_id": scenario_id,
                        "metric": metric,
                        "trial_a": int(trials[i]),
                        "trial_b": int(trials[j]),
                        "delta": float(values[i] - values[j]),
                    }
                )
    return pd.DataFrame(rows, columns=NULL_DELTA_COLUMNS)


def null_pvalues(null_deltas: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the pipeline's own permutation test on each replicate pair.

    Each (metric, domain, trial pair) becomes one pseudo-run: its scenario-level
    replicate deltas go through stats_perturbations.permutation_test unchanged, so
    the returned p-values are what the assumed sign-flip null yields on data known
    to contain no treatment effect. Rejections here are phantom gains, and
    phantom_mean_delta is the effect size the null manufactures on its own.

    Returns a DataFrame with columns model_label, metric, domain, trial_a, trial_b,
    phantom_mean_delta, p_value (model_label filled in by the caller, which knows
    the display label for the alias the deltas were built from).
    """
    if null_deltas.empty:
        return pd.DataFrame(columns=NULL_PVALUE_COLUMNS)

    n_perm: int = config["n_permutations"]
    seed: int = config["random_seed"]

    rows: list[dict] = []
    for (metric, domain, trial_a, trial_b), g in null_deltas.groupby(
        ["metric", "domain", "trial_a", "trial_b"], sort=False
    ):
        d = g["delta"].to_numpy(dtype=float)
        cell_seed = run_seed(f"{seed}:null:{metric}:{domain}:{trial_a}:{trial_b}")
        rows.append(
            {
                "model_label": None,
                "metric": metric,
                "domain": domain,
                "trial_a": int(trial_a),
                "trial_b": int(trial_b),
                "phantom_mean_delta": float(d.mean()),
                "p_value": permutation_test(d, n_perm=n_perm, seed=cell_seed),
            }
        )
    return pd.DataFrame(rows, columns=NULL_PVALUE_COLUMNS)


def calibrated_alpha(null_ps: np.ndarray, alpha: float) -> float:
    """Significance threshold recalibrated to the measured null.

    The alpha-quantile of the measured-null p-values: when the sign-flip null is
    valid those p-values are ~Uniform(0, 1) and the threshold collapses back to
    the nominal alpha; when replicate noise is structured they concentrate near
    zero and only p-values below that floor count as verified. Strict comparison
    against this threshold, so a family whose null already saturates p = 0
    verifies nothing (the honest verdict there).
    """
    ps = np.asarray(null_ps, dtype=float)
    ps = ps[np.isfinite(ps)]
    if len(ps) == 0:
        return float(alpha)
    return float(np.quantile(ps, alpha))


def audit_results(
    results_df: pd.DataFrame,
    null_pv: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Annotate run_analysis() output with the measured null's verdict.

    Joins each (model_label, metric, domain) family with its measured
    false-rejection rate and recalibrated threshold, then marks which rejections
    survive their own noise floor. Families with no replicate pairs fall back to
    the nominal threshold (verified == reject).

    Returns results_df plus columns: n_null_tests, null_reject_rate,
    calibrated_alpha, verified.
    """
    summary = (
        null_pv.groupby(["model_label", "metric", "domain"], sort=False)["p_value"]
        .agg(
            n_null_tests="count",
            null_reject_rate=lambda s: float((s < alpha).mean()),
            calibrated_alpha=lambda s: calibrated_alpha(s.to_numpy(), alpha),
        )
        .reset_index()
    )
    out = results_df.merge(summary, on=["model_label", "metric", "domain"], how="left")
    if "reject" in out.columns:
        threshold = out["calibrated_alpha"].fillna(alpha)
        out["verified"] = out["reject"] & (out["raw_p"] < threshold)
    return out


def pooled_baseline_rate(clean_values: np.ndarray) -> float:
    """Laplace-corrected pooled pass-rate of the clean baseline trials.

    The +0.5/+1 correction keeps degenerate baselines (all-pass / all-fail) inside
    the open interval (0, 1) so the exact test stays defined for every scenario.
    """
    return (clean_values.sum() + 0.5) / (len(clean_values) + 1.0)


def scenario_exact_tests(
    trial_scores: pd.DataFrame,
    alias: str,
    model_label: str,
    condition_map: dict[str, str],
    metric: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-scenario exact tests against the pooled baseline under FDR control.

    For a binary metric: each scenario's perturbation pass-count is tested against
    that scenario's pooled clean pass-rate (Laplace-corrected so always-pass /
    never-pass baselines stay testable), then Benjamini-Hochberg correction is
    applied within each perturbation condition. Scenarios whose corrected p
    survives are verified gains/losses; everything else sits inside the noise a
    thresholded win/loss ledger would misread as transitions.

    Returns a DataFrame with columns model_label, domain, perturbation_condition,
    scenario_id, n_baseline, baseline_rate, n_perturb, perturb_rate, direction,
    p_value, p_fdr, significant.
    """
    cols = [
        "model_label",
        "domain",
        "perturbation_condition",
        "scenario_id",
        "n_baseline",
        "baseline_rate",
        "n_perturb",
        "perturb_rate",
        "direction",
        "p_value",
        "p_fdr",
        "significant",
    ]
    base_cols = [c for c in cols if c not in ("p_fdr", "significant")]
    empty = pd.DataFrame(columns=cols)

    filtered = trial_scores[(trial_scores["system_alias"] == alias) & (trial_scores["metric"] == metric)]
    clean = filtered[filtered["perturbation_category"] == "clean"]
    if filtered.empty or clean.empty:
        return empty

    rows: list[dict] = []
    for cond_label, pert_cat in condition_map.items():
        pert = filtered[filtered["perturbation_category"] == pert_cat]
        for (domain, scenario_id), c_g in clean.groupby(["domain", "scenario_id"], sort=False):
            p_g = pert[(pert["domain"] == domain) & (pert["scenario_id"] == scenario_id)]
            if p_g.empty:
                continue
            c_vals = c_g["value"].to_numpy(dtype=float)
            p_vals = p_g["value"].to_numpy(dtype=float)
            baseline_rate = float(c_vals.mean())
            perturb_rate = float(p_vals.mean())
            direction = "gain" if perturb_rate > baseline_rate else "loss" if perturb_rate < baseline_rate else "flat"
            rows.append(
                {
                    "model_label": model_label,
                    "domain": domain,
                    "perturbation_condition": cond_label,
                    "scenario_id": scenario_id,
                    "n_baseline": len(c_vals),
                    "baseline_rate": baseline_rate,
                    "n_perturb": len(p_vals),
                    "perturb_rate": perturb_rate,
                    "direction": direction,
                    "p_value": float(
                        binomtest(int(round(p_vals.sum())), len(p_vals), pooled_baseline_rate(c_vals)).pvalue
                    ),
                }
            )

    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=base_cols)
    out["p_fdr"] = np.nan
    out["significant"] = False
    for _, g in out.groupby("perturbation_condition", sort=False):
        reject, corrected, _, _ = multipletests(g["p_value"].to_numpy(), alpha=alpha, method="fdr_bh")
        out.loc[g.index, "p_fdr"] = corrected
        out.loc[g.index, "significant"] = reject
    return out[cols]


def main(config_path: Path = CONFIG_PATH) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent.parent
    output_dir = project_root / config["output_dir"]
    alpha: float = config["alpha"]
    metrics: list[str] = config["metrics"]

    if "trial_scores_dir" in config:
        data_dir = project_root / config["trial_scores_dir"]
        subdirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
        if not subdirs:
            raise FileNotFoundError(f"No timestamped runs under {data_dir}")
        trial_scores_path = subdirs[-1] / "trial_scores.csv"
    else:
        trial_scores_path = project_root / config["trial_scores_path"]

    print(f"Loading trial scores from {trial_scores_path} ...")
    trial_scores = load_trial_scores(trial_scores_path)
    binary_metrics = [
        m
        for m in metrics
        if (trial_scores["metric"] == m).any()
        and trial_scores.loc[trial_scores["metric"] == m, "value"].isin([0.0, 1.0]).all()
    ]

    null_rows: list[pd.DataFrame] = []
    transition_rows: list[pd.DataFrame] = []
    for model_label, model_cfg in config["models"].items():
        deltas = replicate_deltas(trial_scores, model_cfg["alias"], metrics)
        # Per-domain null (audits results_per_domain.csv) and pooled null (results_pooled.csv).
        for pooled in (False, True):
            nd = deltas.copy()
            if pooled:
                nd["domain"] = "pooled"
            pv = null_pvalues(nd, config)
            pv["model_label"] = model_label
            null_rows.append(pv)
        for metric in binary_metrics:
            transition_rows.append(
                scenario_exact_tests(
                    trial_scores, model_cfg["alias"], model_label, model_cfg["conditions"], metric, alpha=alpha
                )
            )

    null_pv = pd.concat(null_rows, ignore_index=True)
    null_path = output_dir / "results_measured_null.csv"
    null_pv.to_csv(null_path, index=False)
    print(f"Wrote {len(null_pv):,} measured-null rows -> {null_path}")

    for results_name in ("results_pooled.csv", "results_per_domain.csv"):
        results_path = output_dir / results_name
        if not results_path.exists():
            print(f"  [audit] skipped: {results_path} not found (run stats_perturbations.py)")
            continue
        results_df = pd.read_csv(results_path)
        audited = audit_results(results_df, null_pv, alpha)
        audit_path = output_dir / results_name.replace("results_", "results_null_audit_")
        audited.to_csv(audit_path, index=False)
        n_reject = int(results_df["reject"].sum())
        n_verified = int(audited["verified"].sum())
        print(f"{results_name}: {n_reject} rejection(s), {n_verified} verified against the measured null")
        print(f"  -> {audit_path}")

    if transition_rows:
        transitions = pd.concat(transition_rows, ignore_index=True)
        transitions_path = output_dir / "results_scenario_transitions.csv"
        transitions.to_csv(transitions_path, index=False)
        n_sig = int(transitions["significant"].sum())
        print(f"Wrote {len(transitions):,} scenario transition rows ({n_sig} verified)")
        print(f"  -> {transitions_path}")
    else:
        print("  [transitions] skipped: no binary metrics among configured metrics")


if __name__ == "__main__":
    main()
