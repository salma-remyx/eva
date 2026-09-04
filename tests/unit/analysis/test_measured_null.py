"""Measured-null audit tests for the perturbation significance pipeline.

The audit exists to check the pipeline's own statistic — stats_perturbations'
sign-flip permutation test — against a null measured on frozen-control (clean)
trial replicates. These tests therefore drive the existing pipeline functions
(build_scenario_deltas, run_analysis) on synthetic trial scores and verify the
audit's verdicts, mirroring the failure modes from "Phantom Gains: Auditing
Self-Improvement Against a Measured Null" (arXiv:2608.20290).
"""

import numpy as np
import pandas as pd

import measured_null
import stats_perturbations
from data_perturbations import build_scenario_deltas

ALIAS = "test-model"
MODEL_LABEL = "Test Model"
METRIC = "agent_speech_fidelity"
CONDITIONS = {"A": "accent"}


def _config() -> dict:
    """Minimal perturbations_config subset consumed by run_analysis / null_pvalues."""
    return {"alpha": 0.05, "n_permutations": 2000, "n_bootstrap": 200, "random_seed": 11}


def _trial_scores(
    scenario_base: np.ndarray,
    trial_effects: dict[str, np.ndarray],
    perturb_shift: float,
    noise: float = 0.01,
    seed: int = 7,
    metric: str = METRIC,
) -> pd.DataFrame:
    """Build trial_scores rows.

    Each condition's trials share a per-trial effect (a batch artifact: e.g. trial 0
    always scores high), plus small per-cell noise. `perturb_shift` adds a constant
    to the accent condition only — a uniform shift indistinguishable from a batch
    effect, which is exactly the kind of "gain" the audit must catch.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for j, base in enumerate(scenario_base):
        for cond_label, category in (("clean", "clean"), ("A", "accent")):
            shift = perturb_shift if category == "accent" else 0.0
            for trial, effect in enumerate(trial_effects[cond_label]):
                value = float(np.clip(base + effect + shift + rng.normal(0.0, noise), 0.0, 1.0))
                rows.append(
                    {
                        "system_alias": ALIAS,
                        "domain": "airline",
                        "perturbation_category": category,
                        "scenario_id": f"scenario_{j:02d}",
                        "metric": metric,
                        "trial": trial,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def _pooled_null(trial_scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Measured-null p-values for the pooled family, labeled for run_analysis output."""
    null_deltas = measured_null.replicate_deltas(trial_scores, alias=ALIAS, metrics=[METRIC])
    null_deltas["domain"] = "pooled"
    null_pv = measured_null.null_pvalues(null_deltas, config)
    null_pv["model_label"] = MODEL_LABEL
    return null_pv


def test_phantom_rejection_exposed_by_measured_null() -> None:
    """A batch-shift "gain" passes run_analysis but fails its own noise floor."""
    scenario_base = np.linspace(0.2, 0.6, 40)
    effects = np.array([0.06, 0.0, -0.06])
    trial_scores = _trial_scores(
        scenario_base,
        {"clean": effects, "A": effects},  # same batch structure on both sides
        perturb_shift=0.05,
    )

    # Existing pipeline: consistent +0.05 deltas -> the assumed-null test rejects.
    deltas = build_scenario_deltas(trial_scores, MODEL_LABEL, ALIAS, CONDITIONS, [METRIC])
    pooled = deltas.copy()
    pooled["domain"] = "pooled"
    results = stats_perturbations.run_analysis(pooled, _config())
    rejected = results[results["reject"]]
    assert not rejected.empty
    observed_gain = float(rejected["observed_mean_delta"].iloc[0])
    assert observed_gain > 0.0

    # Measured null: clean-vs-clean replicates manufacture deltas at least this large.
    null_pv = _pooled_null(trial_scores, _config())
    assert not null_pv.empty
    assert float((null_pv["p_value"] < 0.05).mean()) > 0.5  # phantom rejection rate
    assert (null_pv["phantom_mean_delta"].abs() > observed_gain).any()

    audited = measured_null.audit_results(results, null_pv, alpha=0.05)
    assert audited["verified"].dtype == bool
    assert not audited["verified"].any()  # the rejection does not survive the null


def test_valid_null_audit_is_noop() -> None:
    """With i.i.d. symmetric replicate noise the measured null stays near nominal."""
    scenario_base = np.linspace(0.2, 0.6, 40)
    trial_scores = _trial_scores(
        scenario_base,
        {"clean": np.zeros(6), "A": np.zeros(6)},  # no batch structure, 6 replicates
        perturb_shift=0.0,
    )

    null_pv = _pooled_null(trial_scores, _config())
    assert len(null_pv) == 15  # C(6, 2) replicate pairs
    assert float((null_pv["p_value"] < 0.05).mean()) <= 0.2
    summary = measured_null.audit_results(
        pd.DataFrame(
            {"model_label": [MODEL_LABEL], "metric": [METRIC], "domain": ["pooled"], "reject": [False], "raw_p": [0.5]}
        ),
        null_pv,
        alpha=0.05,
    )
    assert float(summary["calibrated_alpha"].iloc[0]) > 0.0

    # Families with no replicate pairs fall back to the nominal threshold.
    fallback = measured_null.audit_results(
        pd.DataFrame(
            {"model_label": ["other"], "metric": [METRIC], "domain": ["pooled"], "reject": [True], "raw_p": [0.01]}
        ),
        null_pv,
        alpha=0.05,
    )
    assert bool(fallback["verified"].iloc[0])


def test_scenario_exact_tests_verify_only_planted_transitions() -> None:
    """Per-scenario exact tests + BH-FDR keep the planted gain/loss, drop the noise."""
    rows = []
    spec = {"gain_00": ((1, 6), (6, 6)), "loss_00": ((6, 6), (1, 6))}
    for i in range(30):
        spec[f"null_{i:02d}"] = ((3, 6), (3, 6)) if i % 2 == 0 else ((2, 6), (4, 6))
    for scenario_id, ((k_clean, n_clean), (k_pert, n_pert)) in spec.items():
        for category, (k, n) in (("clean", (k_clean, n_clean)), ("accent", (k_pert, n_pert))):
            for trial in range(n):
                rows.append(
                    {
                        "system_alias": ALIAS,
                        "domain": "airline",
                        "perturbation_category": category,
                        "scenario_id": scenario_id,
                        "metric": "task_completion",
                        "trial": trial,
                        "value": 1.0 if trial < k else 0.0,
                    }
                )
    trial_scores = pd.DataFrame(rows)

    transitions = measured_null.scenario_exact_tests(
        trial_scores, ALIAS, MODEL_LABEL, CONDITIONS, "task_completion", alpha=0.05
    )
    assert len(transitions) == len(spec)
    significant = transitions[transitions["significant"]]
    assert set(significant["scenario_id"]) == {"gain_00", "loss_00"}
    assert significant.set_index("scenario_id")["direction"].to_dict() == {"gain_00": "gain", "loss_00": "loss"}
    # Corrected p-values are never below their raw ones under BH.
    assert (transitions["p_fdr"] >= transitions["p_value"] - 1e-12).all()


def test_replicate_deltas_ignores_single_trial_scenarios() -> None:
    """Scenarios without repeat trials cannot contribute to the measured null."""
    trial_scores = pd.DataFrame(
        [
            {
                "system_alias": ALIAS,
                "domain": "airline",
                "perturbation_category": "clean",
                "scenario_id": "solo",
                "metric": METRIC,
                "trial": 0,
                "value": 0.5,
            }
        ]
    )
    empty = measured_null.replicate_deltas(trial_scores, alias=ALIAS, metrics=[METRIC])
    assert empty.empty
    assert list(empty.columns) == measured_null.NULL_DELTA_COLUMNS
    assert measured_null.null_pvalues(empty, _config()).empty
