"""Integration tests for per-scenario transition audits against a measured null.

Builds synthetic trial scores, pushes them through the existing
data_perturbations pipeline (scenario means -> deltas), and checks that the
null-measured audit keeps real transitions while suppressing the phantom ones
a naive sign ledger reports on unchanged conditions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "perturbations"))

from data_perturbations import build_scenario_deltas, compute_deltas, compute_scenario_means  # noqa: E402
from transition_audits import audit_transitions, build_null_deltas, main, null_floor  # noqa: E402

METRIC = "score"
ALIAS = "model-a"
NOISE_SD = 0.3
SHIFT = 2.0  # true effect on shifted scenarios, on a 0-10 metric scale
CONDITION_MAP = {"A": "accent", "B": "background_noise"}


def _draw(rng: np.random.Generator, mu: float) -> float:
    return float(np.clip(mu + rng.normal(0.0, NOISE_SD), 0.0, 10.0))


def make_trial_scores(seed: int = 7) -> pd.DataFrame:
    """Trial scores in the trial_scores.csv contract: 2 domains x 24 scenarios x 3 trials.

    The accent condition shifts even-numbered scenarios by +SHIFT; background_noise
    is drawn from the same distribution as clean (a condition with no real effect).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for domain in ("itsm", "airline"):
        for scenario in range(24):
            scenario_id = f"{domain}_{scenario:02d}"
            level = rng.uniform(3.0, 7.0)  # per-scenario difficulty
            for trial in (1, 2, 3):
                for category, mu in (
                    ("clean", level),
                    ("accent", level + (SHIFT if scenario % 2 == 0 else 0.0)),
                    ("background_noise", level),
                ):
                    rows.append(
                        {
                            "system_alias": ALIAS,
                            "domain": domain,
                            "perturbation_category": category,
                            "scenario_id": scenario_id,
                            "trial": trial,
                            "metric": METRIC,
                            "value": _draw(rng, mu),
                        }
                    )
    return pd.DataFrame(rows)


def _is_shifted(scenario_id: str) -> bool:
    return int(scenario_id.split("_")[1]) % 2 == 0


class TestBuildNullDeltas:
    def test_null_pool_measures_noise_not_effect(self):
        """Split-half nulls stay far below the true effect size and center on zero."""
        trial_scores = make_trial_scores()
        clean = trial_scores[trial_scores["perturbation_category"] == "clean"]
        nulls = build_null_deltas(clean, ALIAS, n_splits=8, seed=42)

        assert len(nulls) == 48 * 8  # 48 scenario cells x 8 splits
        assert nulls["null_delta"].abs().max() < SHIFT
        assert nulls["null_delta"].abs().mean() < 0.5


class TestAuditTransitions:
    def test_audit_keeps_real_gains_and_suppresses_phantom_ones(self):
        """Naive ledger calls transitions on a no-effect condition; the audit does not."""
        trial_scores = make_trial_scores()
        clean = trial_scores[trial_scores["perturbation_category"] == "clean"]
        nulls = build_null_deltas(clean, ALIAS, n_splits=8, seed=42)

        deltas = compute_deltas(compute_scenario_means(trial_scores), alias=ALIAS, condition_map=CONDITION_MAP)
        deltas.insert(0, "model_label", ALIAS)

        ledger = audit_transitions(deltas, nulls, alpha=0.05)
        accent = ledger[ledger["perturbation_condition"] == "accent"]
        noise = ledger[ledger["perturbation_condition"] == "background_noise"]

        # Real +2.0 shifts are called gains; unshifted accent scenarios mostly are not.
        shifted = accent[accent["scenario_id"].map(_is_shifted)]
        unshifted = accent[~accent["scenario_id"].map(_is_shifted)]
        assert (shifted["audited_transition"] == "gain").sum() >= 20
        assert (unshifted["audited_transition"] == "no_change").sum() >= 20
        assert (accent.loc[accent["audited_transition"] == "gain", "delta"] > 0).all()

        # Pure-noise condition: the naive ledger is full of phantom transitions,
        # the audited ledger calls none.
        assert noise["naive_transition"].isin(["gain", "lose"]).sum() >= 40
        assert (noise["audited_transition"] == "no_change").all()

    def test_thin_null_pool_is_flagged_not_guessed(self):
        """Families with too few nulls to resolve a p-value are flagged, not called."""
        trial_scores = make_trial_scores()
        clean = trial_scores[trial_scores["perturbation_category"] == "clean"]
        two_scenarios = clean["scenario_id"].unique()[:2]
        thin_nulls = build_null_deltas(clean[clean["scenario_id"].isin(two_scenarios)], ALIAS, n_splits=8, seed=42)
        assert len(thin_nulls) < 20

        deltas = compute_deltas(compute_scenario_means(trial_scores), alias=ALIAS, condition_map=CONDITION_MAP)
        deltas.insert(0, "model_label", ALIAS)

        ledger = audit_transitions(deltas, thin_nulls, alpha=0.05)
        assert len(ledger) == 96  # 48 scenarios x 2 conditions
        assert (ledger["audited_transition"] == "insufficient_null").all()
        assert ledger["p_value"].isna().all()


class TestNullFloor:
    def test_null_floor_measures_the_phantom_rate(self):
        """Auditing the null itself: naive sign rate ~1, audited reject rate 0."""
        trial_scores = make_trial_scores()
        clean = trial_scores[trial_scores["perturbation_category"] == "clean"]
        nulls = build_null_deltas(clean, ALIAS, n_splits=8, seed=42)

        floor = null_floor(nulls, alpha=0.05)

        assert len(floor) == 1
        row = floor.iloc[0]
        assert row["n_null"] == 48 * 8
        assert row["n_scenarios"] == 48
        assert row["naive_transition_rate"] >= 0.95
        assert row["audited_reject_rate"] == 0.0


class TestTransitionAuditsMain:
    def test_main_writes_ledger_and_floor(self, tmp_path: Path):
        """End to end through the documented entry point, on the existing CSV contracts."""
        trial_scores = make_trial_scores()
        (tmp_path / "output" / "testx").mkdir(parents=True)
        trial_scores.to_csv(tmp_path / "output" / "testx" / "trial_scores.csv", index=False)

        # Deltas come from the existing data step (build_scenario_deltas -> scenario_deltas.csv).
        output_dir = tmp_path / "output_processed" / "testx" / "perturbations"
        output_dir.mkdir(parents=True)
        scenario_deltas = build_scenario_deltas(
            trial_scores=trial_scores,
            model_label=ALIAS,
            alias=ALIAS,
            condition_map=CONDITION_MAP,
            metrics=[METRIC],
        )
        scenario_deltas.to_csv(output_dir / "scenario_deltas.csv", index=False)

        config_dir = tmp_path / "local" / "perturbations"
        config_dir.mkdir(parents=True)
        config = {
            "random_seed": 42,
            "alpha": 0.05,
            "n_null_splits": 8,
            "metrics": [METRIC],
            "trial_scores_path": "output/testx/trial_scores.csv",
            "output_dir": "output_processed/testx/perturbations",
            "models": {ALIAS: {"alias": ALIAS, "conditions": CONDITION_MAP}},
        }
        config_path = config_dir / "perturbations_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        main(config_path)

        ledger = pd.read_csv(output_dir / "transition_ledger.csv")
        floor = pd.read_csv(output_dir / "null_floor.csv")

        assert len(ledger) == 96
        noise = ledger[ledger["perturbation_condition"] == "background_noise"]
        accent = ledger[ledger["perturbation_condition"] == "accent"]
        assert (noise["audited_transition"] == "no_change").all()
        assert (accent["audited_transition"] == "gain").sum() >= 20
        assert len(floor) == 1
        assert floor.iloc[0]["audited_reject_rate"] == 0.0
