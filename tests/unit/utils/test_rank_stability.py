"""Unit tests for src/eva/utils/rank_stability.py and its perturbation-pipeline wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.perturbations.stats_perturbations import main as stats_main
from eva.utils.rank_stability import (
    RANK_STABILITY_COLUMNS,
    joint_bootstrap_ranks,
    rank_stability_table,
    rule_sensitivity,
)


def _separated_scores(n_units: int = 30) -> np.ndarray:
    """Three models whose scenario-level distributions do not overlap."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(n_units, 3)) + np.array([2.0, 1.0, 0.0])


def _overlapping_scores(n_units: int = 30) -> np.ndarray:
    """Top two models are statistically indistinguishable, the third is clearly worst."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(n_units, 3)) + np.array([1.0, 1.0, 0.0])


class TestJointBootstrapRanks:
    def test_separated_models_hold_rank(self):
        audit = joint_bootstrap_ranks(_separated_scores(), seed=42, n_boot=400)
        np.testing.assert_allclose(audit.observed_rank, [1.0, 2.0, 3.0])
        assert np.all(audit.rank_hold_freq >= 0.95)
        np.testing.assert_allclose(audit.rank_lo, audit.observed_rank)
        np.testing.assert_allclose(audit.rank_hi, audit.observed_rank)

    def test_overlapping_models_do_not_hold_rank(self):
        audit = joint_bootstrap_ranks(_overlapping_scores(), seed=42, n_boot=400)
        np.testing.assert_allclose(audit.observed_rank[-1], 3.0)
        # The indistinguishable pair swaps constantly: each keeps its observed rank in only
        # ~half the replicates, and each one's band spans both contested ranks. The clear
        # loser keeps last place — the paper's finding that only part of a table is firm.
        assert np.all(audit.rank_hold_freq[:2] <= 0.75)
        np.testing.assert_array_equal(audit.rank_lo[:2], [1.0, 1.0])
        np.testing.assert_array_equal(audit.rank_hi[:2], [2.0, 2.0])
        assert audit.rank_hold_freq[2] >= 0.95

    def test_seed_reproducible(self):
        scores = _overlapping_scores()
        a = joint_bootstrap_ranks(scores, seed=7, n_boot=200)
        b = joint_bootstrap_ranks(scores, seed=7, n_boot=200)
        np.testing.assert_array_equal(a.rank_hold_freq, b.rank_hold_freq)
        np.testing.assert_array_equal(a.rank_lo, b.rank_lo)
        c = joint_bootstrap_ranks(scores, seed=8, n_boot=200)
        # Either seed: the contested pair holds rank about half the time.
        for audit in (a, c):
            assert abs(audit.rank_hold_freq[0] - 0.5) < 0.15

    def test_tied_models_share_fractional_rank(self):
        scores = np.tile(np.array([[0.7, 0.7, 0.2]]), (12, 1))
        audit = joint_bootstrap_ranks(scores, seed=3, n_boot=50)
        np.testing.assert_allclose(audit.observed_rank, [1.5, 1.5, 3.0])
        assert np.all(audit.rank_hold_freq == 1.0)  # the tie is structural, not sampling noise

    def test_lower_is_better_reverses_ranking(self):
        scores = _separated_scores(n_units=20)
        asc = joint_bootstrap_ranks(scores, seed=42, n_boot=200, higher_is_better=False)
        np.testing.assert_allclose(asc.observed_rank, [3.0, 2.0, 1.0])

    def test_missing_entries_ranked_on_available_units(self):
        scores = _separated_scores(n_units=25)
        scores[:15, 2] = np.nan  # third model unscored on over half the units
        audit = joint_bootstrap_ranks(scores, seed=42, n_boot=200)
        np.testing.assert_allclose(audit.observed_rank, [1.0, 2.0, 3.0])
        assert 0.0 <= audit.rank_hold_freq[2] <= 1.0

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="2D"):
            joint_bootstrap_ranks(np.zeros(3), seed=1)
        with pytest.raises(ValueError, match="resampling units"):
            joint_bootstrap_ranks(np.zeros((1, 3)), seed=1)
        with pytest.raises(ValueError, match="no finite scores"):
            joint_bootstrap_ranks(np.array([[1.0, np.nan], [0.0, np.nan]]), seed=1)


class TestRuleSensitivity:
    def test_small_cluster_flips_the_ranking(self):
        # Big cluster (4 units) favours X, small cluster (1 unit) favours Y.
        scores = np.array([[0.9, 0.1]] * 4 + [[0.0, 1.0]])
        clusters = np.array(["itsm"] * 4 + ["airline"])
        by_row, by_cluster = rule_sensitivity(scores, clusters)
        np.testing.assert_allclose(by_row, [1.0, 2.0])  # scenario weighting: X wins
        np.testing.assert_allclose(by_cluster, [2.0, 1.0])  # domain weighting: Y wins

    def test_equal_clusters_agree(self):
        scores = _separated_scores(n_units=24)
        clusters = np.array(["a", "b"] * 12)
        by_row, by_cluster = rule_sensitivity(scores, clusters)
        np.testing.assert_array_equal(by_row, by_cluster)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            rule_sensitivity(np.zeros((4, 2)), np.array(["a", "a", "b"]))


def _values_long(model_domain_values: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    """Build the scenario-level metric-value table (analysis/perturbations layout)."""
    return pd.DataFrame(
        [
            {
                "model_label": model,
                "domain": domain,
                "condition": "A",
                "scenario_id": f"{domain}_s{i}",
                "metric": "faithfulness",
                "value": value,
            }
            for model, domains in model_domain_values.items()
            for domain, values in domains.items()
            for i, value in enumerate(values)
        ]
    )


class TestRankStabilityTable:
    def test_reports_rank_audit_per_model(self):
        rng = np.random.default_rng(2)
        values = _values_long(
            {
                "good": {"itsm": list(rng.normal(0.8, 0.05, 8)), "airline": list(rng.normal(0.8, 0.05, 4))},
                "mid": {"itsm": list(rng.normal(0.6, 0.05, 8)), "airline": list(rng.normal(0.6, 0.05, 4))},
                "bad": {"itsm": list(rng.normal(0.3, 0.05, 8)), "airline": list(rng.normal(0.3, 0.05, 4))},
            }
        )
        out = rank_stability_table(values, n_boot=400, seed=42)

        assert list(out.columns) == RANK_STABILITY_COLUMNS
        assert len(out) == 3
        np.testing.assert_array_equal(out["rank"].to_numpy(), [1.0, 2.0, 3.0])
        assert out["rank_hold_freq"].between(0.0, 1.0).all()
        assert (out["rank_lo"] <= out["rank"]).all() and (out["rank"] <= out["rank_hi"]).all()
        assert not out["rank_changed"].any()  # balanced domains: both pooling rules agree
        assert (out["n_models"] == 3).all() and (out["n_scenarios"] == 12).all()

        # Same input, same seed -> byte-identical audit (run_seed-derived, like the CIs).
        pd.testing.assert_frame_equal(out, rank_stability_table(values, n_boot=400, seed=42))

    def test_pooling_rule_sensitivity_flagged(self):
        values = _values_long({"x": {"itsm": [0.9] * 4, "airline": [0.0]}, "y": {"itsm": [0.1] * 4, "airline": [1.0]}})
        out = rank_stability_table(values, n_boot=200, seed=42).set_index("model_label")
        assert out.loc["x", "rank"] == 1.0 and out.loc["y", "rank"] == 2.0  # scenario-weighted
        assert out.loc["x", "rank_domain_weighted"] == 2.0  # domain-weighted flips it
        assert out["rank_changed"].all()

    def test_unrankable_cells_and_empty_input(self):
        solo = _values_long({"solo": {"itsm": [0.5, 0.6]}})  # 1 model: nothing to rank
        assert rank_stability_table(solo, n_boot=50, seed=42).empty
        columns = ["model_label", "domain", "condition", "scenario_id", "metric", "value"]
        empty = rank_stability_table(pd.DataFrame(columns=columns), n_boot=50, seed=42)
        assert list(empty.columns) == RANK_STABILITY_COLUMNS and empty.empty


class TestPerturbationPipelineWiring:
    def test_stats_main_writes_the_rank_audit(self, tmp_path):
        """The stats_perturbations main() hook must emit the audit next to the metric-value CIs."""
        rng = np.random.default_rng(3)
        metric_values = _values_long(
            {
                "good": {"itsm": list(rng.normal(0.8, 0.05, 6)), "airline": list(rng.normal(0.8, 0.05, 4))},
                "bad": {"itsm": list(rng.normal(0.3, 0.05, 6)), "airline": list(rng.normal(0.3, 0.05, 4))},
            }
        )
        deltas = metric_values.rename(columns={"condition": "perturbation_condition", "value": "delta"})
        deltas["baseline_mean"] = deltas["delta"] + 0.1
        deltas["perturb_mean"] = deltas["delta"] + deltas["baseline_mean"]

        out_dir = tmp_path / "output_processed" / "run" / "perturbations"
        out_dir.mkdir(parents=True)
        deltas.to_csv(out_dir / "scenario_deltas.csv", index=False)
        metric_values.to_csv(out_dir / "scenario_metricvalues.csv", index=False)

        config_path = tmp_path / "local" / "perturbations" / "perturbations_config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            "output_dir: output_processed/run/perturbations\n"
            "random_seed: 42\n"
            "alpha: 0.05\n"
            "n_permutations: 20\n"
            "n_bootstrap: 200\n"
        )

        stats_main(config_path)

        ranks_path = out_dir / "results_rank_stability.csv"
        assert ranks_path.exists(), "main() must write results_rank_stability.csv alongside the metric-value CIs"
        ranks = pd.read_csv(ranks_path)
        assert list(ranks.columns) == RANK_STABILITY_COLUMNS
        by_model = ranks.set_index("model_label")
        assert by_model.loc["good", "rank"] == 1.0 and by_model.loc["bad", "rank"] == 2.0
        assert by_model.loc["good", "rank_hold_freq"] >= 0.95  # well-separated: the ranking is firm
        assert not by_model["rank_changed"].any()
