"""Unit tests for src/eva/utils/rank_stability.py."""

from __future__ import annotations

import numpy as np

from eva.utils.rank_stability import (
    FIRM_RANK_HOLD,
    joint_cluster_bootstrap_ranks,
    rank_stability_table,
    stability_summary,
)


def _stepped_table() -> dict[str, dict[str, float]]:
    """Four systems where only the top two are close enough to swap under resampling."""
    units = [f"s{u}" for u in range(20)]
    return {
        "top_a": {unit: (0.75 if u % 2 == 0 else 0.5) for u, unit in enumerate(units)},
        "top_b": {unit: (0.5 if u % 2 == 0 else 0.7) for u, unit in enumerate(units)},
        "mid": dict.fromkeys(units, 0.45),
        "clear_worst": dict.fromkeys(units, 0.1),
    }


class TestJointClusterBootstrapRanks:
    def test_separated_systems_hold_ranks(self):
        values = {"a": {f"s{u}": 0.9 for u in range(8)}, "b": {f"s{u}": 0.5 for u in range(8)}}
        ranks = joint_cluster_bootstrap_ranks(values, n_boot=100, seed=1)
        np.testing.assert_array_equal(ranks["a"], np.ones(100))
        np.testing.assert_array_equal(ranks["b"], np.full(100, 2.0))

    def test_identical_systems_share_the_top_rank(self):
        # The joint draw gives both systems the same resampled mean, so every
        # replicate is a tie at rank 1.5 rather than an arbitrary winner.
        values = {"a": {f"s{u}": float(u % 3) for u in range(9)}, "b": {f"s{u}": float(u % 3) for u in range(9)}}
        ranks = joint_cluster_bootstrap_ranks(values, n_boot=50, seed=2)
        np.testing.assert_array_equal(ranks["a"], np.full(50, 1.5))
        np.testing.assert_array_equal(ranks["b"], np.full(50, 1.5))

    def test_deterministic_per_seed(self):
        values = _stepped_table()
        a = joint_cluster_bootstrap_ranks(values, n_boot=100, seed=7)
        b = joint_cluster_bootstrap_ranks(values, n_boot=100, seed=7)
        for system in values:
            np.testing.assert_array_equal(a[system], b[system])

    def test_empty_input(self):
        assert joint_cluster_bootstrap_ranks({}, n_boot=10, seed=0) == {}


class TestRankStabilityTable:
    def test_worst_row_firm_top_rows_flip(self):
        # The paper's headline shape: the bottom of the ranking is firm while the
        # top two swap often enough that the table cannot name the best system.
        table = rank_stability_table(_stepped_table(), seed=7, n_boot=400)

        assert list(table) == ["top_a", "top_b", "mid", "clear_worst"]
        assert table["top_a"]["rank"] == 1.0
        assert table["top_a"]["rank_hold_share"] < FIRM_RANK_HOLD
        assert table["top_b"]["rank_hold_share"] < FIRM_RANK_HOLD
        # Exactly one of the top two is rank 1 in each replicate they don't tie.
        assert table["top_a"]["share_rank_1"] + table["top_b"]["share_rank_1"] == 1.0
        assert 0.0 < table["top_b"]["share_rank_1"] < 1.0
        assert table["clear_worst"]["rank"] == 4.0
        assert table["clear_worst"]["rank_hold_share"] == 1.0
        assert table["clear_worst"]["firm"] is True

    def test_summary_identifies_worst_but_not_best(self):
        summary = stability_summary(rank_stability_table(_stepped_table(), seed=7, n_boot=400))

        assert summary["n_systems"] == 4
        assert summary["n_units"] == 20
        assert summary["n_firm_rows"] == 2
        assert summary["best_system"] == "top_a"
        assert summary["best_system_firm"] is False
        assert summary["worst_system"] == "clear_worst"
        assert summary["worst_system_firm"] is True

    def test_missing_units_reported_as_provenance(self):
        # A system that errored on some scenarios keeps those scenarios out of its
        # means and reports them as missing, instead of being silently padded.
        table = rank_stability_table(
            {"a": {"s1": 1.0, "s2": 0.5, "s3": 0.2}, "b": {"s1": 0.4}},
            seed=3,
            n_boot=200,
        )
        assert table["b"]["n_units"] == 1
        assert table["b"]["n_missing_units"] == 2
        assert table["b"]["mean"] == 0.4
        assert table["a"]["n_missing_units"] == 0

    def test_system_without_values_gets_no_rank(self):
        table = rank_stability_table({"a": {"s1": 1.0}, "b": {}}, seed=3, n_boot=50)
        assert table["b"]["rank"] is None
        assert table["b"]["rank_hold_share"] is None
        assert table["b"]["firm"] is False
        assert table["a"]["rank"] == 1.0

    def test_empty_table_summary(self):
        assert stability_summary({})["n_systems"] == 0
