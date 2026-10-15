"""Group-fairness summaries over the perturbation pipeline.

Exercises the wiring end to end: a grouped trial_scores table flows through
data_perturbations.build_condition_scenario_values, eva.utils.group_fairness turns
it into per-group scores plus worst-group/gap summaries, and stats_perturbations.main
writes those tables next to the existing results CSVs.

Reporting protocol adapted from "Pruning for Efficiency, Paying in Fairness:
Demographic Disparities in Pruned Speech-LLMs" (arXiv:2609.38106).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from analysis.perturbations import data_perturbations, stats_perturbations
from eva.utils.group_fairness import group_fairness_summary, metric_lower_is_better

CONDITION_MAP = {"A": "accent"}
METRICS = ["agent_speech_fidelity"]
CONFIG = {"alpha": 0.05, "n_permutations": 200, "n_bootstrap": 500, "random_seed": 42}


def _trial_scores() -> pd.DataFrame:
    """Four scenarios × two accent groups × two trials, plus a clean baseline.

    The indian-accent group scores 0.4 below the french-accent group on every
    scenario, so the worst group and the gap are known in advance.
    """
    rows: list[dict] = []
    for scenario_id in ["s1", "s2", "s3", "s4"]:
        base = 0.8 if scenario_id in {"s1", "s2"} else 0.6
        for trial in [1, 2]:
            rows.append(
                {
                    "system_alias": "agent",
                    "domain": "airline",
                    "perturbation_category": "clean",
                    "scenario_id": scenario_id,
                    "metric": "agent_speech_fidelity",
                    "trial": trial,
                    "value": base,
                    "accent": None,
                }
            )
            for accent, penalty in (("french", 0.0), ("indian", 0.4)):
                rows.append(
                    {
                        "system_alias": "agent",
                        "domain": "airline",
                        "perturbation_category": "accent",
                        "scenario_id": scenario_id,
                        "metric": "agent_speech_fidelity",
                        "trial": trial,
                        "value": base - penalty,
                        "accent": accent,
                    }
                )
    return pd.DataFrame(rows)


def _scenario_values() -> pd.DataFrame:
    """Metric values with the accent group carried through, as the pipeline produces them."""
    return data_perturbations.build_condition_scenario_values(
        trial_scores=_trial_scores(),
        alias="agent",
        model_label="agent",
        condition_map=CONDITION_MAP,
        metrics=METRICS,
        group_col="accent",
    )


class TestGroupCarryThrough:
    def test_group_column_flows_through_scenario_values(self) -> None:
        values = _scenario_values()

        assert "accent" in values.columns
        # clean rows keep no label; the accent condition keeps one row per group
        assert values[values["condition"] == "clean"]["accent"].isna().all()
        assert (values["condition"] == "accent").sum() == 8

        # means stay within group instead of pooling both accents into one aggregate
        by_group = values[values["condition"] == "accent"].groupby("accent")["value"].mean()
        assert by_group["french"] == pytest.approx(0.7)
        assert by_group["indian"] == pytest.approx(0.3)

    def test_group_fairness_without_group_column_is_empty(self) -> None:
        ungrouped = data_perturbations.build_condition_scenario_values(
            trial_scores=_trial_scores(),
            alias="agent",
            model_label="agent",
            condition_map=CONDITION_MAP,
            metrics=METRICS,
        )
        assert "accent" not in ungrouped.columns

        per_group, summary = group_fairness_summary(ungrouped, CONFIG, group_col="accent")
        assert per_group.empty
        assert summary.empty


class TestWorstGroupAndGap:
    def test_group_fairness_summary_reports_worst_group_and_gap(self) -> None:
        per_group, summary = group_fairness_summary(_scenario_values(), CONFIG, group_col="accent")

        row = summary[summary["condition"] == "accent"].iloc[0]
        assert row["n_groups"] == 2
        assert row["n_scenarios"] == 8
        assert bool(row["lower_is_better"]) is False
        assert row["best_group"] == "french"
        assert row["worst_group"] == "indian"
        assert row["best_group_point"] == pytest.approx(0.7)
        assert row["worst_group_point"] == pytest.approx(0.3)
        # the aggregate a deployment decision would otherwise stop at sits between the groups
        assert row["aggregate_point"] == pytest.approx(0.5)
        assert row["group_gap"] == pytest.approx(0.4)
        # bootstrap resamples cannot close a uniform 0.4 gap, so the disparity is resolvable
        assert 0.0 < row["gap_ci_lower"] <= 0.4 <= row["gap_ci_upper"]
        assert bool(row["gap_ci_excludes_zero"]) is True
        assert row["worst_group_share"] == 1.0
        # the worst group's own interval brackets its point estimate
        assert row["worst_group_ci_lower"] < row["worst_group_point"] < row["worst_group_ci_upper"]

        assert set(per_group["group"]) == {"french", "indian"}
        assert len(per_group) == 2


class TestMetricOrientation:
    def test_error_rate_metrics_are_lower_is_better(self) -> None:
        assert metric_lower_is_better("stt_wer") is True
        assert metric_lower_is_better("interruption_rate") is True
        assert metric_lower_is_better("agent_speech_fidelity") is False
        assert metric_lower_is_better("custom_score", {"lower_is_better_metrics": ["custom_score"]}) is True
        # an explicit override beats name inference
        assert metric_lower_is_better("stt_wer", {"higher_is_better_metrics": ["stt_wer"]}) is False


class TestStatsMainEmitsGroupFairnessCsvs:
    def test_stats_main_writes_group_fairness_csvs(self, tmp_path: Path) -> None:
        """stats_perturbations.main() emits the group-fairness tables alongside its other outputs."""
        output_dir = tmp_path / "output_processed" / "test"
        output_dir.mkdir(parents=True)
        _scenario_values().to_csv(output_dir / "scenario_metricvalues.csv", index=False)
        data_perturbations.build_scenario_deltas(
            trial_scores=_trial_scores(),
            model_label="agent",
            alias="agent",
            condition_map=CONDITION_MAP,
            metrics=METRICS,
        ).to_csv(output_dir / "scenario_deltas.csv", index=False)

        config_path = tmp_path / "local" / "perturbations" / "perturbations_config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump({**CONFIG, "output_dir": "output_processed/test", "group_col": "accent"}))

        stats_perturbations.main(config_path)

        summary = pd.read_csv(output_dir / "results_groupfairness_summary.csv")
        assert set(summary["condition"]) == {"accent"}  # clean carries no group label
        assert summary.iloc[0]["worst_group"] == "indian"
        assert summary.iloc[0]["group_gap"] == pytest.approx(0.4)

        per_group = pd.read_csv(output_dir / "results_groupfairness_per_group.csv")
        assert set(per_group["group"]) == {"french", "indian"}
