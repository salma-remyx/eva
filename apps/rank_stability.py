"""Cluster-bootstrap rank stability for the cross-run comparison table.

Mean columns rank systems, but they say nothing about how much evidence backs
each row. This module resamples scenarios jointly — every system is scored on
the same resampled scenario set — and reports how often each system holds its
observed rank, how often it ranks first, and how often the observed winner of
a matchup still wins. Low rank-hold rates mean an ordering that looks settled
in the mean table is not actually settled by the data.

Adapted from "How Reproducible Are Evaluation Conclusions? A Self-Audit of
LLM-Inferred Prompt Structure" (arXiv:2609.30074), whose audit of a ranked
table found only its bottom was firm while its top was not.

The resampling unit matches eva's per-run bootstrap spine: scenarios are base
record ids recovered with ``parse_trial_record_id``, the same clustering
``eva.metrics.aggregation`` uses for its ``mean_ci_*`` fields, and seeding
follows ``eva.utils.bootstrap.run_seed`` so audits are stable across app
reloads.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from eva.utils.bootstrap import N_BOOT, run_seed
from eva.utils.pass_at_k import parse_trial_record_id

DECIMALS = 4
MIN_SCENARIOS = 2


@dataclass(frozen=True)
class RankStability:
    """Rank-audit result for one metric across systems."""

    metric: str
    n_systems: int
    n_scenarios: int
    n_dropped_scenarios: int
    n_boot: int
    observed_rank: dict[str, int]
    observed_mean: dict[str, float]
    rank_hold_rate: dict[str, float]
    best_rate: dict[str, float]
    win_rate: dict[tuple[str, str], float]

    def system_rows(self) -> list[dict[str, object]]:
        """Per-system summary rows ordered by observed rank (best first)."""
        ordered = sorted(self.observed_rank, key=lambda s: self.observed_rank[s])
        return [
            {
                "system": s,
                "mean": self.observed_mean[s],
                "rank": self.observed_rank[s],
                "rank_hold_rate": self.rank_hold_rate[s],
                "best_rate": self.best_rate[s],
            }
            for s in ordered
        ]

    def pairwise_rows(self) -> list[dict[str, object]]:
        """Adjacent-rank matchups with the observed winner's replicate win rate."""
        ordered = sorted(self.observed_rank, key=lambda s: self.observed_rank[s])
        rows: list[dict[str, object]] = []
        for better, worse in itertools.pairwise(ordered):
            rate = self.win_rate.get((better, worse))
            if rate is not None:
                rows.append({"winner": better, "loser": worse, "win_rate": rate})
        return rows


def rank_stability_from_rows(
    rows: Iterable[Mapping[str, object]],
    metric: str,
    *,
    higher_is_better: bool = True,
    n_boot: int = N_BOOT,
    seed: int | None = None,
) -> RankStability | None:
    """Compute rank stability from per-record rows, as collected for the heatmap.

    ``rows`` follow the ``_collect_run_metrics`` shape: each row carries
    ``system`` (display label), ``record`` (record id; multiple trials share
    it), and this row's value under ``metric``. Rows for the same system and
    scenario are averaged, matching the per-scenario means behind the run-level
    ``mean_ci_*`` fields.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        system = row.get("system")
        value = row.get(metric)
        if system is None or not isinstance(value, (int, float)):
            continue
        base_id, _ = parse_trial_record_id(str(row.get("record", "")))
        grouped.setdefault((str(system), base_id), []).append(float(value))

    scenario_scores: dict[str, dict[str, float]] = {}
    for (system, base_id), values in grouped.items():
        scenario_scores.setdefault(system, {})[base_id] = sum(values) / len(values)

    return rank_stability_from_matrix(
        scenario_scores,
        metric=metric,
        higher_is_better=higher_is_better,
        n_boot=n_boot,
        seed=seed,
    )


def rank_stability_from_matrix(
    scenario_scores: Mapping[str, Mapping[str, float]],
    *,
    metric: str = "",
    higher_is_better: bool = True,
    n_boot: int = N_BOOT,
    seed: int | None = None,
) -> RankStability | None:
    """Joint cluster bootstrap of system ranks over the scenarios they share.

    Every replicate resamples scenario ids with replacement and re-ranks all
    systems on that shared resample, so differences track the scenario sample
    rather than per-system noise. Scenarios missing from any system are dropped
    (and counted) because the resample index must pair systems on the same
    scenarios. Returns ``None`` when fewer than two systems or fewer than
    ``MIN_SCENARIOS`` shared scenarios are available.
    """
    systems = list(scenario_scores)
    if len(systems) < 2:
        return None
    shared, all_scenarios, matrix = _scenario_matrix(scenario_scores)
    if len(shared) < MIN_SCENARIOS:
        return None
    if seed is None:
        # Deterministic across app reloads without depending on session state.
        seed = run_seed(f"{metric}|{'|'.join(sorted(systems))}")

    orientation = 1.0 if higher_is_better else -1.0
    observed = matrix.mean(axis=1) * orientation  # bigger = better
    order = np.argsort(-observed, kind="stable")
    observed_rank = {systems[i]: rank for rank, i in enumerate(order, start=1)}
    observed_mean = {s: round(float(matrix[i].mean()), DECIMALS) for i, s in enumerate(systems)}

    rng = np.random.default_rng(seed)
    resample = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    replicates = matrix[:, resample].mean(axis=2) * orientation  # (systems, n_boot)

    ranks = np.empty((len(systems), n_boot), dtype=int)
    rep_order = np.argsort(-replicates, axis=0, kind="stable")
    ranks[rep_order, np.arange(n_boot)] = np.arange(1, len(systems) + 1)[:, None]

    observed_ranks = np.array([observed_rank[s] for s in systems])[:, None]
    hold = (ranks == observed_ranks).mean(axis=1)
    best = (ranks == 1).mean(axis=1)

    position = {s: i for i, s in enumerate(systems)}
    ordered_systems = [systems[i] for i in order]
    win_rate: dict[tuple[str, str], float] = {}
    for better, worse in itertools.combinations(ordered_systems, 2):
        wins = (replicates[position[better]] > replicates[position[worse]]).mean()
        win_rate[(better, worse)] = round(float(wins), DECIMALS)

    return RankStability(
        metric=metric,
        n_systems=len(systems),
        n_scenarios=len(shared),
        n_dropped_scenarios=len(all_scenarios) - len(shared),
        n_boot=n_boot,
        observed_rank=observed_rank,
        observed_mean=observed_mean,
        rank_hold_rate={s: round(float(hold[position[s]]), DECIMALS) for s in systems},
        best_rate={s: round(float(best[position[s]]), DECIMALS) for s in systems},
        win_rate=win_rate,
    )


def _scenario_matrix(
    scenario_scores: Mapping[str, Mapping[str, float]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Align per-system scenario means on the scenarios every system covers.

    Returns (shared scenario ids, all scenario ids seen, system x
    shared-scenario mean matrix in first-seen system order).
    """
    scenario_sets = [set(scores) for scores in scenario_scores.values()]
    shared = sorted(set.intersection(*scenario_sets))
    all_scenarios = sorted(set.union(*scenario_sets))
    matrix = np.array([[scenario_scores[s][sid] for sid in shared] for s in scenario_scores], dtype=float)
    return shared, all_scenarios, matrix
