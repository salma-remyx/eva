"""Rank-stability audit for ranked evaluation tables.

Adapted from "How Reproducible Are Evaluation Conclusions? A Self-Audit of
LLM-Inferred Prompt Structure" (arXiv:2609.30074), which recommends reporting
rank stability alongside any ranked table: under a joint cluster bootstrap over
the shared sampling units (prompts in the paper, scenarios here), each row is
scored by how often it holds its observed rank, so the reader can tell which
rows of the table are firm and which are noise. The paper's own audit found the
bottom of a ranking firm (86-99% of replicates) while the middle held rank in
only 27-48% — a table can identify the worst system reliably without reliably
identifying the best.

Pure computation: takes per-unit values, returns dicts/arrays. No file I/O. The
bootstrap is *joint* — every replicate draws one set of unit ids shared by all
systems, preserving the paired design in which each system is scored on the same
scenarios. Seeds follow the same discipline as eva.utils.bootstrap, so an audit
of the same runs reproduces exactly.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from eva.utils.bootstrap import N_BOOT

# A row is "firm" when it holds its observed rank in at least this share of
# replicates.
FIRM_RANK_HOLD = 0.90


def _average_ranks_desc(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks of ``values``, highest first, ties sharing their average rank.

    NaN entries (a system with no value to rank) stay NaN rather than sorting last.
    """
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    ranks[np.isnan(values)] = np.nan
    return ranks


def _rank_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[bool, float]:
    """Sort table rows by observed rank, keeping unranked systems last."""
    rank = item[1]["rank"]
    return (rank is None, float("inf") if rank is None else rank)


def _unit_matrix(per_unit_values: dict[str, dict[str, float]]) -> tuple[list[str], list[str], np.ndarray]:
    """Align per-system unit values into a (n_systems, n_units) matrix, NaN where a cell is missing.

    Units are keyed (not positional) so callers cannot silently misalign two
    runs that covered their scenarios in different orders.
    """
    systems = list(per_unit_values)
    unit_ids = sorted({unit for values in per_unit_values.values() for unit in values})
    matrix = np.full((len(systems), len(unit_ids)), np.nan)
    for row, system in enumerate(systems):
        values = per_unit_values[system]
        for col, unit in enumerate(unit_ids):
            if unit in values:
                matrix[row, col] = values[unit]
    return systems, unit_ids, matrix


def joint_cluster_bootstrap_ranks(
    per_unit_values: dict[str, dict[str, float]],
    *,
    n_boot: int = N_BOOT,
    seed: int,
) -> dict[str, np.ndarray]:
    """Per-replicate rank of each system under a joint cluster bootstrap over units.

    Each replicate resamples the unit ids with replacement and ranks every system
    by its mean over the drawn units (1 = best, higher values = better). One draw
    is shared by all systems per replicate, so a unit that is hard for everyone
    is hard for everyone within the same replicate.

    Args:
        per_unit_values: system label -> {unit id: value}. Units missing for a
            system are excluded from that system's replicate means; a system with
            no value among the drawn units has no rank in that replicate.
        n_boot: Number of bootstrap replicates.
        seed: Seed for the shared unit draws.

    Returns:
        Dict mapping system label to an array of ``n_boot`` replicate ranks.
    """
    systems, unit_ids, matrix = _unit_matrix(per_unit_values)
    if not systems or not unit_ids:
        return {system: np.array([], dtype=float) for system in systems}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unit_ids), size=(n_boot, len(unit_ids)))

    ranks = np.empty((len(systems), n_boot))
    with warnings.catch_warnings():
        # A replicate whose drawn units are all missing for a system yields an
        # all-NaN slice; nanmean warns before returning the NaN we keep.
        warnings.simplefilter("ignore", RuntimeWarning)
        for replicate in range(n_boot):
            means = np.nanmean(matrix[:, draws[replicate]], axis=1)
            ranks[:, replicate] = _average_ranks_desc(means)
    return {system: ranks[row] for row, system in enumerate(systems)}


def rank_stability_table(
    per_unit_values: dict[str, dict[str, float]],
    *,
    seed: int,
    n_boot: int = N_BOOT,
    firm_threshold: float = FIRM_RANK_HOLD,
) -> dict[str, dict[str, Any]]:
    """Observed ranking plus per-row rank stability and per-cell provenance.

    This is the audit the run-level mean CI cannot give: a CI on the mean says
    where a system's score lies, not whether the ranking that follows from it
    survives resampling the scenarios it was measured on.

    Args:
        per_unit_values: system label -> {unit id: value} (see
            joint_cluster_bootstrap_ranks).
        seed: Seed for the joint bootstrap, as in eva.utils.bootstrap.
        n_boot: Number of bootstrap replicates.
        firm_threshold: Minimum ``rank_hold_share`` for a row to be flagged firm.

    Returns:
        Dict keyed by system label and ordered by observed rank. Each row holds
        mean, rank, rank_hold_share, share_rank_1 (outright best),
        share_at_or_better, mean_bootstrap_rank, rank_ci_lower, rank_ci_upper
        (2.5/97.5 percentiles of the replicate ranks), n_units, n_missing_units
        and firm. Systems with no values get None scores rather than a rank.
    """
    systems, unit_ids, matrix = _unit_matrix(per_unit_values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN row == system with no values
        observed_means = np.nanmean(matrix, axis=1)
    observed_ranks = _average_ranks_desc(observed_means)
    bootstrap_ranks = joint_cluster_bootstrap_ranks(per_unit_values, n_boot=n_boot, seed=seed)

    rows: dict[str, dict[str, Any]] = {}
    for row, system in enumerate(systems):
        replicates = bootstrap_ranks[system]
        scored = replicates[~np.isnan(replicates)]
        observed_rank = float(observed_ranks[row])
        holds = float(np.mean(scored == observed_rank)) if len(scored) else None
        rows[system] = {
            "mean": round(float(observed_means[row]), 4) if np.isfinite(observed_means[row]) else None,
            # A system with no values sorts last but is not ranked.
            "rank": observed_rank if np.isfinite(observed_means[row]) else None,
            "rank_hold_share": round(holds, 4) if holds is not None else None,
            "share_rank_1": round(float(np.mean(scored == 1.0)), 4) if len(scored) else None,
            "share_at_or_better": round(float(np.mean(scored <= observed_rank)), 4) if len(scored) else None,
            "mean_bootstrap_rank": round(float(np.mean(scored)), 4) if len(scored) else None,
            "rank_ci_lower": round(float(np.percentile(scored, 2.5)), 4) if len(scored) else None,
            "rank_ci_upper": round(float(np.percentile(scored, 97.5)), 4) if len(scored) else None,
            "n_units": len(per_unit_values[system]),
            "n_missing_units": len(unit_ids) - len(per_unit_values[system]),
            "firm": bool(holds is not None and holds >= firm_threshold),
        }
    return dict(sorted(rows.items(), key=_rank_sort_key))


def stability_summary(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Table-level verdict on how much confidence the ranking deserves.

    Args:
        table: Output of rank_stability_table.

    Returns:
        Dict with n_systems, n_units, n_firm_rows, firm_share, best_system,
        best_system_firm and worst_system_firm. The two flags routinely
        disagree, which is the paper's headline: the bottom of a ranking tends
        to be firm while the top is not.
    """
    if not table:
        return {
            "n_systems": 0,
            "n_units": 0,
            "n_firm_rows": 0,
            "firm_share": None,
            "best_system": None,
            "best_system_firm": None,
            "worst_system": None,
            "worst_system_firm": None,
        }

    ranked = [system for system, _ in sorted(table.items(), key=_rank_sort_key)]
    best_system = ranked[0]
    worst_system = ranked[-1]
    n_firm = sum(1 for row in table.values() if row["firm"])
    return {
        "n_systems": len(table),
        "n_units": max(row["n_units"] + row["n_missing_units"] for row in table.values()),
        "n_firm_rows": n_firm,
        "firm_share": round(n_firm / len(table), 4),
        "best_system": best_system,
        "best_system_firm": bool(table[best_system]["firm"]),
        "worst_system": worst_system,
        "worst_system_firm": bool(table[worst_system]["firm"]),
    }
