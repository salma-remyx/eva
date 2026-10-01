"""Rank-stability primitives: how much confidence does a ranked table deserve?

Adapted from "How Reproducible Are Evaluation Conclusions? A Self-Audit of
LLM-Inferred Prompt Structure" (arXiv:2609.30074). Auditing a small-sample LLM
evaluation, the paper finds its ranked table looks far more definitive than the
evidence supports: under a joint cluster bootstrap only part of the ranking
holds, and two equally defensible merging rules move rows. It recommends
reporting rank stability and an executed sensitivity comparison alongside any
ranking — the primitives here, in the style of eva.utils.bootstrap:

  1. ``joint_bootstrap_ranks`` — cluster bootstrap over rows (scenarios). Every
     model is resampled with the SAME row indices, so the ranking question is
     asked jointly instead of model-by-model, and each model gets a rank-hold
     frequency: the fraction of replicates in which it keeps its observed rank.
  2. ``rule_sensitivity`` — re-ranks under a second defensible aggregation rule
     (equal weight per cluster instead of per row) and reports which ranks move.
  3. ``rank_stability_table`` — applies both to a long scenario-level score
     table (the leaderboard/analysis shape) and returns one audit row per model.

The paper's case study (prompt-structure inference, repeated campaigns,
endpoint withdrawals) has no analogue in this repo and is intentionally out of
scope; the primitives apply to any scenario-level score table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eva.utils.bootstrap import ALPHA, N_BOOT, run_seed

# Columns of the rank-audit table (see rank_stability_table).
RANK_STABILITY_COLUMNS: list[str] = [
    "model_label",
    "metric",
    "condition",
    "point",
    "rank",
    "rank_hold_freq",
    "rank_lo",
    "rank_hi",
    "rank_domain_weighted",
    "rank_changed",
    "n_models",
    "n_scenarios",
]


@dataclass(frozen=True)
class RankStability:
    """Rank audit for one score table (rows = resampling units, columns = models).

    Attributes:
        observed_rank: Average (fractional) rank in the observed data, 1 = best; tied models share a rank.
        rank_hold_freq: Fraction of bootstrap replicates where the model held its observed rank.
        rank_lo: Alpha/2 percentile of the replicate ranks.
        rank_hi: 1 - alpha/2 percentile of the replicate ranks.
        n_units: Number of resampling units (rows).
        n_boot: Number of bootstrap replicates run.
    """

    observed_rank: np.ndarray
    rank_hold_freq: np.ndarray
    rank_lo: np.ndarray
    rank_hi: np.ndarray
    n_units: int
    n_boot: int


def _column_means(scores: np.ndarray) -> np.ndarray:
    """Per-column mean over finite entries; a column with no finite entry is NaN."""
    finite = np.isfinite(scores)
    counts = finite.sum(axis=0)
    totals = np.where(finite, scores, 0.0).sum(axis=0)
    return np.divide(totals, counts, out=np.full(counts.shape, np.nan), where=counts > 0)


def _average_ranks(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    """Average (fractional) ranks, 1 = best; tied values share their mean rank; NaN is unranked."""
    out = np.full(values.shape, np.nan)
    finite_ix = np.flatnonzero(np.isfinite(values))
    if finite_ix.size == 0:
        return out
    vals = values[finite_ix]
    order = np.argsort(-vals if higher_is_better else vals, kind="stable")
    sorted_vals = vals[order]
    ranks = np.arange(1, vals.size + 1, dtype=float)
    # Average the ordinal positions inside each run of exactly equal values.
    start = 0
    while start < sorted_vals.size:
        stop = start
        while stop + 1 < sorted_vals.size and sorted_vals[stop + 1] == sorted_vals[start]:
            stop += 1
        if stop > start:
            ranks[start : stop + 1] = ranks[start : stop + 1].mean()
        start = stop + 1
    out[finite_ix[order]] = ranks
    return out


def joint_bootstrap_ranks(
    scores: np.ndarray,
    *,
    seed: int,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    higher_is_better: bool = True,
) -> RankStability:
    """Cluster bootstrap over rows: how often does each model's rank survive resampling?

    Rows are the resampling clusters (scenarios): each replicate draws row
    indices with replacement and applies the SAME indices to every model, so
    models stay comparable within a replicate. Missing entries (NaN) are
    skipped per model, and a model that a replicate leaves with no scores is
    simply unranked there rather than counted as losing its rank.

    Args:
        scores: Scenario-level scores, shape (n_units, n_models); NaN marks a model not scored on that unit.
        seed: RNG seed, for reproducible rank audits.
        n_boot: Number of bootstrap replicates.
        alpha: Percentile band for the replicate ranks, (alpha/2, 1 - alpha/2).
        higher_is_better: Rank descending (True) or ascending (False).

    Returns:
        RankStability with the per-model observed rank, rank-hold frequency, and replicate-rank band.

    Raises:
        ValueError: If scores is not 2D, has fewer than 2 rows, or a model column has no finite score.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D (units x models), got shape {scores.shape}")
    n_units, n_models = scores.shape
    if n_units < 2:
        raise ValueError(f"need >= 2 resampling units to bootstrap, got {n_units}")
    empty_col = np.flatnonzero(np.isfinite(scores).sum(axis=0) == 0)
    if empty_col.size:
        raise ValueError(f"model column {int(empty_col[0])} has no finite scores; drop it before ranking")

    observed_rank = _average_ranks(_column_means(scores), higher_is_better=higher_is_better)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_units, size=(n_boot, n_units))
    resampled = scores[idx]  # (n_boot, n_units, n_models): one shared draw per replicate
    finite = np.isfinite(resampled)
    counts = finite.sum(axis=1)
    rep_means = np.divide(
        np.where(finite, resampled, 0.0).sum(axis=1),
        counts,
        out=np.full(counts.shape, np.nan),
        where=counts > 0,
    )
    rep_ranks = np.stack([_average_ranks(row, higher_is_better=higher_is_better) for row in rep_means])

    rankable = np.isfinite(rep_ranks)
    held = (rep_ranks == observed_rank) & rankable
    denom = rankable.sum(axis=0)
    hold_freq = np.divide(held.sum(axis=0), denom, out=np.full(n_models, np.nan), where=denom > 0)

    return RankStability(
        observed_rank=observed_rank,
        rank_hold_freq=hold_freq,
        rank_lo=np.nanpercentile(rep_ranks, 100 * alpha / 2, axis=0),
        rank_hi=np.nanpercentile(rep_ranks, 100 * (1 - alpha / 2), axis=0),
        n_units=n_units,
        n_boot=n_boot,
    )


def rule_sensitivity(
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Ranks under two defensible aggregation rules: per-row and per-cluster weighting.

    Rule A weights every resampling unit equally (all scenarios concatenated);
    rule B weights every cluster equally (mean of per-cluster means). When
    clusters differ in size the two rules are equally defensible and can
    disagree — the executed sensitivity comparison the self-audit recommends
    reporting alongside any ranking.

    Args:
        scores: Scenario-level scores, shape (n_units, n_models); NaN marks a missing score.
        clusters: Cluster label (e.g. domain) per row, shape (n_units,).
        higher_is_better: Rank descending (True) or ascending (False).

    Returns:
        (ranks_rule_a, ranks_rule_b) as average ranks, 1 = best.
    """
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters)
    if scores.ndim != 2 or clusters.ndim != 1 or len(clusters) != len(scores):
        raise ValueError(f"scores/clusters shape mismatch: {scores.shape} vs {clusters.shape}")
    if len(scores) == 0:
        raise ValueError("need at least one row and one cluster label")

    ranks_rows = _average_ranks(_column_means(scores), higher_is_better=higher_is_better)
    cluster_means = np.stack([_column_means(scores[clusters == label]) for label in np.unique(clusters)])
    ranks_clusters = _average_ranks(_column_means(cluster_means), higher_is_better=higher_is_better)
    return ranks_rows, ranks_clusters


def rank_stability_table(
    values_long: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Rank audit of the model table: does each ranking survive resampling and a rule change?

    For every (metric, condition), pivots scenario-level values to a
    (domain, scenario) x model matrix and reports per model: the observed
    rank, its joint-cluster-bootstrap rank-hold frequency (every model gets
    the same resampled scenarios), and its rank under domain-weighted
    pooling, where a changed rank flags sensitivity to an equally defensible
    aggregation rule.

    Args:
        values_long: One row per (model, domain, condition, scenario, metric), with columns
            model_label, domain, condition, scenario_id, metric, value.
        n_boot: Bootstrap replicates per (metric, condition) cell.
        seed: Base seed; each cell derives its own deterministically, like the bootstrap CIs.
        alpha: Percentile band for the replicate ranks.

    Returns:
        One row per ranked model with columns as in RANK_STABILITY_COLUMNS, sorted by rank.
    """
    if values_long.empty:
        return pd.DataFrame(columns=RANK_STABILITY_COLUMNS)

    rows: list[dict] = []
    for (metric, condition), g in values_long.groupby(["metric", "condition"], sort=False):
        wide = g.pivot_table(index=["domain", "scenario_id"], columns="model_label", values="value", aggfunc="mean")
        wide = wide.dropna(axis="columns", how="all")
        if wide.shape[1] < 2 or len(wide) < 2:
            continue  # a ranking needs >= 2 models and the bootstrap needs >= 2 scenario units

        scores = wide.to_numpy(dtype=float)
        domains = wide.index.get_level_values("domain").to_numpy()
        cell_seed = run_seed(f"{seed}:rank:{metric}:{condition}")
        audit = joint_bootstrap_ranks(scores, seed=cell_seed, n_boot=n_boot, alpha=alpha)
        by_scenario, by_domain = rule_sensitivity(scores, domains)

        for col, model in enumerate(wide.columns):
            rows.append(
                {
                    "model_label": model,
                    "metric": metric,
                    "condition": condition,
                    "point": float(np.nanmean(scores[:, col])),
                    "rank": float(audit.observed_rank[col]),
                    "rank_hold_freq": float(audit.rank_hold_freq[col]),
                    "rank_lo": float(audit.rank_lo[col]),
                    "rank_hi": float(audit.rank_hi[col]),
                    "rank_domain_weighted": float(by_domain[col]),
                    "rank_changed": bool(by_scenario[col] != by_domain[col]),
                    "n_models": wide.shape[1],
                    "n_scenarios": len(wide),
                }
            )

    out = pd.DataFrame(rows, columns=RANK_STABILITY_COLUMNS)
    return out.sort_values(["metric", "condition", "rank"], kind="stable").reset_index(drop=True)
