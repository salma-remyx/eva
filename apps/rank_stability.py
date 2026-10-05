"""Rank-stability audit for the cross-run comparison table.

A ranked table of mean scores can look far more definitive than its evidence
supports (see "How Reproducible Are Evaluation Conclusions? A Self-Audit of
LLM-Inferred Prompt Structure", arXiv:2609.30074): under a bootstrap over the
scenario sample, middle-of-the-table rows routinely swap places. This module
re-runs the ranking inside a joint cluster bootstrap — the same resampled
record indices are applied to every system, so score differences reflect the
shared scenario sample rather than independent resampling noise — and reports
how often each system holds its observed rank, plus the pairwise probability
that one system outscores another.

A row is flagged unstable when it holds its rank in fewer than
``FIRM_RANK_HOLD`` of replicates (calibrated against the paper's reported
levels: 86–99% read as firm, 27–68% as unstable). An unstable top of the
table means the leaderboard identifies the worst system reliably without
reliably identifying the best.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eva.utils.bootstrap import N_BOOT, run_seed

# Share of bootstrap replicates a system must hold its rank in to be "firm".
FIRM_RANK_HOLD = 0.75

# Metric columns offered first in the stability selector (EVA composites read
# better as a headline ranking than individual metrics).
_PREFERRED_METRICS = ("EVA-A_mean", "EVA-A_pass", "EVA-X_mean", "EVA-X_pass")


@dataclass
class RankStability:
    """Rank-stability audit for one metric across systems."""

    metric: str
    systems: list[str]  # best first
    means: dict[str, float]
    observed_ranks: dict[str, int]  # 1 = best
    rank_hold: dict[str, float]  # P(system keeps its observed rank)
    pairwise: dict[tuple[str, str], float]  # (row, col) -> P(row outscores col)
    n_records: int  # number of shared records the bootstrap ran over


def _numeric(value: object) -> float | None:
    """Return ``value`` as a float, or None when it is not a usable score."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if np.isnan(number) else number


def compute_rank_stability(
    rows: list[dict],
    metric: str,
    *,
    higher_is_better: bool = True,
    n_boot: int = N_BOOT,
    seed: int | None = None,
) -> RankStability | None:
    """Joint cluster bootstrap over the shared records of ``rows``.

    ``rows`` follows the cross-run comparison's per-record shape (``system``,
    ``record``, one column per metric; trials of the same record are averaged,
    making the record the resampling cluster). Records missing the metric for
    any system are dropped so every system is ranked on the same scenario
    sample. Returns None when fewer than two systems share two or more records.
    """
    systems = sorted({str(r["system"]) for r in rows if r.get("system") is not None})
    if len(systems) < 2:
        return None

    # record -> system -> per-trial values
    values_by_record: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        record, system = row.get("record"), row.get("system")
        value = _numeric(row.get(metric))
        if record is None or system is None or value is None:
            continue
        values_by_record.setdefault(str(record), {}).setdefault(str(system), []).append(value)

    shared = [rec for rec, per_system in values_by_record.items() if all(s in per_system for s in systems)]
    if len(shared) < 2:
        return None

    # (systems, records) matrix of per-record means; rows follow `systems` order.
    matrix = np.array([[float(np.mean(values_by_record[rec][s])) for rec in shared] for s in systems])
    means = {s: float(matrix[i].mean()) for i, s in enumerate(systems)}

    def _scores(values: np.ndarray) -> np.ndarray:
        return values if higher_is_better else -values

    # Observed ranking (1 = best), ties broken alphabetically to match the
    # stable sort used for the bootstrap replicates below.
    observed_scores = _scores(matrix).mean(axis=1)
    observed_order = sorted(range(len(systems)), key=lambda i: (-float(observed_scores[i]), systems[i]))
    observed_pos = {systems[i]: pos for pos, i in enumerate(observed_order)}

    rng = np.random.default_rng(seed if seed is not None else run_seed(f"rank_stability:{metric}"))
    idx = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    boot_means = _scores(matrix[:, idx].mean(axis=2))  # (systems, n_boot)

    # Rank of each system in each replicate (0 = best), alphabetical tie-break.
    order = np.argsort(-boot_means, axis=0, kind="stable")
    ranks = np.empty_like(order)
    ranks[order, np.arange(n_boot)] = np.arange(len(systems))[:, None]
    rank_hold = {s: float((ranks[i] == observed_pos[s]).mean()) for i, s in enumerate(systems)}

    pairwise: dict[tuple[str, str], float] = {}
    for i, row_sys in enumerate(systems):
        for j, col_sys in enumerate(systems):
            diff = boot_means[i] - boot_means[j]
            wins = int((diff > 0).sum()) + 0.5 * int((diff == 0).sum())
            pairwise[(row_sys, col_sys)] = wins / n_boot

    ordered = [systems[i] for i in observed_order]
    return RankStability(
        metric=metric,
        systems=ordered,
        means=means,
        observed_ranks={s: observed_pos[s] + 1 for s in systems},
        rank_hold=rank_hold,
        pairwise=pairwise,
        n_records=len(shared),
    )


def stability_frame(result: RankStability) -> pd.DataFrame:
    """Build the per-system stability table (one row per system, best first)."""
    return pd.DataFrame(
        [
            {
                "System": s,
                "Mean": result.means[s],
                "Rank": result.observed_ranks[s],
                "Rank Hold": result.rank_hold[s],
                "Verdict": "firm" if result.rank_hold[s] >= FIRM_RANK_HOLD else "unstable",
            }
            for s in result.systems
        ]
    )


def pairwise_frame(result: RankStability) -> pd.DataFrame:
    """Build the P(row outscores column) matrix as a labelled DataFrame."""
    return pd.DataFrame(
        [[result.pairwise[(row, col)] if row != col else np.nan for col in result.systems] for row in result.systems],
        index=pd.Index(result.systems, name="System"),
        columns=result.systems,
    )


def render_rank_stability_section(
    rows: list[dict],
    ordered_metrics: list[str],
    is_lower_is_better: Callable[[str], bool],
    *,
    n_boot: int = N_BOOT,
) -> None:
    """Render the rank-stability tables below the cross-run comparison.

    The streamlit import is deferred so the computation above stays importable
    (and testable) without the app extras installed.
    """
    import streamlit as st

    frame = pd.DataFrame(rows)
    candidates = [c for c in _PREFERRED_METRICS if c in frame.columns]
    candidates += [m for m in ordered_metrics if m in frame.columns and m not in candidates]
    if not st.session_state.get("show_sub_metrics"):
        candidates = [m for m in candidates if "__" not in m]
    if not candidates:
        return

    default = candidates.index("EVA-A_mean") if "EVA-A_mean" in candidates else 0
    metric = st.selectbox("Metric", candidates, index=default)
    result = compute_rank_stability(rows, metric, higher_is_better=not is_lower_is_better(metric), n_boot=n_boot)

    st.markdown("#### Rank Stability")
    if result is None:
        st.info(
            "Rank stability needs at least two systems that share two or more records with values for this metric."
        )
        return

    st.caption(
        f"Self-audit of the table above: joint cluster bootstrap over the {result.n_records} shared records "
        f"({n_boot:,} replicates), applying the same resampled record indices to every system. "
        "**Rank Hold** is the share of replicates in which a system keeps its rank; the pairwise table gives "
        f"the probability that the row system outscores the column one. Rows holding rank in under "
        f"{FIRM_RANK_HOLD:.0%} of replicates are unstable — the ranking may identify the worst system reliably "
        "without reliably identifying the best."
    )
    stability = stability_frame(result).style.format({"Mean": "{:.3f}", "Rank Hold": "{:.0%}"})
    st.dataframe(stability, hide_index=True)
    pairwise = pairwise_frame(result).style.format("{:.2f}", na_rep="—")
    st.dataframe(pairwise)
