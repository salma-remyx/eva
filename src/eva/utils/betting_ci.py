"""Betting-based confidence intervals for the mean of a bounded metric.

EVA metric scores are bounded (normalized scores live in ``[0, 1]``), and the
per-scenario sample sizes are small (~200 scenarios, far fewer per slice). In
that regime the percentile bootstrap in :mod:`eva.utils.bootstrap` is convenient
but gives no finite-sample coverage guarantee, while classical valid intervals
(Hoeffding) are needlessly wide.

This module implements a *testing-by-betting* confidence interval for the mean
of a bounded random variable: a candidate mean ``m`` is retained iff a hedged
capital (wealth) process betting against ``H0: mean == m`` never grows past
``1/alpha``. The retained set is a coverage-valid ``(1 - alpha)`` interval that
is markedly tighter than Hoeffding for typical low-variance metric samples.

The construction follows Waudby-Smith & Ramdas, *Estimating means of bounded
random variables by betting* (JRSS-B 2024), the framework that the STaR-Bets
paper (*Sequential Target-Recalculating Bets for Tighter Confidence Intervals*,
arXiv:2505.22422) builds upon. We implement the fixed-sample hedged-capital
interval with predictable-mixture bets, which delivers the paper's core result —
tighter, coverage-valid intervals for bounded means — without porting its
sequential target-recalculation machinery, which targets the streaming setting
EVA does not use.

Because the wealth process is order-dependent, the interval is derandomized by
averaging the wealth over several random orderings of the sample (a convex
combination of valid test martingales), keeping the result robust to how the
per-scenario values happen to be arranged. Ordering is fixed by ``seed`` so the
reported bounds are reproducible across runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

ALPHA = 0.05
# Truncation constant keeping every wealth factor strictly positive (Hedged-Capital
# "c" in Waudby-Smith & Ramdas); 0.5 is the value they recommend.
BET_FRACTION_CAP = 0.5
# Random orderings averaged over to derandomize the order-dependent wealth process.
N_PERMUTATIONS = 16
# Grid used to bracket the interval boundaries before bisection refinement.
GRID_SIZE = 200
# Bisection tolerance (in the rescaled [0, 1] space); 1e-5 is well below the 4
# decimals the aggregation layer rounds CI bounds to.
BISECT_TOL = 1e-5


def _predictable_bets(u: np.ndarray, alpha: float) -> np.ndarray:
    """Predictable-mixture betting fractions for rescaled samples ``u`` in ``[0, 1]``.

    Each ``lambda_t`` depends only on ``u[:t]`` (it is *predictable*), so the
    resulting wealth process is a valid test (super)martingale under ``H0``.
    """
    n = len(u)
    log_term = 2.0 * math.log(2.0 / alpha)
    bets = np.empty(n, dtype=float)
    running_sum = 0.0
    running_sq_dev = 0.0
    # sigma_hat^2 with a 1/4 prior; index 0 uses the prior only.
    sigma2_prev = 0.25
    for t in range(1, n + 1):
        # Fixed-sample predictable-mixture bet (scales with n, not t): tuned for the
        # batch interval EVA needs rather than an anytime-valid sequence.
        denom = sigma2_prev * n
        bets[t - 1] = math.sqrt(log_term / denom) if denom > 0 else 0.0
        x_t = float(u[t - 1])
        running_sum += x_t
        mu_hat_t = (0.5 + running_sum) / (1.0 + t)
        running_sq_dev += (x_t - mu_hat_t) ** 2
        sigma2_prev = (0.25 + running_sq_dev) / (1.0 + t)
    return bets


def _rejects(orderings: list[tuple[np.ndarray, np.ndarray]], p: float, alpha: float) -> bool:
    """Whether the derandomized hedged capital rejects candidate mean ``p`` (rescaled).

    ``orderings`` is a list of ``(u, bets)`` pairs, one per random permutation. The
    wealth processes are averaged across orderings before the ``1/alpha`` crossing
    test, so a single unlucky arrangement of the sample cannot drive the decision.
    """
    if p <= 0.0:
        return any(bool(np.any(u > 0.0)) for u, _ in orderings)
    if p >= 1.0:
        return any(bool(np.any(u < 1.0)) for u, _ in orderings)
    mean_wealth = np.zeros(len(orderings[0][0]), dtype=float)
    for u, bets in orderings:
        dev = u - p
        # Two one-sided test martingales with side-specific bet truncations that keep
        # each wealth factor strictly positive (Waudby-Smith & Ramdas hedged capital):
        # the "up" process needs lambda < 1/p, the "down" process needs lambda < 1/(1-p).
        lam_up = np.minimum(bets, BET_FRACTION_CAP / p)
        lam_down = np.minimum(bets, BET_FRACTION_CAP / (1.0 - p))
        capital_up = np.cumprod(1.0 + lam_up * dev)
        capital_down = np.cumprod(1.0 - lam_down * dev)
        # Even 50/50 hedge of the two martingales.
        mean_wealth += 0.5 * (capital_up + capital_down)
    mean_wealth /= len(orderings)
    return bool(np.max(mean_wealth) >= 1.0 / alpha)


def _refine_boundary(
    orderings: list[tuple[np.ndarray, np.ndarray]], alpha: float, rejected: float, retained: float
) -> float:
    """Bisect the rejection boundary between a rejected and a retained candidate."""
    for _ in range(64):
        if abs(retained - rejected) <= BISECT_TOL:
            break
        mid = 0.5 * (rejected + retained)
        if _rejects(orderings, mid, alpha):
            rejected = mid
        else:
            retained = mid
    return retained


def betting_ci(
    values: Sequence[float] | np.ndarray,
    *,
    alpha: float = ALPHA,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Coverage-valid ``(1 - alpha)`` betting confidence interval on the mean.

    ``values`` are assumed bounded within ``[lower_bound, upper_bound]``; the
    bounds are widened automatically if the data fall outside them so the call is
    robust to metrics that exceed the nominal ``[0, 1]`` range. ``seed`` fixes the
    random orderings used for derandomization, making the bounds reproducible.
    Returns ``(None, None)`` for an empty sample. The returned interval always
    contains the sample mean.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None

    lo = min(lower_bound, float(arr.min()))
    hi = max(upper_bound, float(arr.max()))
    if hi <= lo:
        # Degenerate support: every observation equals the single feasible value.
        return lo, lo

    span = hi - lo
    u = (arr - lo) / span
    p_hat = float(u.mean())

    rng = np.random.default_rng(seed)
    orderings: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(N_PERMUTATIONS):
        permuted = u[rng.permutation(u.size)]
        orderings.append((permuted, _predictable_bets(permuted, alpha)))

    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    retained = [g for g in grid if not _rejects(orderings, float(g), alpha)]
    if not retained:
        # No grid point survived (extreme small-n case); fall back to the estimate.
        return lo + p_hat * span, lo + p_hat * span

    lo_p = min(min(retained), p_hat)
    hi_p = max(max(retained), p_hat)
    step = grid[1] - grid[0]
    # Refine each boundary against the adjacent rejected candidate for precision.
    if lo_p > 0.0 and _rejects(orderings, max(lo_p - step, 0.0), alpha):
        lo_p = _refine_boundary(orderings, alpha, rejected=max(lo_p - step, 0.0), retained=lo_p)
    if hi_p < 1.0 and _rejects(orderings, min(hi_p + step, 1.0), alpha):
        hi_p = _refine_boundary(orderings, alpha, rejected=min(hi_p + step, 1.0), retained=hi_p)

    return lo + lo_p * span, lo + hi_p * span
