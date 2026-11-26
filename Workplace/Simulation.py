"""Simulation helpers for the wealth-distribution exercises.

The main function, :func:`simulate_wealth`, builds one artificial economy of
``n`` adults. Age groups are drawn from the supplied age shares and wealth is
assigned either from each group's median or from a randomly drawn percentile.
"""

from __future__ import annotations

import numpy as np


def simulate_wealth(age_shares, thresholds, n=10_000, seed=42, within_group=False):
    """Simulate ages and net wealth for ``n`` adults.

    Parameters
    ----------
    age_shares : array_like of float
        Eight probabilities in the same order as the rows of `thresholds`.
        They must be non-negative and sum to one.
    thresholds : array_like of float, shape (8, 99)
        Age-specific wealth thresholds. Row ``j`` contains P1, ..., P99 for
        age group ``j``; P1 is column 0.
    n : int, optional
        Number of simulated adults. Default is 10,000.
    seed : int, optional
        Seed passed to ``np.random.default_rng``. Default is 42.
    within_group : bool, optional
        If False, every adult receives the P50 (column 49) of their age group.
        If True, each adult draws a rank from 1 to 99 with equal probability
        and receives that percentile of their age group. Default is False.

    Returns
    -------
    age : ndarray of int, shape (n,)
        Index of the simulated age group for each adult.
    wealth : ndarray of float, shape (n,)
        Simulated net wealth in DKK.

    Notes
    -----
    Ages are drawn first: a uniform draw ``u`` on [0, 1) is placed in the
    interval whose length equals the corresponding age share. When
    ``within_group`` is True, the 1-99 ranks are drawn afterwards, so two runs
    with the same seed share both the uniform draws and the rank draws.

    Examples
    --------
    >>> age, wealth = simulate_wealth([0.5] + [0.5 / 7] * 7,
    ...                                np.zeros((8, 99)))
    >>> age.shape, wealth.shape
    ((10000,), (10000,))
    """
    age_shares = np.asarray(age_shares, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if age_shares.ndim != 1:
        raise ValueError("age_shares must be a one-dimensional array")
    if thresholds.shape != (age_shares.size, 99):
        raise ValueError("thresholds must have shape (number of age groups, 99)")
    if np.any(age_shares < 0):
        raise ValueError("age_shares cannot be negative")
    if not np.isclose(age_shares.sum(), 1.0):
        raise ValueError("age_shares must sum to one")

    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n)

    age = np.empty(n, dtype=int)
    lower = 0.0
    for j, share in enumerate(age_shares):
        upper = lower + share
        if j == age_shares.size - 1:
            # The last interval is [lower, 1); u is always below 1.
            mask = u >= lower
        else:
            mask = (u >= lower) & (u < upper)
        age[mask] = j
        lower = upper

    if within_group:
        ranks = rng.integers(1, 100, size=n)
        wealth = thresholds[age, ranks - 1]
    else:
        wealth = thresholds[age, 49]  # column 49 is P50 because P1 is column 0

    return age, wealth
