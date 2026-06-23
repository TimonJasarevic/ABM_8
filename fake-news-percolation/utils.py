"""Reconstructed analysis helpers for EDA.ipynb.

The original ``utils.py`` from the deepfake-percolation project was not included in
this extraction, so these are best-effort, standard reconstructions of the three
functions the notebook imports. ``bimodality_coefficient`` is the unambiguous Sarle
formula; ``sen_welfare`` and ``polarization`` are standard definitions that may
differ numerically from the author's originals -- swap them in if the real
``utils.py`` ever turns up.

Pure-numpy by design (no scipy / sklearn) so the module stays portable.
"""

import numpy as np


def _gini(x):
    """Gini coefficient of non-negative values (0 = perfect equality)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n


def sen_welfare(x):
    """Sen social welfare: mean welfare discounted by inequality, W = mean * (1 - Gini).

    ``x`` is per-agent normalized payoff (``agent_veracity``, in [-1, 1]). The mean is
    kept as the welfare *level* (it may be negative); only the Gini inequality term
    shifts the values to be non-negative.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(x.mean() * (1.0 - _gini(x - x.min())))


def polarization(x):
    """Belief polarization as normalized variance on [0, 1]-bounded beliefs.

    ``4 * Var(x)``: 0 = full consensus, 1 = maximally split (half at 0, half at 1).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(4.0 * np.var(x))


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient.

    ``BC = (g1**2 + 1) / (g2 + 3*(n-1)**2 / ((n-2)*(n-3)))`` with bias-corrected sample
    skewness ``g1`` and excess kurtosis ``g2`` (matches Excel SKEW/KURT, scipy
    ``bias=False``). ``BC > 5/9`` (~0.555) suggests bimodality. Returns ``nan`` when
    undefined (n < 4 or zero spread).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return np.nan
    s = x.std(ddof=1)
    if s == 0:
        return np.nan
    z = (x - x.mean()) / s
    g1 = (n / ((n - 1.0) * (n - 2.0))) * np.sum(z ** 3)
    g2 = ((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * np.sum(z ** 4) \
        - (3.0 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
    denom = g2 + (3.0 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
    return float((g1 ** 2 + 1.0) / denom)
