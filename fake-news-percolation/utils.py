"""Analysis helpers for the deepfake-percolation EDA / evidence scripts.

Originally a best-effort reconstruction of three notebook helpers; now extended with the
methodological fixes.

Key references (verified): Raffinetti, Siletti & Vernizzi (2015) renormalised Gini;
Sen (1976) / Stark (2025) welfare; Lakens (2013) d_av; Kirby & Gerlanc (2013) bootstrap CIs;
Benjamini & Hochberg (1995) / Benjamini & Yekutieli (2001) FDR; Lakens et al. (2018) TOST;
Hartigan & Hartigan (1985) dip test; Van Calster et al. (2019) flexible calibration;
Esteban & Ray (1994) polarization; Hoad/Robinson/Davies (2010) MSER-5; Secchi & Seri (2017).
"""

import numpy as np


# ----------------------------------------------------------------------------- inequality / welfare
def _gini(x):
    """Classical Gini coefficient. VALID ONLY FOR NON-NEGATIVE inputs (can exceed 1 otherwise).

    For signed data use :func:`gini_negatives`.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n


def gini_negatives(x):
    """Renormalised Gini for data that may contain negative values (Raffinetti, Siletti &
    Vernizzi 2015), a direct port of ``GiniWegNeg::Gini_RSV`` (unit weights).

    The standard Gini numerator is normalised by ``mean(|x|)`` instead of ``mean(x)`` — the
    'polarised' scenario assigns the whole negative total to one unit and the whole positive
    total to another — keeping the index in [0, 1] for signed data. Reduces to the classical
    Gini when all values are non-negative.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return 0.0
    abs_total = np.abs(x).sum()
    if abs_total == 0.0:
        return 0.0
    idx = np.arange(1, n + 1)
    g_num = 2.0 * np.sum(idx * x) - (n + 1.0) * np.sum(x)   # = n^2 * classical Gini numerator
    return float(g_num / (n * abs_total))


def sen_welfare(x):
    """Sen (1976) social welfare, W = mean(x) * (1 - Gini(x)), evaluated on the ORIGINAL
    signed values via the Raffinetti-Siletti-Vernizzi renormalised Gini.

    ``x`` is per-agent normalised payoff (``agent_veracity`` in [-1, 1]). NOTE the prior
    implementation computed ``mean(x) * (1 - Gini(x - min(x)))``, mixing the original mean
    with a Gini on min-shifted values; the shift is not Gini-invariant and varies per
    simulation, distorting cross-scenario comparisons. This version removes the shift.

    Two documented limitations remain (report alongside the number): (1) mu*(1-G) is monotone
    increasing in EVERY value, so a gain accruing entirely at the top still raises W and it
    cannot register rising concentration as welfare-reducing (Stark 2025); (2) when mean(x) < 0
    the (1-G) factor pushes a negative-mean welfare toward zero as inequality rises.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(x.mean() * (1.0 - gini_negatives(x)))


# ----------------------------------------------------------------------------- spread / polarization
def belief_dispersion(x):
    """Belief DISPERSION as normalised variance, ``4 * Var(x)`` on [0, 1]-bounded beliefs
    (0 = consensus, 1 = half at 0 / half at 1).

    Renamed from ``polarization``: variance is the *dispersion* sense only (Bramson et al. 2017,
    "nine senses of polarization") and does NOT imply two groups/modes — use :func:`dip_test`
    for bimodality and :func:`esteban_ray` for group divergence.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(4.0 * np.var(x))


# Backward-compatible alias (deprecated): existing callers importing ``polarization`` keep working.
polarization = belief_dispersion


def esteban_ray(x, alpha=1.0, n_bins=10, K=1.0):
    """Esteban & Ray (1994) polarization index P = K * sum_i sum_j pi_i^(1+alpha) pi_j |y_i - y_j|,
    the group-divergence sense. Continuous ``x`` is histogram-binned (pi_i = mass in bin i,
    y_i = bin centre). ``alpha`` in (0, 1.6] (default 1.0). Provided as the sense-appropriate
    alternative to dispersion; not wired into the evidence scripts by default.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 1.0)
    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return 0.0
    pi = counts / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    diff = np.abs(centres[:, None] - centres[None, :])
    P = np.sum((pi ** (1.0 + alpha))[:, None] * pi[None, :] * diff)
    return float(K * P)


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient ``BC = (g1^2 + 1) / (g2 + 3*(n-1)^2/((n-2)(n-3)))``.

    WARNING: ``BC > 5/9`` (~0.555) is a documented FALSE-POSITIVE generator on skewed data
    (Pfister et al. 2013) — 0.555 is merely the value for a uniform distribution. Prefer
    :func:`dip_test`. Kept for backward comparison only.
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


def dip_test(x):
    """Hartigan & Hartigan (1985) dip test of unimodality (via the ``diptest`` package).

    Returns ``{'dip': statistic, 'p': p-value}``; ``p < 0.05`` rejects unimodality (i.e.
    indicates multimodality) and is the recommended replacement for ``BC > 5/9``.
    """
    import diptest as _dt
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return {"dip": float("nan"), "p": float("nan")}
    dip, pval = _dt.diptest(x)
    return {"dip": float(dip), "p": float(pval)}


# ----------------------------------------------------------------------------- effect sizes / CIs
def cohens_d_av(group1, group2):
    """Lakens (2013) d_av for paired/two-condition data: (mean(g1) - mean(g2)) divided by the
    AVERAGE of the two SDs. Unlike d_z it does NOT shrink with the inter-condition correlation,
    so it is comparable across designs and is the right number to read under common-random-number
    pairing (where d_z = d_av / sqrt(2(1-r)) is inflated for r > 0.5).
    """
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group2, dtype=float)
    s = (a.std(ddof=1) + b.std(ddof=1)) / 2.0
    if s == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / s)


def bootstrap_ci(data, statistic=np.mean, confidence=0.95, n_resamples=9999,
                 method="BCa", seed=0):
    """BCa bootstrap confidence interval (Kirby & Gerlanc 2013) via ``scipy.stats.bootstrap`` —
    makes no normality assumption, unlike a +/-1.96*SE interval. ``data`` is a 1-D sample (for a
    paired mean difference, pass the difference array with ``statistic=np.mean``).
    """
    from scipy import stats as sps
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 3:
        return (float("nan"), float("nan"))
    # BCa estimates its acceleration by jackknife, which scipy materialises as an (n, n-1) array
    # -> O(n^2) memory (~18.6 GiB at n=50000). For large n the BCa bias/acceleration corrections
    # are negligible and the percentile interval coincides with BCa (Efron & Tibshirani 1993;
    # Kirby & Gerlanc 2013), so fall back to the percentile method.
    if method == "BCa" and data.size > 5000:
        method = "percentile"
    kw = dict(n_resamples=n_resamples, confidence_level=confidence, method=method)
    try:
        res = sps.bootstrap((data,), statistic, rng=np.random.default_rng(seed), **kw)
    except TypeError:  # older scipy used random_state instead of rng
        res = sps.bootstrap((data,), statistic, random_state=seed, **kw)
    ci = res.confidence_interval
    return (float(ci.low), float(ci.high))


def bh_fdr(pvals, alpha=0.05, method="fdr_by"):
    """Multiple-comparison correction across the many outcome tests. Default ``fdr_by`` =
    Benjamini-Yekutieli (valid under arbitrary dependence, appropriate for correlated ABM
    outcomes); use ``fdr_bh`` for the independent/PRDS Benjamini-Hochberg case. Returns an array
    of adjusted q-values aligned to ``pvals`` (NaNs preserved).
    """
    from statsmodels.stats.multitest import multipletests
    pvals = np.asarray(pvals, dtype=float)
    q = np.full(pvals.shape, np.nan)
    mask = np.isfinite(pvals)
    if mask.sum() > 0:
        _, qv, _, _ = multipletests(pvals[mask], alpha=alpha, method=method)
        q[mask] = qv
    return q


def tost_paired(a, b, low, high, alpha=0.05):
    """Two One-Sided Tests of equivalence (Lakens et al. 2018) for paired data: tests whether the
    mean paired difference (a - b) lies within the equivalence band [low, high] (the smallest
    effect size of interest, in raw metric units). Returns the TOST p-value (max of the two
    one-sided p-values) and an ``equivalent`` flag. Use to formally support "no meaningful
    difference" claims at large N rather than reading a small d_z.
    """
    from scipy import stats as sps
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return {"p_tost": float("nan"), "equivalent": False, "mean_diff": float("nan")}
    md = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        eq = bool(low < md < high)
        return {"p_tost": 0.0 if eq else 1.0, "equivalent": eq, "mean_diff": float(md)}
    p_low = sps.t.sf((md - low) / se, n - 1)    # H0: diff <= low
    p_high = sps.t.cdf((md - high) / se, n - 1)  # H0: diff >= high
    p_tost = float(max(p_low, p_high))
    return {"p_tost": p_tost, "equivalent": bool(p_tost < alpha),
            "mean_diff": float(md), "bounds": [float(low), float(high)]}


# ----------------------------------------------------------------------------- calibration
def flexible_calibration(true, believed, frac=0.5, n_grid=25):
    """Flexible (loess) calibration curve plus the linear slope/intercept, with a non-linearity
    gain (Van Calster et al. 2019; Austin & Steyerberg 2014). A single linear slope can hide a
    curved believed-vs-true relationship; the loess curve does not. Fits on the raw points (no
    pre-binning). Returns slope, intercept, a downsampled loess curve, and ``nonlinearity_gain``
    = fraction of the linear residual variance additionally explained by the loess fit.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    true = np.asarray(true, dtype=float)
    believed = np.asarray(believed, dtype=float)
    m = np.isfinite(true) & np.isfinite(believed)
    true, believed = true[m], believed[m]
    if true.size < 3 or np.ptp(true) == 0:
        return {"slope": float("nan"), "intercept": float("nan"),
                "loess_x": [], "loess_y": [], "nonlinearity_gain": float("nan")}
    slope, intercept = np.polyfit(true, believed, 1)
    lin_pred = slope * true + intercept
    sm = lowess(believed, true, frac=frac, return_sorted=True)
    loess_pred = np.interp(true, sm[:, 0], sm[:, 1])
    ss_lin = float(np.sum((believed - lin_pred) ** 2))
    ss_loess = float(np.sum((believed - loess_pred) ** 2))
    gain = float(1.0 - ss_loess / ss_lin) if ss_lin > 0 else 0.0
    grid = np.linspace(true.min(), true.max(), n_grid)
    return {"slope": float(slope), "intercept": float(intercept),
            "loess_x": [round(float(v), 4) for v in grid],
            "loess_y": [round(float(v), 4) for v in np.interp(grid, sm[:, 0], sm[:, 1])],
            "nonlinearity_gain": round(gain, 4)}


# ----------------------------------------------------------------------------- ABM design helpers
def mser5(series, batch=5):
    """MSER-5 warm-up / truncation estimator (Hoad, Robinson & Davies 2010; White & Robinson
    2010): batches the output series into groups of ``batch`` and picks the truncation point that
    minimises the marginal standard error of the truncated batch mean (an approximation to the
    MSE of the steady-state mean). Search is restricted to the first half to avoid degenerate
    end-of-series solutions. Returns the estimated warm-up length in ORIGINAL series units.
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    m = y.size // batch
    if m < 2:
        return {"truncation": 0, "n_batches": int(m), "mser": float("nan")}
    b = y[:m * batch].reshape(m, batch).mean(axis=1)
    best_d, best_val = 0, np.inf
    for d in range(0, max(1, m // 2)):
        seg = b[d:]
        k = seg.size
        val = float(np.sum((seg - seg.mean()) ** 2) / (k ** 2))
        if val < best_val:
            best_val, best_d = val, d
    return {"truncation": int(best_d * batch), "n_batches": int(m), "mser": best_val}


def required_runs(J, es):
    """Secchi & Seri (2017) empirical approximation of the number of simulation runs needed,
    ``n ~= 14.091 * J^-0.640 * es^-1.986`` — CALIBRATED FOR alpha=0.01 and power=0.95 only.
    ``J`` = number of parameter configurations, ``es`` = Cohen's f effect size. Use to justify a
    replication count by a power target instead of a round number (pair with a smallest effect
    of interest, since ~50k sims over-powers trivial effects).
    """
    J = float(J)
    es = float(es)
    if J <= 0 or es <= 0:
        return float("nan")
    return float(14.091 * J ** (-0.640) * es ** (-1.986))
