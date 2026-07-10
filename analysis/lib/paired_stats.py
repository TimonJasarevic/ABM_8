"""Paired seeding-contrast inference conventions shared by build_evidence.py and
sv_decomposition.py: the paired-statistics core (d_z, d_av, BCa/percentile bootstrap CI,
Wilcoxon, CR1 block-clustered p/CI), the block-aware row extras, the BY-FDR family pass,
and the two evidence-row formatters.

Pipeline glue, not citable generic statistics (those live in :mod:`utils`) and not IO/data
facts (:mod:`data_io`). Function bodies are verbatim moves from build_evidence.py so that
every evidence JSON reproduces byte-identically.
"""
import numpy as np
from scipy import stats

import utils
from utils import cohens_d_av, bootstrap_ci, bh_fdr


def g3(x):
    return float(f"{x:.3g}")


def _paired_core(a, b, mask_nonfinite=False, blocks=None):
    """Common paired statistics for diff = b - a: means, d_z, correlation-free d_av (Cumming 2012;
    Lakens 2013), bootstrap CI on the mean difference (BCa below n=5000, percentile above; Kirby &
    Gerlanc 2013), normal CI, paired t and Wilcoxon p-values, and the pairing correlation r.
    With ``blocks`` (the Saltelli base block of each pair) it adds block-bootstrap CIs for d_av
    and the mean difference plus a CR1 block-clustered two-sided p and CI (utils.tost_clustered).
    d_z exceeds d_av only where r > 0.5; read d_av."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if mask_nonfinite:
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if blocks is not None:
            blocks = np.asarray(blocks)[m]
    d = b - a; n = len(d)
    md, sd, medd = float(d.mean()), float(d.std(ddof=1)), float(np.median(d))
    dz = md / sd if sd > 0 else float("nan")
    se = sd / np.sqrt(n)
    dav = cohens_d_av(b, a)
    r = float(np.corrcoef(a, b)[0, 1]) if n > 1 and a.std() > 0 and b.std() > 0 else float("nan")
    blo, bhi = bootstrap_ci(d, batch=100)   # chunk the ~491k-row resample; bit-identical CI, bounds peak RAM
    t, pt = stats.ttest_rel(b, a)
    try:
        w, pw = stats.wilcoxon(b, a)
    except Exception:
        pw = float("nan")
    core = {"a": a, "b": b, "d": d, "n": n, "md": md, "medd": medd,
            "dz": dz, "se": se, "dav": dav, "r": r, "blo": blo, "bhi": bhi, "pt": pt, "pw": pw}
    if blocks is not None:
        core["dav_ci_block"] = utils.block_bootstrap_ci(a, b, blocks, stat="d_av")
        core["md_ci_block"] = utils.block_bootstrap_ci(a, b, blocks, stat="mean_diff")
        tc = utils.tost_clustered(d, np.asarray(blocks), expect_sizes=None)
        core["p_clu"] = float(tc["p_zero_two_sided"])
        core["ci95_clu"] = [float(tc["ci95"][0]), float(tc["ci95"][1])]
    return core


def _block_extras(c, blocks):
    """Additive per-row keys for the block-aware paired statistics (appended after the legacy
    keys so existing consumers and diffs are unaffected). ``p_ttest_clustered`` stays full
    precision here; cmd_* g3-round it only AFTER the FDR pass has consumed it."""
    row = {"pairing_r": round(c["r"], 4)}
    if blocks is not None:
        row["cohens_d_av_ci95_block"] = [round(c["dav_ci_block"][0], 4), round(c["dav_ci_block"][1], 4)]
        row["ci95_diff_block_boot"] = [round(c["md_ci_block"][0], 5), round(c["md_ci_block"][1], 5)]
        row["ci95_diff_clustered"] = [round(c["ci95_clu"][0], 5), round(c["ci95_clu"][1], 5)]
        row["p_ttest_clustered"] = c["p_clu"]
    return row


def _clustered_family(rows):
    """BY-FDR over the block-clustered p's of one family of rows (additive ``q_fdr_by_clustered``
    key). Consumes the full-precision ``p_ttest_clustered`` values, then g3-rounds them."""
    keys = [k for k, v in rows.items() if isinstance(v, dict) and "p_ttest_clustered" in v]
    if keys:
        q = bh_fdr([rows[k]["p_ttest_clustered"] for k in keys], method="fdr_by")
        for k, qi in zip(keys, q):
            rows[k]["q_fdr_by_clustered"] = g3(float(qi))
            rows[k]["p_ttest_clustered"] = g3(rows[k]["p_ttest_clustered"])


def paired_cmp(a, b, name, blocks=None):
    """Baseline (a) vs influencer (b) paired-comparison row for compare_evidence.json."""
    c = _paired_core(a, b, blocks=blocks)
    mean_a, mean_b = float(c["a"].mean()), float(c["b"].mean())
    rel = c["md"] / abs(mean_a) * 100 if mean_a != 0 else None
    row = {
        "mean_baseline": round(mean_a, 5), "mean_influencer": round(mean_b, 5),
        "mean_diff": round(c["md"], 5), "median_diff": round(c["medd"], 5),
        "cohens_dz": round(c["dz"], 4), "cohens_d_av": round(c["dav"], 4),
        "ci95_diff_boot": [round(c["blo"], 5), round(c["bhi"], 5)],
        "ci95_diff_normal": [round(c["md"] - 1.96*c["se"], 5), round(c["md"] + 1.96*c["se"], 5)],
        "p_ttest": g3(c["pt"]), "p_wilcoxon": g3(c["pw"]),
        "frac_influencer_gt_baseline": round(float((c["d"] > 0).mean()), 4),
        "rel_change_pct": (round(rel, 2) if rel is not None else None),
    }
    row.update(_block_extras(c, blocks))
    return {name: row}


def paired_inf(inf, non, name, blocks=None):
    """Influencer-group (inf) vs non-influencer-group (non) paired-comparison row for
    influencer_node_evidence.json; drops sims where either group value is non-finite."""
    c = _paired_core(non, inf, mask_nonfinite=True, blocks=blocks)
    row = {
        "n_sims_used": int(c["n"]),
        "mean_influencer": round(float(c["b"].mean()), 5),
        "mean_noninfluencer": round(float(c["a"].mean()), 5),
        "mean_diff": round(c["md"], 5), "median_diff": round(c["medd"], 5),
        "cohens_dz": round(c["dz"], 4), "cohens_d_av": round(c["dav"], 4),
        "ci95_diff_boot": [round(c["blo"], 5), round(c["bhi"], 5)],
        "ci95_diff_normal": [round(c["md"] - 1.96*c["se"], 5), round(c["md"] + 1.96*c["se"], 5)],
        "p_ttest": g3(c["pt"]), "p_wilcoxon": g3(c["pw"]),
        "frac_sims_influencer_higher": round(float((c["d"] > 0).mean()), 4)}
    row.update(_block_extras(c, blocks))
    return {name: row}
