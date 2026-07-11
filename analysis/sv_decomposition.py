"""Two-part decomposition of structural virality over the full per-cascade sweeps.

The per-simulation ``avg_structural_virality`` used by the sensitivity analysis and the
seeding comparison is a grand mean over all 1000 recorded cascades in which non-spreading
(size-1) cascades are scored zero, so it equals P(spread) x E[SV | spread]. Canonical
practice never pools those zeros: structural virality is undefined for single-node
cascades and must be reported size-conditioned (Goel et al. 2016, Manage. Sci.,
doi:10.1287/mnsc.2015.2158), and between-condition differences are interpretable only
after matching on cascade size (Juul & Ugander 2021, PNAS, doi:10.1073/pnas.2100786118).
This script therefore streams the full 491,520,000-cascade ``cascades.arrow`` of each
seeding configuration once (no sampling; exact bincount accumulators over memory-mapped
zero-copy slices) and decomposes the pooled metric into
  (i)  the ignition rate P(cascade size >= 2), and
  (ii) the conditional structural virality of spreading cascades, overall, per
       log10-size bin, and size-standardized over fixed pooled bin weights,
with the same paired-inference conventions as build_evidence.py (d_z, d_av, BCa/percentile
bootstrap CI, Wilcoxon, Benjamini-Yekutieli FDR). Cascade depth receives the identical
treatment (its pooled mean carries the same zero mass). Exact per-size histograms
(structural virality on a 0.001 grid, depth per hop) provide distribution-level
descriptives (quantiles, KS D) for the whole population; at n ~ 1e8 KS p-values are
meaningless, so inference rides on the per-simulation paired statistics.

Usage (the main analysis environment, see requirements.txt; stream the two seeding
configurations in parallel processes, then reduce):
  python analysis/sv_decomposition.py stream --sweep data/sweep_<ts>_baseline   --label baseline   --out analysis/runs/svdecomp
  python analysis/sv_decomposition.py stream --sweep data/sweep_<ts>_influencer --label influencer --out analysis/runs/svdecomp
  python analysis/sv_decomposition.py reduce --out analysis/runs/svdecomp \
      --baseline-csv data/baseline_svd/simulations.csv.gz \
      --influencer-csv data/influencer_svd/simulations.csv.gz \
      [--write-sobol-csv]
``stream --max-rows N`` exists only for the smoke gate; ``reduce`` refuses partial streams.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))
import paired_stats  # shared paired-inference conventions (_paired_core, _block_extras, g3)
from utils import bh_fdr
import data_io

N_SIMS = data_io.N_PAIRS
SIM_LO = 1                     # global_sim_id range verified [1, 491520] in both configurations
N_PER_SIM = data_io.N_CASC_PER_SIM   # recorded post-burn-in cascades per simulation
SIZE_MAX = data_io.N_NODES     # network size; cascade_size in [1, 300]
DEPTH_MAX = 300
CHUNK = 5_000_000
N_BINS = 8                     # log10 size bins, the Goel Fig. 5 convention used project-wide
SV_STEP = 0.001
SV_MAX = 35.0
SV_NBINS = int(SV_MAX / SV_STEP) + 1   # last bin collects any overflow >= SV_MAX
DEG_MAX = 1024                 # seed_degree / seed_node bincount width (degree < N)

ARM_LABELS = ("baseline", "influencer")


def size_bin_lut():
    """Bin lookup table over exact sizes 2..300: eight equal-width log10 bins spanning
    [2, 300], the Goel Fig. 5 convention. Assignments equal pd.cut(log10(sizes), bins=8)
    on the full size range, but the edges here are the true linspace values (pd.cut only
    stores its interval labels rounded to 3 decimals, which cannot reproduce the cut)."""
    sizes = np.arange(2, SIZE_MAX + 1)
    edges = np.linspace(np.log10(2), np.log10(SIZE_MAX), N_BINS + 1)
    codes = np.clip(np.searchsorted(edges, np.log10(sizes), side="right") - 1, 0, N_BINS - 1)
    lut = np.full(SIZE_MAX + 1, -1, np.int64)
    lut[sizes] = codes
    labels = [f"{10**lo:.0f}-{10**hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:])]
    assert (lut[2:] >= 0).all() and lut[2] == 0 and lut[SIZE_MAX] == N_BINS - 1
    return lut, labels, [float(e) for e in edges]


def cmd_stream(args):
    lut, _, _ = size_bin_lut()
    batch, n_rows = data_io.open_arrow(os.path.join(args.sweep, "cascades.arrow"))
    if args.max_rows is not None:
        n_rows = min(n_rows, args.max_rows)

    n_total = np.zeros(N_SIMS, np.int64)
    n_spread = np.zeros(N_SIMS, np.int64)
    sum_sv = np.zeros(N_SIMS, np.float64)          # SV is 0.0 at size 1, so = spread-only sum
    sum_depth_all = np.zeros(N_SIMS, np.float64)
    sum_depth_spread = np.zeros(N_SIMS, np.float64)
    bin_cnt = np.zeros(N_SIMS * N_BINS, np.int64)
    bin_sv_sum = np.zeros(N_SIMS * N_BINS, np.float64)
    size_hist = np.zeros(SIZE_MAX + 1, np.int64)
    size_sv_hist = np.zeros((SIZE_MAX + 1) * SV_NBINS, np.int64)
    size_depth_hist = np.zeros((SIZE_MAX + 1) * (DEPTH_MAX + 1), np.int64)
    seed_degree_hist = np.zeros(DEG_MAX, np.int64)
    seed_node_hist = np.zeros(DEG_MAX, np.int64)
    n_sv_zero = n_size1 = n_sv_overflow = 0
    sv_true_max = 0.0

    t0 = time.time()
    for off, sl in data_io.stream_chunks(batch, n_rows, CHUNK, label=args.label, print_every=10):
        sid = sl.column("global_sim_id").to_numpy(zero_copy_only=True)
        size = sl.column("cascade_size").to_numpy(zero_copy_only=True)
        sv = sl.column("structural_virality").to_numpy(zero_copy_only=True)
        depth = sl.column("max_depth").to_numpy(zero_copy_only=True)
        sdeg = sl.column("seed_degree").to_numpy(zero_copy_only=True)
        snode = sl.column("seed_node").to_numpy(zero_copy_only=True)
        assert sid.min() >= SIM_LO and sid.max() < SIM_LO + N_SIMS
        assert size.min() >= 1 and size.max() <= SIZE_MAX
        assert depth.max() <= DEPTH_MAX and sdeg.max() < DEG_MAX and snode.max() < DEG_MAX

        key = sid - SIM_LO
        spread = size >= 2
        n_total += np.bincount(key, minlength=N_SIMS)
        n_spread += np.bincount(key[spread], minlength=N_SIMS)
        sum_sv += np.bincount(key, weights=sv, minlength=N_SIMS)
        sum_depth_all += np.bincount(key, weights=depth, minlength=N_SIMS)
        sum_depth_spread += np.bincount(key[spread], weights=depth[spread], minlength=N_SIMS)
        kb = key[spread] * N_BINS + lut[size[spread]]
        bin_cnt += np.bincount(kb, minlength=N_SIMS * N_BINS)
        bin_sv_sum += np.bincount(kb, weights=sv[spread], minlength=N_SIMS * N_BINS)
        size_hist += np.bincount(size, minlength=SIZE_MAX + 1)
        svbin = np.minimum((sv / SV_STEP).astype(np.int64), SV_NBINS - 1)
        size_sv_hist += np.bincount(size * SV_NBINS + svbin,
                                    minlength=(SIZE_MAX + 1) * SV_NBINS)
        size_depth_hist += np.bincount(size * (DEPTH_MAX + 1) + depth,
                                       minlength=(SIZE_MAX + 1) * (DEPTH_MAX + 1))
        seed_degree_hist += np.bincount(sdeg, minlength=DEG_MAX)
        seed_node_hist += np.bincount(snode, minlength=DEG_MAX)
        n_sv_zero += int((sv == 0.0).sum())
        n_size1 += int((size == 1).sum())
        n_sv_overflow += int((sv >= SV_MAX).sum())
        sv_true_max = max(sv_true_max, float(sv.max()))

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.max_rows is None else "_smoke"
    out = os.path.join(args.out, f"acc_{args.label}{tag}.npz")
    np.savez_compressed(
        out, n_rows_streamed=n_rows, n_total=n_total, n_spread=n_spread, sum_sv=sum_sv,
        sum_depth_all=sum_depth_all, sum_depth_spread=sum_depth_spread,
        bin_cnt=bin_cnt.reshape(N_SIMS, N_BINS), bin_sv_sum=bin_sv_sum.reshape(N_SIMS, N_BINS),
        size_hist=size_hist, size_sv_hist=size_sv_hist.reshape(SIZE_MAX + 1, SV_NBINS),
        size_depth_hist=size_depth_hist.reshape(SIZE_MAX + 1, DEPTH_MAX + 1),
        seed_degree_hist=seed_degree_hist, seed_node_hist=seed_node_hist,
        n_sv_zero=n_sv_zero, n_size1=n_size1, n_sv_overflow=n_sv_overflow,
        sv_true_max=sv_true_max)
    print(f"WROTE {out} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ reduce helpers
def _load_acc(out_dir, label):
    path = os.path.join(out_dir, f"acc_{label}.npz")
    acc = dict(np.load(path).items())
    if int(acc["n_rows_streamed"]) != N_SIMS * N_PER_SIM:
        sys.exit(f"ERROR: {path} is a partial (smoke) stream; refusing to reduce")
    return acc


def _per_sim(acc):
    n_spread = acc["n_spread"].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "pooled_sv": acc["sum_sv"] / N_PER_SIM,
            "pooled_depth": acc["sum_depth_all"] / N_PER_SIM,
            "ignition": n_spread / N_PER_SIM,
            "cond_sv": np.where(n_spread > 0, acc["sum_sv"] / n_spread, np.nan),
            "cond_depth": np.where(n_spread > 0, acc["sum_depth_spread"] / n_spread, np.nan),
            "bin_mean_sv": np.where(acc["bin_cnt"] > 0,
                                    acc["bin_sv_sum"] / acc["bin_cnt"], np.nan),
        }


def _hist_quantiles(hist, qs, step=None):
    """Exact quantiles of a binned population; values are bin indices (times ``step``)."""
    total = hist.sum()
    if total == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    cum = np.cumsum(hist)
    out = {}
    for q in qs:
        idx = int(np.searchsorted(cum, q * total))
        out[f"p{int(q*100)}"] = float(idx * step) if step else float(idx)
    return out


def _ks_d(hist_a, hist_b):
    ta, tb = hist_a.sum(), hist_b.sum()
    if ta == 0 or tb == 0:
        return float("nan")
    return float(np.abs(np.cumsum(hist_a) / ta - np.cumsum(hist_b) / tb).max())


def _paired_row(a, b, blocks=None):
    """paired_cmp-formatted row that tolerates NaNs (sims lacking spread cascades)."""
    n_finite = int((np.isfinite(a) & np.isfinite(b)).sum())
    if n_finite < 3:  # e.g. a size bin no cascade ever lands in
        return {"n_sims_used": n_finite, **{k: float("nan") for k in (
            "mean_baseline", "mean_influencer", "mean_diff", "median_diff",
            "cohens_dz", "cohens_d_av", "p_ttest", "p_wilcoxon",
            "frac_influencer_gt_baseline", "rel_change_pct")},
            "ci95_diff_boot": [float("nan")] * 2, "ci95_diff_normal": [float("nan")] * 2}
    c = paired_stats._paired_core(a, b, mask_nonfinite=True, blocks=blocks)
    mean_a, mean_b = float(c["a"].mean()), float(c["b"].mean())
    rel = c["md"] / abs(mean_a) * 100 if mean_a != 0 else None
    row = {
        "n_sims_used": int(c["n"]),
        # CRN pairing correlation: d_z = d_av * s_bar/sd_diff is inflated only for r > 0.5
        # (Lakens 2013); conditional metrics dilute r, where d_z legitimately deflates.
        "pairing_r": round(float(np.corrcoef(c["a"], c["b"])[0, 1]), 4),
        "mean_baseline": round(mean_a, 5), "mean_influencer": round(mean_b, 5),
        "mean_diff": round(c["md"], 5), "median_diff": round(c["medd"], 5),
        "cohens_dz": round(c["dz"], 4), "cohens_d_av": round(c["dav"], 4),
        "ci95_diff_boot": [round(c["blo"], 5), round(c["bhi"], 5)],
        "ci95_diff_normal": [round(c["md"] - 1.96 * c["se"], 5),
                             round(c["md"] + 1.96 * c["se"], 5)],
        "p_ttest": paired_stats.g3(c["pt"]), "p_wilcoxon": paired_stats.g3(c["pw"]),
        "frac_influencer_gt_baseline": round(float((c["d"] > 0).mean()), 4),
        "rel_change_pct": (round(rel, 2) if rel is not None else None),
    }
    if blocks is not None:
        row.update(paired_stats._block_extras(c, blocks))
    return row


def cmd_reduce(args):
    lut, bin_labels, bin_edges = size_bin_lut()
    acc = {lab: _load_acc(args.out, lab) for lab in ARM_LABELS}
    per = {lab: _per_sim(acc[lab]) for lab in ARM_LABELS}
    B, I = acc["baseline"], acc["influencer"]
    pB, pI = per["baseline"], per["influencer"]

    # ---------------- population aggregates per configuration
    agg = {}
    for lab, a in acc.items():
        tot, spread = int(a["n_total"].sum()), int(a["n_spread"].sum())
        deg = a["seed_degree_hist"]
        agg[lab] = {
            "n_cascades": tot, "n_spread": spread,
            "ignition_rate": spread / tot,
            "grand_mean_sv": float(a["sum_sv"].sum()) / tot,
            "cond_mean_sv": float(a["sum_sv"].sum()) / spread,
            "grand_mean_depth": float(a["sum_depth_all"].sum()) / tot,
            "cond_mean_depth": float(a["sum_depth_spread"].sum()) / spread,
            "size1_share": int(a["n_size1"]) / tot,
            "n_sv_zero": int(a["n_sv_zero"]), "n_size1": int(a["n_size1"]),
            "sv_true_max": float(a["sv_true_max"]),
            "sv_overflow_share_of_spread": int(a["n_sv_overflow"]) / spread,
            "spread_share_sv_ge2": float(a["size_sv_hist"][:, int(2.0 / SV_STEP):].sum()) / spread,
            "mean_seed_degree": float((np.arange(DEG_MAX) * deg).sum() / deg.sum()),
            "distinct_seed_nodes": int((a["seed_node_hist"] > 0).sum()),
            "n_sims_no_spread": int((a["n_spread"] == 0).sum()),
        }

    # ---------------- exact two-part identity on population aggregates
    P_B, P_I = agg["baseline"]["ignition_rate"], agg["influencer"]["ignition_rate"]
    C_B, C_I = agg["baseline"]["cond_mean_sv"], agg["influencer"]["cond_mean_sv"]
    d_pooled = P_I * C_I - P_B * C_B
    ign_comp = (P_I - P_B) * (C_B + C_I) / 2
    shape_comp = (C_I - C_B) * (P_B + P_I) / 2
    identity = {
        "pooled_diff": d_pooled, "ignition_component": ign_comp,
        "shape_component": shape_comp,
        "ignition_share": ign_comp / d_pooled if d_pooled else float("nan"),
        "shape_share": shape_comp / d_pooled if d_pooled else float("nan"),
        "note": "exact: P_I*C_I - P_B*C_B = dP*(C_B+C_I)/2 + dC*(P_B+P_I)/2; "
                "P = ignition rate, C = E[SV | spread], population aggregates",
    }

    # ---------------- per-log-size-bin population stats + KS + quantiles
    bins = []
    sv2 = int(2.0 / SV_STEP)
    for b in range(N_BINS):
        sizes_in = np.where(lut == b)[0]
        row = {"bin": b, "label": bin_labels[b],
               "log10_edges": [bin_edges[b], bin_edges[b + 1]]}
        hists = {}
        for lab, a in acc.items():
            h_sv = a["size_sv_hist"][sizes_in].sum(axis=0)
            h_dp = a["size_depth_hist"][sizes_in].sum(axis=0)
            cnt = int(a["size_hist"][sizes_in].sum())
            cmean = (float(a["bin_sv_sum"][:, b].sum()) / a["bin_cnt"][:, b].sum()
                     if a["bin_cnt"][:, b].sum() else float("nan"))
            row[lab] = {
                "n": cnt, "share_of_spread": cnt / agg[lab]["n_spread"],
                "mean_sv": cmean,
                "sv_quantiles": _hist_quantiles(h_sv, (.05, .25, .5, .75, .95), SV_STEP),
                "share_sv_ge2": float(h_sv[sv2:].sum()) / cnt if cnt else float("nan"),
                "depth_quantiles": _hist_quantiles(h_dp, (.05, .25, .5, .75, .95), 1.0),
                "mean_depth": (float((np.arange(DEPTH_MAX + 1) * h_dp).sum() / cnt)
                               if cnt else float("nan")),
            }
            hists[lab] = h_sv
        row["ks_d_sv"] = _ks_d(hists["baseline"], hists["influencer"])
        bins.append(row)

    spread_hist = {lab: a["size_sv_hist"][2:].sum(axis=0) for lab, a in acc.items()}
    ks_overall = _ks_d(spread_hist["baseline"], spread_hist["influencer"])
    ks_size300 = _ks_d(B["size_sv_hist"][SIZE_MAX], I["size_sv_hist"][SIZE_MAX])

    # Finite-network saturation: mean conditional SV rises with size only while cascades are
    # small relative to the network (Goel 2016 pattern); near size = N the diffusion tree
    # spans the graph and its mean pairwise distance is bounded by the graph, so SV declines.
    saturation = {}
    grid = np.arange(SV_NBINS) * SV_STEP
    for lab, a in acc.items():
        tot = a["size_sv_hist"].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(tot >= 10_000, (a["size_sv_hist"] * grid).sum(axis=1)
                         / np.maximum(tot, 1), np.nan)
        peak = int(np.nanargmax(m))
        saturation[lab] = {"peak_size": peak, "mean_sv_at_peak": float(m[peak]),
                           "mean_sv_at_N": float(m[SIZE_MAX]),
                           "min_size_threshold": 10_000}

    # ---------------- paired per-sim inference (build_evidence conventions)
    # per-sim arrays are gsid-ordered; block = (gsid-1)//480 (16 design points x 30 reps),
    # layout asserted against a table carrying both id columns
    _ids = pd.read_csv(args.baseline_csv, usecols=["global_sim_id", "design_id"])
    assert len(_ids) == N_SIMS, len(_ids)
    assert np.array_equal(_ids["design_id"].values,
                          (_ids["global_sim_id"].values - 1) // data_io.REPS + 1)
    blocks = np.arange(N_SIMS) // data_io.PAIRS_PER_BLOCK
    paired = {}
    for name, key in (("pooled_sv", "pooled_sv"), ("pooled_depth", "pooled_depth"),
                      ("ignition_rate", "ignition"), ("cond_sv", "cond_sv"),
                      ("cond_depth", "cond_depth")):
        paired[name] = _paired_row(pB[key], pI[key], blocks=blocks)

    # size-standardized conditional SV: fixed pooled bin weights, renormalized per sim
    w = (B["bin_cnt"].sum(axis=0) + I["bin_cnt"].sum(axis=0)).astype(np.float64)
    w /= w.sum()
    std_vals = {}
    for lab, p in per.items():
        m = p["bin_mean_sv"]
        wt = np.where(np.isfinite(m), w, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            std_vals[lab] = np.nansum(m * wt, axis=1) / wt.sum(axis=1)
    paired["cond_sv_size_std"] = _paired_row(std_vals["baseline"], std_vals["influencer"], blocks=blocks)
    paired["cond_sv_size_std"]["note"] = ("per-sim bin means weighted by fixed pooled "
                                          "spread shares, renormalized over occupied bins")

    # robustness variant: equal weights over the populated bins, in raw SV units, as a check on
    # the pooled-occupancy weighting (terminal bin weight ~0.915). Computed on the per-bin paired
    # mean differences at fixed composition; a per-sim equal-weight average over occupied bins
    # would re-import the occupancy composition the fixed weights remove (occupied-bin sets
    # differ by configuration) and reverses sign for that artefactual reason.
    with np.errstate(invalid="ignore"):
        Db = pI["bin_mean_sv"] - pB["bin_mean_sv"]
    bin_mean_diff = np.nanmean(Db, axis=0)
    populated = np.isfinite(bin_mean_diff)
    eqw_diff = float(bin_mean_diff[populated].mean())
    G_eqw = int(blocks.max()) + 1
    rng_eqw = np.random.default_rng(0)
    idx_eqw = rng_eqw.integers(0, G_eqw, size=(9999, G_eqw))
    bm_cols = []
    for b in range(N_BINS):
        col, m = Db[:, b], np.isfinite(Db[:, b])
        if not m.any():
            bm_cols.append(np.full(9999, np.nan))
            continue
        s_g = np.bincount(blocks[m], weights=col[m], minlength=G_eqw)
        c_g = np.bincount(blocks[m], minlength=G_eqw).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            bm_cols.append(s_g[idx_eqw].sum(axis=1) / c_g[idx_eqw].sum(axis=1))
    bm = np.column_stack(bm_cols)[:, populated]
    vals = np.nanmean(bm, axis=1)
    vals = vals[np.isfinite(vals)]
    lo_eqw, hi_eqw = np.percentile(vals, [2.5, 97.5])
    paired["cond_sv_size_std_eqw"] = {
        "eqw_mean_diff_sv": round(eqw_diff, 5),
        "ci95_block_boot": [round(float(lo_eqw), 5), round(float(hi_eqw), 5)],
        "per_bin_mean_diff": [round(float(x), 5) if np.isfinite(x) else None for x in bin_mean_diff],
        "n_bins_populated": int(populated.sum()),
        "note": ("equal weights over the populated bins, raw SV units; robustness check for the "
                 "pooled-occupancy weighting of cond_sv_size_std. Computed on per-bin paired mean "
                 "differences at fixed composition; a per-sim equal-weight average over occupied "
                 "bins re-imports occupancy composition and is not a shape comparison"),
    }

    per_bin_paired, pvals = [], []
    for b in range(N_BINS):
        row = _paired_row(pB["bin_mean_sv"][:, b], pI["bin_mean_sv"][:, b], blocks=blocks)
        row["bin"], row["label"] = b, bin_labels[b]
        row["occupancy_share_pooled"] = float(w[b])
        per_bin_paired.append(row)
        pvals.append(row["p_ttest"])
    qvals = bh_fdr(np.array(pvals, float))
    for row, q in zip(per_bin_paired, qvals):
        row["q_fdr_by"] = paired_stats.g3(float(q))
    # the same six-populated-bin family on the CR1 block-clustered p's (NaN rows preserved)
    qvals_clu = bh_fdr(np.array([row.get("p_ttest_clustered", float("nan"))
                                 for row in per_bin_paired], float))
    for row, q in zip(per_bin_paired, qvals_clu):
        if "p_ttest_clustered" in row:
            row["q_fdr_by_clustered"] = paired_stats.g3(float(q))
            row["p_ttest_clustered"] = paired_stats.g3(row["p_ttest_clustered"])

    # ---------------- pass criteria V1-V7
    csv_means = {}
    for lab, p in (("baseline", args.baseline_csv), ("influencer", args.influencer_csv)):
        col = pd.read_csv(p, usecols=["avg_structural_virality"])["avg_structural_virality"]
        csv_means[lab] = float(col.mean())
    checks = {}
    # published-value comparisons only apply to the canonical full sweep; scaled runs keep
    # the scale-free internal-consistency parts of V1/V2
    checks["V1_reproduction"] = all(
        abs(agg[lab]["grand_mean_sv"] - csv_means[lab]) < 1e-6 for lab in ARM_LABELS
    ) and (not data_io.CANONICAL
           or (abs(agg["baseline"]["grand_mean_sv"] - 1.80253) < 1e-3
               and abs(agg["influencer"]["grand_mean_sv"] - 1.5689) < 1e-3))
    checks["V2_zero_pooling_identity"] = all(
        agg[lab]["n_sv_zero"] == agg[lab]["n_size1"] for lab in ARM_LABELS
    ) and (not data_io.CANONICAL
           or (abs(agg["baseline"]["size1_share"] - 0.6045) < 0.005
               and abs(agg["baseline"]["cond_mean_sv"] - 4.557) < 0.01))
    mdeg_b = agg["baseline"]["mean_seed_degree"]
    mdeg_i = agg["influencer"]["mean_seed_degree"]
    checks["V3_provenance"] = (mdeg_i >= 3 * mdeg_b and abs(mdeg_i - 27.4) <= 3
                               and 4.5 <= mdeg_b <= 7.5)
    # Goel Fig. 5 pattern holds for cascades small relative to the network; the terminal bin
    # contains the saturation regime (empirical peak ~ 0.6 N, decline toward N), so it is
    # exempt from the monotonicity requirement and reported via the saturation diagnostics.
    mono = []
    for lab in ARM_LABELS:
        m = [bins[b][lab]["mean_sv"] for b in range(N_BINS - 1)
             if bins[b][lab]["share_of_spread"] >= 0.001]
        mono.append(all(x < y for x, y in zip(m, m[1:])))
    checks["V4_goel_monotone_cond_sv"] = all(mono)
    # |d_av| <= |d_z| holds only where CRN pairing survives (r > 0.5, Lakens 2013);
    # per-sim conditional metrics dilute r below 0.5 and are exempt.
    rows = list(paired.values()) + per_bin_paired
    checks["V5_dav_le_dz"] = all(
        abs(r["cohens_d_av"]) <= abs(r["cohens_dz"]) + 1e-9
        for r in rows
        if np.isfinite(r.get("cohens_dz", float("nan"))) and r.get("pairing_r", 0.0) > 0.5)
    checks["V6_pair_alignment"] = all(
        (a["n_total"] == N_PER_SIM).all() for a in acc.values())
    checks["V7_hist_coverage"] = all(
        agg[lab]["sv_overflow_share_of_spread"] < 1e-4 for lab in ARM_LABELS
    ) and all(agg[lab]["n_cascades"] == N_SIMS * N_PER_SIM for lab in ARM_LABELS)
    print("\n== pass criteria ==")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # ---------------- outputs
    evidence = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {"n_sims": N_SIMS, "n_per_sim": N_PER_SIM, "chunk": CHUNK,
                   "sv_grid_step": SV_STEP, "sv_grid_max": SV_MAX, "n_bins": N_BINS,
                   "bin_labels": bin_labels, "bin_log10_edges": bin_edges,
                   "acc_files": {lab: f"acc_{lab}.npz" for lab in ARM_LABELS}},
        "aggregates": agg, "identity_decomposition": identity,
        "saturation": saturation,
        "paired": paired, "per_bin_paired": per_bin_paired, "bins": bins,
        "ks": {"overall_spread_sv_D": ks_overall, "size300_sv_D": ks_size300,
               "note": "descriptive only; n ~ 1e8 makes KS p-values meaningless"},
        "csv_means": csv_means, "checks": checks,
    }
    out_json = os.path.join(args.out, "sv_evidence.json")
    with open(out_json, "w") as f:
        json.dump(evidence, f, indent=1, default=float)
    print(f"WROTE {out_json}")

    per_sim = pd.DataFrame({"global_sim_id": np.arange(SIM_LO, SIM_LO + N_SIMS)})
    for lab, p in per.items():
        per_sim[f"ignition_{lab}"] = p["ignition"]
        per_sim[f"cond_sv_{lab}"] = p["cond_sv"]
        per_sim[f"cond_depth_{lab}"] = p["cond_depth"]
    out_csv = os.path.join(args.out, "per_sim_decomposition.csv")
    per_sim.to_csv(out_csv, index=False)
    print(f"WROTE {out_csv}")

    if args.write_sobol_csv:
        for lab, src in (("baseline", args.baseline_csv), ("influencer", args.influencer_csv)):
            sim = pd.read_csv(src)
            sim = sim.drop(columns=["ignition_rate", "cond_sv"], errors="ignore")  # idempotent
            add = per_sim[["global_sim_id", f"ignition_{lab}", f"cond_sv_{lab}"]].rename(
                columns={f"ignition_{lab}": "ignition_rate", f"cond_sv_{lab}": "cond_sv"})
            sim = sim.merge(add, on="global_sim_id", how="left")
            src_dir = os.path.dirname(src)
            # legacy _sv views got a sibling _svd dir; an already-_svd source is augmented in place
            dst_dir = src_dir if src_dir.endswith("_svd") else src_dir.replace("_sv", "_svd")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, "simulations.csv")
            sim.to_csv(dst, index=False)
            print(f"WROTE {dst} (NaN cond_sv rows: {int(sim['cond_sv'].isna().sum())})")

    sys.exit(0 if all(checks.values()) else 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stream", help="stream one arm's cascades.arrow into accumulators")
    st.add_argument("--sweep", required=True)
    st.add_argument("--label", required=True, choices=ARM_LABELS)
    st.add_argument("--out", required=True)
    st.add_argument("--max-rows", type=int, default=None,
                    help="smoke gate only; reduce refuses partial streams")
    st.set_defaults(fn=cmd_stream)
    rd = sub.add_parser("reduce", help="combine both arms into sv_evidence.json")
    rd.add_argument("--out", required=True)
    rd.add_argument("--baseline-csv", required=True)
    rd.add_argument("--influencer-csv", required=True)
    rd.add_argument("--write-sobol-csv", action="store_true")
    rd.set_defaults(fn=cmd_reduce)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
