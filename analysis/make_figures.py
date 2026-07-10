"""Render ALL analysis figures from the evidence JSONs / accumulators and the baseline sweep.

This is the single figure entry point: every figure in the analysis lives here, grouped by
analysis type. Each analysis SCRIPT is pure-compute (writes an evidence JSON / .npz); this script
renders from those outputs. The groups and their inputs:

  * descriptive / comparison  -- eda_evidence.json, compare_evidence.json + the baseline sweep CSV
  * structural virality       -- runs/svdecomp_*/sv_evidence.json (+ acc_*.npz)
  * equivalence / TOST        -- runs/tost_blocks_*/tost_blocks_evidence.json
  * model schematic           -- self-contained (rebuilds a representative graph; no evidence)
  * switchover robustness     -- runs/switchover_audit_*/switchover_audit_evidence.json

Each group applies its own Matplotlib style before rendering (the styles differ by group), so the
figures are byte-identical to those produced by the former per-type scripts.

Usage:
    python analysis/make_figures.py [run_dir]

With no argument the eda/compare/influencer JSONs are read from this directory (build_evidence.py
writes them here) and the other groups from their run dirs (data_io.*). Pass a run directory to
read its eda/compare/influencer JSONs and write figures into <run_dir>/figures/. A figure whose
evidence is absent is skipped rather than treated as an error.
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "lib"))
import data_io
from plotstyle import set_pub_style, BLUE, ORANGE, GREY, DARK, CRIMSON, UVABLUE, ARM_COLORS

ROOT = os.path.dirname(os.path.abspath(__file__))
# Evidence directory: pass a run directory as the first argument to read its eda/compare/influencer
# JSONs and write into <run_dir>/figures/; with no argument it defaults to this directory.
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else ROOT
if not os.path.isdir(DATA):
    sys.exit(f"evidence dir not found: {DATA}")
OUT = os.path.join(DATA, "figures")
os.makedirs(OUT, exist_ok=True)

BASELINE_SWEEP = os.path.dirname(data_io.BASELINE_SVD_CSV)   # per-simulation table for descriptives
# The TOST figures read the equivalence bound from ev["sesoi"] so the drawn band can never
# drift from the tested one (tost_blocks.py SESOI).


# ------------------------------------------------------------------ evidence loaders
def _load(fname, d=DATA):
    p = os.path.join(d, fname)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _load_sims():
    p = os.path.join(BASELINE_SWEEP, "simulations.csv")
    if not os.path.exists(p):
        p += ".gz"          # bundled sweeps ship gzipped; pd.read_csv decompresses by extension
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, usecols=["p_fake", "avg_fake_cascade", "avg_true_cascade",
                                   "veracity_differential", "avg_verify_rate"])


eda = _load("eda_evidence.json")
cmp = _load("compare_evidence.json")
inf = _load("influencer_node_evidence.json")
sims = _load_sims()
sv_ev = _load("sv_evidence.json", data_io.SVDECOMP_DIR)
sv_acc = ({a: np.load(os.path.join(data_io.SVDECOMP_DIR, f"acc_{a}.npz")) for a in ("baseline", "influencer")}
          if sv_ev is not None and all(os.path.exists(os.path.join(data_io.SVDECOMP_DIR, f"acc_{a}.npz"))
                                       for a in ("baseline", "influencer")) else None)
tost_ev = _load("tost_blocks_evidence.json", data_io.TOST_DIR)
switch_ev = _load("switchover_audit_evidence.json", data_io.SWITCHOVER_DIR)
mf_ev = _load("mean_field_evidence.json", data_io.MEANFIELD_DIR)
_mf_npz = os.path.join(data_io.MEANFIELD_DIR, "mf_points.npz")
mf_pts = np.load(_mf_npz) if (mf_ev is not None and os.path.exists(_mf_npz)) else None


def save(fig, name, dpi=150):
    p = os.path.join(OUT, name)
    fig.tight_layout(); fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print("WROTE", os.path.relpath(p, ROOT))


def binned(x, y, nbins=20):
    edges = np.linspace(x.min(), x.max(), nbins+1)
    idx = np.clip(np.digitize(x, edges)-1, 0, nbins-1)
    mx = np.array([x[idx==b].mean() if (idx==b).any() else np.nan for b in range(nbins)])
    my = np.array([y[idx==b].mean() if (idx==b).any() else np.nan for b in range(nbins)])
    return mx, my


# ================================================================== descriptive / comparison
# Style: the house serif style (set_pub_style() defaults). Applied once in main() before this group.

def fig_calibration():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.4, label="perfect calibration")
    # baseline (random-seed) bin points from eda
    pts = eda["calibration"]
    ax.scatter([d["p_true"] for d in pts], [d["belief"] for d in pts], s=45, color=BLUE,
               zorder=5, label="baseline (binned means)")
    for key, col, lab in [("baseline", BLUE, "baseline"), ("influencer", ORANGE, "influencer seed")]:
        cal = cmp["calibration"][key]; fx = cal["flexible"]
        xs = np.linspace(0.05, 0.95, 50)
        ax.plot(xs, cal["slope"]*xs + cal["intercept"], "-", color=col, lw=1.2, alpha=0.6,
                label=f"{lab} OLS (slope {cal['slope']:.2f})")
        ax.plot(fx["loess_x"], fx["loess_y"], "-", color=col, lw=2.6,
                label=f"{lab} loess")
    ax.set_xlabel(r"actual $P(\mathrm{true})=1-p_{\mathrm{fake}}$")
    ax.set_ylabel("mean believed $P(\\mathrm{true})$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=10, loc="lower right")
    save(fig, "calibration_patched.png")


def fig_welfare_radar():
    rm = eda["regime_means"]
    axes_lbl = ["tpr", "fpr", "v_cost", "loss", "p_fake", "rationality", "loss_aversion", "veracity_differential"]
    disp = ["TPR", "FPR", "v_cost", "loss", r"$p_{\mathrm{fake}}$", "rationality", "loss av.", "veracity diff."]
    M = np.array([[r[k] for k in axes_lbl] for r in rm], float)
    mn, mx = M.min(0), M.max(0); scaled = (M - mn) / np.where(mx-mn == 0, 1, mx-mn)
    ang = np.linspace(0, 2*np.pi, len(axes_lbl), endpoint=False).tolist(); ang += ang[:1]
    fig, axes = plt.subplots(1, 4, subplot_kw=dict(polar=True), figsize=(18, 5))
    cols = sns.color_palette("viridis", 4)
    for i, ax in enumerate(axes):
        vals = scaled[i].tolist() + scaled[i][:1].tolist()
        ax.plot(ang, vals, color=cols[i], lw=2); ax.fill(ang, vals, color=cols[i], alpha=0.25)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(disp, fontsize=10)
        ax.set_yticklabels([]); ax.set_ylim(0, 1)
    save(fig, "welfare_radar_patched.png")


def fig_pfake_drivers():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, col, c in [(ax[0], "avg_fake_cascade", CRIMSON),
                      (ax[1], "avg_true_cascade", UVABLUE)]:
        a.scatter(sims["p_fake"], sims[col], s=2, alpha=0.05, color=c, rasterized=True)
        mx, my = binned(sims["p_fake"].values, sims[col].values)
        a.plot(mx, my, color="black", lw=2)
        a.set_xlabel(r"fake-news rate $p_{\mathrm{fake}}$"); a.set_ylabel("reach (fraction of network)")
        a.set_ylim(0, 1)
    save(fig, "pfake_drivers.png")


def fig_cascade_sizes():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, col, c in [(ax[0], "avg_true_cascade", UVABLUE),
                      (ax[1], "avg_fake_cascade", CRIMSON)]:
        a.hist(sims[col], bins=40, color=c, alpha=0.85)
        a.set_xlabel("reach (fraction of network)"); a.set_ylabel("simulations")
    save(fig, "cascade_sizes.png")


def fig_verify_hump():
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.scatter(sims["p_fake"], sims["avg_verify_rate"], s=2, alpha=0.05, color="#666666", rasterized=True)
    mx, my = binned(sims["p_fake"].values, sims["avg_verify_rate"].values, nbins=24)
    ax.plot(mx, my, color=CRIMSON, lw=2.5, label="binned mean")
    ax.set_xlabel(r"fake-news rate $p_{\mathrm{fake}}$"); ax.set_ylabel("verification rate")
    ax.legend()
    save(fig, "verify_hump.png")


def fig_verify_vs_veracity():
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(sims["avg_verify_rate"], sims["veracity_differential"], s=2, alpha=0.05, color=UVABLUE, rasterized=True)
    ax.set_xlabel("verification rate"); ax.set_ylabel("veracity differential (true $-$ fake reach)")
    save(fig, "verify_vs_veracity.png")


# ================================================================== structural virality
# Style: sns.set_theme(context="paper", style="whitegrid"). Applied in main() before this group.
# From the full-population accumulators (every curve/box reflects all 491.5M cascades per configuration).
NEGLIGIBLE_DAV = 0.07          # the paper's own negligibility threshold


def fig_virality_by_size(ev):
    bins = ev["bins"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (arm, off) in enumerate((("baseline", -0.19), ("influencer", 0.19))):
        stats, positions = [], []
        for b in bins:
            s = b[arm]
            if s["n"] == 0 or not np.isfinite(s["mean_sv"]):
                continue
            q = s["sv_quantiles"]
            stats.append({"med": q["p50"], "q1": q["p25"], "q3": q["p75"],
                          "whislo": q["p5"], "whishi": q["p95"],
                          "mean": s["mean_sv"], "fliers": []})
            positions.append(b["bin"] + off)
        artists = ax.bxp(stats, positions=positions, widths=0.32, showfliers=False,
                         showmeans=True, meanline=True, patch_artist=True)
        for box in artists["boxes"]:
            box.set(facecolor=ARM_COLORS[arm], alpha=0.45, edgecolor=ARM_COLORS[arm])
        for part in ("medians", "whiskers", "caps", "means"):
            for line in artists[part]:
                line.set(color=ARM_COLORS[arm])
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b["label"] for b in bins], rotation=30, ha="right")
    ax.set_xlabel("cascade size (nodes)")
    ax.set_ylabel("structural virality (Wiener index)")
    ax.legend(handles=[Patch(facecolor=ARM_COLORS[a], alpha=0.45, label=a) for a in ARM_COLORS])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "virality_by_size.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("WROTE virality_by_size.png")


def _ccdf(hist):
    total = hist.sum()
    return 1.0 - np.cumsum(hist) / total


def fig_cascade_ccdfs(ev, acc):
    panels = [
        ("cascade size", {a: acc[a]["size_hist"] for a in acc},
         np.arange(301), True),
        ("max depth", {a: acc[a]["size_depth_hist"].sum(axis=0) for a in acc},
         np.arange(301), False),
        ("structural virality", {a: acc[a]["size_sv_hist"].sum(axis=0) for a in acc},
         np.arange(int(ev["config"]["sv_grid_max"] / ev["config"]["sv_grid_step"]) + 1)
         * ev["config"]["sv_grid_step"], True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (name, hists, x, logx) in zip(axes, panels):
        for arm, h in hists.items():
            ax.step(x, _ccdf(h), where="post", color=ARM_COLORS[arm], label=arm)
        support = max(int(np.nonzero(h)[0].max()) for h in hists.values())
        ax.set_xlim(right=x[min(support + 1, len(x) - 1)] * 1.05)
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel(r"CCDF  $P(X > x)$")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cascade_ccdfs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("WROTE cascade_ccdfs.png")


def fig_sv_decomposition(ev):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=[1, 1.6])

    ign = ev["paired"]["ignition_rate"]
    arms = ("baseline", "influencer")
    vals = [ign[f"mean_{a}"] for a in arms]
    ax1.bar(arms, vals, color=[ARM_COLORS[a] for a in arms], alpha=0.75, width=0.55)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_ylabel(r"ignition rate  $P(\mathrm{size} \geq 2)$")
    ax1.set_ylim(0, max(vals) * 1.18)

    # bins with too few paired simulations give unstable d_av (the 7-13 bin holds 20 simulations);
    # they remain in the JSON but are not drawn
    pb = [r for r in ev["per_bin_paired"]
          if np.isfinite(r["cohens_d_av"]) and r["n_sims_used"] >= 1000]
    xs = [r["bin"] for r in pb]
    ys = [r["cohens_d_av"] for r in pb]
    sig = [r["q_fdr_by"] < 0.05 for r in pb]
    ax2.axhspan(-NEGLIGIBLE_DAV, NEGLIGIBLE_DAV, color="0.85", alpha=0.6,
                label=f"negligible ($|d_{{av}}|<{NEGLIGIBLE_DAV}$)")
    ax2.axhline(0, color="0.4", lw=0.8)
    std = ev["paired"]["cond_sv_size_std"]["cohens_d_av"]
    ax2.axhline(std, color=CRIMSON, ls="--", lw=1.2,
                label=f"size-standardized $d_{{av}}$ = {std:+.3f}")
    ax2.scatter([x for x, s in zip(xs, sig) if s], [y for y, s in zip(ys, sig) if s],
                color="#1B1918", zorder=3, label="FDR-significant bin")
    ax2.scatter([x for x, s in zip(xs, sig) if not s], [y for y, s in zip(ys, sig) if not s],
                facecolors="none", edgecolors="#1B1918", zorder=3, label="n.s. bin")
    ax2.set_xticks([r["bin"] for r in ev["per_bin_paired"]])
    ax2.set_xticklabels([r["label"] for r in ev["per_bin_paired"]], rotation=30, ha="right")
    ax2.set_xlabel("cascade size (nodes)")
    ax2.set_ylabel(r"paired $d_{av}$ (influencer $-$ baseline)")
    ax2.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sv_decomposition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("WROTE sv_decomposition.png")


def g3_checks(ev, acc):
    """Render-consistency gate: medians vs paired signs on the three most populated bins;
    CCDF(size) left intercept equals the ignition rate."""
    fails = []
    pb = sorted((r for r in ev["per_bin_paired"] if np.isfinite(r["cohens_d_av"])),
                key=lambda r: -r["occupancy_share_pooled"])[:3]
    for r in pb:
        b = ev["bins"][r["bin"]]
        med_diff = b["influencer"]["sv_quantiles"]["p50"] - b["baseline"]["sv_quantiles"]["p50"]
        # sign agreement is expected, not guaranteed (median vs mean); report only
        print(f"G3 bin {r['label']}: median diff {med_diff:+.4f}, "
              f"paired mean d_av {r['cohens_d_av']:+.4f}")
    for arm in ("baseline", "influencer"):
        ccdf1 = 1.0 - acc[arm]["size_hist"][:2].sum() / acc[arm]["size_hist"].sum()
        ign = ev["aggregates"][arm]["ignition_rate"]
        ok = abs(ccdf1 - ign) < 1e-12
        print(f"G3 {arm}: CCDF(size>1) {ccdf1:.6f} == ignition {ign:.6f} "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(arm)
    print("G3", "FAIL" if fails else "PASS")
    return fails


# ================================================================== equivalence / TOST
# Style: set_pub_style(12, 14, 11, figsize=None). Applied in main() before this group.
# Rendered from tost_blocks_evidence.json (which already carries the superseded-CI overlay values).

def fig_fake_reach_equivalence(ev):
    ov = ev["overall"]
    md = ov["mean_diff"]
    ci95b = ov["ci95"]
    sesoi = float(ev["sesoi"])
    assert sesoi == 0.05, sesoi   # the tested bound; guards against a stale evidence file

    with sns.axes_style("ticks"):        # scoped: must not leak into fig_fake_reach_profile
        fig, ax = plt.subplots(figsize=(8, 2.6))
    band = sns.color_palette("deep")[2]   # muted green: the acceptance (equivalence) region
    ax.axvspan(-sesoi, sesoi, color=band, alpha=0.12, lw=0,
               label=f"equivalence region ($\\pm${sesoi:g})")
    for edge in (-sesoi, sesoi):
        ax.axvline(edge, color=GREY, ls="--", lw=1.0)
    ax.axvline(0, color=DARK, lw=1.0)
    ax.plot(ci95b, [0, 0], color=DARK, lw=2.4, solid_capstyle="round",
            label="95% CI (block-clustered)")
    ax.plot([md], [0], "|", color=CRIMSON, ms=16, mew=2.4, zorder=5, label="mean difference")
    ax.text(0.028, -0.34, f"mean {md:+.4f}\n95% CI [{ci95b[0]:+.4f}, {ci95b[1]:+.4f}]",
            ha="center", va="top", fontsize=9, color=DARK)
    ax.set_yticks([])
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlim(-0.06, 0.06)
    ax.set_xlabel("difference in fake-news reach (fraction of network)")
    ax.legend(fontsize=9, loc="upper left", ncol=2, frameon=False)
    sns.despine(ax=ax, left=True, trim=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fake_reach_equivalence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("WROTE fake_reach_equivalence.png")


def fig_fake_reach_profile(ev):
    pf, lam = ev["pfake_bins"], ev["lambda_bins"]
    sesoi = float(ev["sesoi"])
    assert sesoi == 0.05, sesoi   # the tested bound; guards against a stale evidence file
    fig = plt.figure(figsize=(10, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1], hspace=0.12, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=ax)
    axl = fig.add_subplot(gs[:, 1])

    x = [s["midpoint"] for s in pf]
    ax.axhline(0, color="black", lw=1.0)
    ax.axhline(sesoi, color=GREY, ls="--", lw=1.2)
    ax.axhline(-sesoi, color=GREY, ls="--", lw=1.2, label=f"equivalence bounds ($\\pm${sesoi:g})")
    for s in pf:
        ax.plot([s["midpoint"] + 0.012] * 2, s["old_designpoint"]["ci90_clustered"],
                color=GREY, lw=1.4, solid_capstyle="round")
        ax.plot([s["midpoint"]] * 2, s["ci90"], color="black", lw=3.0,
                solid_capstyle="round")
    ax.plot([], [], color=GREY, lw=1.4, label="design-point 90% CI (superseded)")
    ax.plot(x, [s["mean_diff"] for s in pf], "s", color="black", ms=6,
            label="mean difference (block-clustered 90% CI)")
    ax.set_ylim(-0.06, 0.06)
    ax.set_ylabel("difference in\nfake-news reach")
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=9, loc="upper right", frameon=True, framealpha=0.95, edgecolor=GREY)

    axr.plot(x, [s["base_reach_mean"] for s in pf], color=GREY, ls=":", lw=2.0, marker="o", ms=4)
    axr.set_ylabel("baseline\nreach", fontsize=11)
    axr.set_ylim(0, 0.75)
    axr.set_yticks([0, 0.3, 0.6])
    axr.set_xlabel("fake-news prevalence $p_{\\mathrm{fake}}$ (bin midpoint)")

    xl = [s["midpoint"] for s in lam]
    axl.axhline(0, color="black", lw=1.0)
    axl.axhline(sesoi, color=GREY, ls="--", lw=1.2)
    axl.axhline(-sesoi, color=GREY, ls="--", lw=1.2)
    for s in lam:
        axl.plot([s["midpoint"] + 0.035] * 2, s["old_designpoint"]["ci90_clustered"],
                 color=GREY, lw=1.4, solid_capstyle="round")
        axl.plot([s["midpoint"]] * 2, s["ci90"], color="black", lw=3.0,
                 solid_capstyle="round")
    axl.plot(xl, [s["mean_diff"] for s in lam], "s", color="black", ms=6)
    axl.set_ylim(-0.06, 0.06)
    axl.set_xlabel("rationality $\\log_{10}\\lambda$")

    fig.savefig(os.path.join(OUT, "fake_reach_profile.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("WROTE fake_reach_profile.png")


# ================================================================== model schematic
# Self-contained, deterministic. Applies its own style inside fig_cascade_schematic().
_SCHEM_C = dict(
    fake="#BC0031", true="#1B6FBC", hub="#E69F00", base="#7f7f7f",
    share="#009E73", verify="#56B4E9", discard="#BBBBBB", belief="#CC79A7", ink="#222222",
)
_SCHEM_FS = dict(header=12.5, seed=12, legend=12, note=11.5)
_SCHEM_SEED = 42
_TREE = {
    "S":  dict(d=0, p=None, a="seed"),
    "a1": dict(d=1, p="S",  a="share"),
    "a2": dict(d=1, p="S",  a="verify_clean"),
    "a3": dict(d=1, p="S",  a="discard"),
    "a4": dict(d=1, p="S",  a="verify_flag"),
    "a5": dict(d=1, p="S",  a="share"),
    "b1": dict(d=2, p="a1", a="share"),
    "b2": dict(d=2, p="a1", a="discard"),
    "b3": dict(d=2, p="a2", a="verify_clean"),
    "b4": dict(d=2, p="a2", a="share"),
    "b5": dict(d=2, p="a5", a="share"),
    "b6": dict(d=2, p="a5", a="discard"),
    "c1": dict(d=3, p="b1", a="share"),
    "c2": dict(d=3, p="b3", a="share"),
    "c3": dict(d=3, p="b4", a="discard"),
    "c4": dict(d=3, p="b5", a="share"),
}
_STYLE = {
    "seed":         (_SCHEM_C["share"], "o"),
    "share":        (_SCHEM_C["share"], "o"),
    "verify_clean": (_SCHEM_C["verify"], "s"),
    "verify_flag":  (_SCHEM_C["verify"], "s"),
    "discard":      (_SCHEM_C["discard"], "v"),
}


def _tree_layout_h(tree):
    """Left-to-right tree: x = BFS depth, y = leaf slot (internal node = mean of children)."""
    kids = {n: [] for n in tree}
    root = None
    for n, dat in tree.items():
        (kids[dat["p"]].append(n) if dat["p"] is not None else None)
        if dat["p"] is None:
            root = n
    for n in kids:
        kids[n].sort()
    slot = {}
    ctr = [0]

    def leaves(n):
        if not kids[n]:
            slot[n] = ctr[0]
            ctr[0] += 1
        else:
            for c in kids[n]:
                leaves(c)
    leaves(root)
    pos = {}

    def setp(n):
        y = slot[n] if not kids[n] else float(np.mean([setp(c) for c in kids[n]]))
        pos[n] = (float(tree[n]["d"]), y)
        return y
    setp(root)
    return pos, kids, root, ctr[0]


def _schem_network():
    G = nx.powerlaw_cluster_graph(300, 3, 0.5, seed=_SCHEM_SEED)   # Holme-Kim (matches the Julia generator)
    deg = dict(G.degree())
    hubs = sorted(G.nodes(), key=lambda v: deg[v], reverse=True)[:15]   # top 5% of 300
    return deg, hubs


def _draw_cascade(ax, kseed):
    C, FS = _SCHEM_C, _SCHEM_FS
    ax.set_axis_off()
    pos, kids, root, nleaf = _tree_layout_h(_TREE)     # x = BFS depth, y = leaf slot
    ax.set_xlim(-1.75, 4.15)
    ax.set_ylim(-0.85, 8.45)

    # (0) light column guides + depth headers
    for d, lab in [(0, "depth 0 (seed)"), (1, "depth 1"), (2, "depth 2"), (3, "depth 3")]:
        ax.plot([d, d], [-0.35, 7.65], color="#E3E3E3", lw=1.2, zorder=0)
        ax.text(d, 7.95, lab, ha="center", va="bottom", fontsize=FS["header"], style="italic",
                color="#555555")

    # (2) transmission edges (parent shared / clean-verified -> child reached)
    for n, dat in _TREE.items():
        if dat["p"] is not None:
            sA = 22 if dat["p"] == "S" else 17
            ax.add_patch(FancyArrowPatch(pos[dat["p"]], pos[n], arrowstyle="-|>", mutation_scale=18,
                                         lw=2.6, color=C["fake"], shrinkA=sA, shrinkB=17, zorder=3))

    # (3) nodes: all drawn as circles, grouped by fill colour (marker overridden locally, not in ms)
    groups = {}
    for n, dat in _TREE.items():
        groups.setdefault(_STYLE[dat["a"]][0], []).append(n)
    for fc, ns in groups.items():
        ec = "#888888" if fc == C["discard"] else "white"   # dark rim so the light grey circle remains visible
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns], s=700, marker="o",
                   facecolors=fc, edgecolors=ec, linewidths=1.8, zorder=4)

    # (4) seed ring + two-line label; flagged-VERIFY crimson X
    sx, sy = pos["S"]
    ax.scatter([sx], [sy], s=1260, marker="o", facecolors="none", edgecolors=C["hub"],
               linewidths=3.2, zorder=5)
    ax.text(sx - 0.30, sy, f"hub seed, $k={kseed}$\n(only 5 of {kseed} edges drawn)", ha="right",
            va="center", fontsize=FS["seed"], color=C["ink"], zorder=7,
            linespacing=1.4)
    fx, fy = pos["a4"]
    ax.scatter([fx], [fy], s=300, marker="x", color=C["fake"], linewidths=3.2, zorder=6)

    # (5) legend in the empty lower-left region
    handles = [
        Line2D([], [], marker="o", ls="none", ms=13, mfc=C["share"], mec="white", label="SHARE — spreads"),
        Line2D([], [], marker="o", ls="none", ms=13, mfc=C["verify"], mec="white",
               label="VERIFY, clean — still shares"),
        Line2D([], [], marker="x", ls="none", ms=12, mec=C["fake"], mfc=C["fake"],
               label="VERIFY, flagged — discards"),
        Line2D([], [], marker="o", ls="none", ms=13, mfc=C["discard"], mec="#888888", label="DISCARD — stops"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.006, 0.02), frameon=True,
              framealpha=0.94, fontsize=FS["legend"], borderpad=0.7, labelspacing=0.55,
              handletextpad=0.6).set_zorder(10)


def fig_cascade_schematic(stem="cascade_schematic"):
    # Inlined house style for this figure (formerly module-level in make_cascade_schematic.py).
    sns.set_context("notebook")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False,
        "font.size": 11, "axes.axisbelow": True, "grid.alpha": 0.3,
        "pdf.fonttype": 42,   # embed TrueType so PDF text stays selectable/editable
    })
    deg, hubs = _schem_network()          # deterministic; identical seed degree to the model panels
    kseed = deg[hubs[2]]

    fig = plt.figure(figsize=(10.5, 7.8))
    ax = fig.add_axes([0.028, 0.085, 0.958, 0.795])
    _draw_cascade(ax, kseed)

    pdf = os.path.join(OUT, f"{stem}.pdf")
    png = os.path.join(OUT, f"{stem}.png")
    fig.savefig(pdf)                 # NO bbox_inches="tight" (fig.text + add_axes layout is final)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("WROTE", os.path.relpath(png, ROOT))


# ================================================================== switchover robustness
# Self-contained (own style); rendered from switchover_audit_evidence.json.

def fig_switchover_profile(ev):
    set_pub_style(12, 14, 11, figsize=None)   # figsize=None: this figure never set figure.figsize
    grey = GREY

    pf = ev["pfake_bins"]
    x = [(float(s["bin"].split(",")[0][1:]) + float(s["bin"].split(",")[1][:-1])) / 2
         for s in pf]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    (axP, axC), (axdP, axdC) = axes

    axP.plot(x, [s["baseline"]["P_ign_fake"] for s in pf], color=grey, ls=":", lw=2.0,
             marker="o", ms=5, label="baseline (uniform seeding)")
    axP.plot(x, [s["influencer"]["P_ign_fake"] for s in pf], color="black", ls="-", lw=2.0,
             marker="s", ms=5, label="influencer (hub seeding)")
    axP.set_ylabel("$P(\\mathrm{ignite}\\mid\\mathrm{fake})$")
    axP.legend(fontsize=9, loc="upper right")

    axC.plot(x, [s["baseline"]["C_tilde_fake"] for s in pf], color=grey, ls=":", lw=2.0,
             marker="o", ms=5)
    axC.plot(x, [s["influencer"]["C_tilde_fake"] for s in pf], color="black", ls="-", lw=2.0,
             marker="s", ms=5)
    axC.set_ylabel("$\\tilde{C}$ (reach $\\mid$ ignited)")

    axdP.axhline(0, color="black", lw=1.0)
    for s, xi in zip(pf, x):
        axdP.plot([xi] * 2, s["delta_ci"]["P_ign_fake"]["ci90"], color="black", lw=3.0,
                  solid_capstyle="round")
    axdP.plot(x, [s["decomposition"]["delta_P"] for s in pf], "s", color="black", ms=5)
    axdP.set_ylabel("$\\Delta P$ (block CI90)")
    axdP.set_xlabel("fake-news prevalence $p_{\\mathrm{fake}}$ (bin midpoint)")

    axdC.axhline(0, color="black", lw=1.0)
    for s, xi in zip(pf, x):
        axdC.plot([xi] * 2, s["delta_ci"]["C_completepairs"]["ci90"], color="black", lw=3.0,
                  solid_capstyle="round")
    axdC.plot(x, [s["delta_ci"]["C_completepairs"]["mean_diff"] for s in pf], "s",
              color="black", ms=5)
    axdC.set_ylabel("$\\Delta C$ (complete pairs, block CI90)")
    axdC.set_xlabel("fake-news prevalence $p_{\\mathrm{fake}}$ (bin midpoint)")

    lb = ev["direction_relation"]["low_bin"]
    axdC.text(0.97, 0.05,
              f"low bin {lb['bin']}:\nignition share {lb['ignition_share']:+.0%}, "
              f"conditional share {lb['conditional_share']:+.0%}",
              transform=axdC.transAxes, ha="right", va="bottom", fontsize=9,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor=grey))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "switchover_profile.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("WROTE switchover_profile.png")


# ================================================================== mean-field benchmark
# Style: set_pub_style(12, 14, 11, figsize=None) after an rcdefaults reset. Applied in
# main() before this group. Rendered from mean_field_evidence.json + mf_points.npz
# (analysis/mean_field.py; one point per Saltelli design point, means over 30 reps).

def fig_mean_field(ev, pts):
    fig, (axR, axS) = plt.subplots(1, 2, figsize=(10, 3.9))

    # (a) observed fake reach against the mean-field branching ratio; the percolation
    # transition sits at R = 1 (Newman 2002)
    R = np.clip(pts["R_base_fake"], 1e-2, None)
    axR.scatter(R, pts["obs_base_fake"], s=2, alpha=0.05, color=GREY, rasterized=True)
    lo = np.log10(R)
    mx, my = binned(lo, pts["obs_base_fake"], nbins=28)
    axR.plot(10 ** mx, my, color=CRIMSON, lw=2.5, label="binned mean")
    axR.axvline(1.0, color=DARK, lw=1.4, ls="--")
    axR.text(1.0, 1.02, "$R=1$", ha="center", va="bottom", fontsize=10,
             transform=axR.get_xaxis_transform())
    axR.set_xscale("log")
    axR.set_xlabel("mean-field branching ratio $R$ (fake messages)")
    axR.set_ylabel("simulated fake-news reach")
    axR.set_ylim(0, 1)
    axR.legend(fontsize=9, loc="upper left")

    # (b) predicted against simulated reach, both veracities, both configurations
    axS.plot([0, 1], [0, 1], "--", color=DARK, lw=1.4)
    for xk, yk, col, mark, lab in (
            ("mf_base_true", "obs_base_true", UVABLUE, "o", "true, baseline"),
            ("mf_infl_true", "obs_infl_true", UVABLUE, "^", "true, influencer"),
            ("mf_base_fake", "obs_base_fake", CRIMSON, "o", "fake, baseline"),
            ("mf_infl_fake", "obs_infl_fake", CRIMSON, "^", "fake, influencer")):
        axS.scatter(pts[xk], pts[yk], s=2, alpha=0.04, color=col, marker=mark,
                    rasterized=True)
    cb = ev["comparison_baseline"]
    axS.text(0.03, 0.97,
             f"fake: $r={cb['fake_reach']['pearson_r']:.2f}$\n"
             f"true: $r={cb['true_reach']['pearson_r']:.2f}$",
             transform=axS.transAxes, ha="left", va="top", fontsize=10)
    axS.set_xlabel("mean-field predicted reach")
    axS.set_ylabel("simulated reach")
    axS.set_xlim(0, 1)
    axS.set_ylim(0, 1)
    axS.legend(handles=[Line2D([], [], color=UVABLUE, marker="o", ls="", label="true news"),
                        Line2D([], [], color=CRIMSON, marker="o", ls="", label="fake news")],
               fontsize=9, loc="lower right")
    save(fig, "mean_field.png", dpi=200)


# ------------------------------------------------------------------ driver
def _render(label, fn, *args):
    try:
        fn(*args)
    except Exception as e:   # one figure's missing key must not abort the rest
        print(f"SKIP {label}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # --- descriptive / comparison (house serif style) ---
    set_pub_style()
    if eda and cmp:
        _render("calibration", fig_calibration)
    if eda:
        _render("welfare_radar", fig_welfare_radar)
    if sims is not None:
        for fn in (fig_cascade_sizes, fig_verify_hump, fig_verify_vs_veracity, fig_pfake_drivers):
            _render(fn.__name__, fn)
    else:
        print("SKIP descriptive sweep figures: baseline simulations.csv absent")

    # --- structural virality (seaborn paper style) ---
    if sv_ev is not None:
        matplotlib.rcdefaults()   # plot_sv_figures rendered from a fresh process; reset the
        sns.set_theme(context="paper", style="whitegrid")   # style state before its seaborn theme
        _render("virality_by_size", fig_virality_by_size, sv_ev)
        _render("sv_decomposition", fig_sv_decomposition, sv_ev)
        if sv_acc is not None:
            _render("cascade_ccdfs", fig_cascade_ccdfs, sv_ev, sv_acc)
            g3_checks(sv_ev, sv_acc)
    else:
        print("SKIP structural-virality figures: sv_evidence.json absent")

    # --- equivalence / TOST ---
    if tost_ev is not None:
        set_pub_style(12, 14, 11, figsize=None)
        _render("fake_reach_equivalence", fig_fake_reach_equivalence, tost_ev)
        _render("fake_reach_profile", fig_fake_reach_profile, tost_ev)
    else:
        print("SKIP equivalence figures: tost_blocks_evidence.json absent")

    # --- model schematic (own style) ---
    _render("cascade_schematic", fig_cascade_schematic)

    # --- switchover robustness (own style) ---
    if switch_ev is not None:
        _render("switchover_profile", fig_switchover_profile, switch_ev)
    else:
        print("SKIP switchover figure: switchover_audit_evidence.json absent")

    # --- mean-field benchmark ---
    if mf_ev is not None and mf_pts is not None:
        matplotlib.rcdefaults()
        set_pub_style(12, 14, 11, figsize=None)
        _render("mean_field", fig_mean_field, mf_ev, mf_pts)
    else:
        print("SKIP mean-field figure: mean_field_evidence.json / mf_points.npz absent")

    print(f"DONE; figures in {os.path.relpath(OUT, ROOT)}")
