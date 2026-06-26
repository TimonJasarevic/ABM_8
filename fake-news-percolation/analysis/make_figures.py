"""Regenerate figures from the evidence JSONs (eda_evidence.json, compare_evidence.json,
influencer_node_evidence.json) into figures/.

Covers (i) descriptive figures (calibration + loess, sen_welfare under RSV Gini, dispersion +
dip, welfare radar) and (ii) influencer-dynamics figures (seeding d_z-vs-d_av forest, influencer
vs ordinary, fake-reach TOST equivalence, pseudoreplication false precision).

Usage:  python make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# match EDA.ipynb style
sns.set_context("notebook", font_scale=1.2)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 15, "figure.dpi": 100, "grid.alpha": 0.3, "axes.axisbelow": True,
    "figure.figsize": (8, 6), "mathtext.fontset": "cm",
    "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 12,
})
plt.rc("text", usetex=False); plt.rc("font", family="serif")
BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"

eda = json.load(open(os.path.join(ROOT, "eda_evidence.json")))
cmp = json.load(open(os.path.join(ROOT, "compare_evidence.json")))
inf = json.load(open(os.path.join(ROOT, "influencer_node_evidence.json")))

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("WROTE", os.path.relpath(p, ROOT))

def bars_from_hist(ax, hist, color, label=None):
    e = np.array(hist["edges"]); c = np.array(hist["counts"])
    ax.bar((e[:-1]+e[1:])/2, c, width=(e[1:]-e[:-1])*0.95, color=color, alpha=0.85,
           edgecolor="white", linewidth=0.5, label=label)

# ---------------------------------------------------------------- F1 calibration + loess
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
    ax.set_title("Belief calibration: over-optimism (above diagonal)\nlinear fit vs flexible loess curve")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=10, loc="lower right")
    save(fig, "calibration_patched.png")

# ---------------------------------------------------------------- F2 sen welfare (RSV Gini)
def fig_sen_welfare():
    fig, ax = plt.subplots()
    bars_from_hist(ax, eda["sen_welfare_hist"], BLUE)
    d = eda["sen_welfare_desc"]
    ax.axvline(d["mean"], color=RED, lw=2, label=f"mean = {d['mean']:.3f}")
    ax.axvline(d["50%"], color=ORANGE, lw=2, ls="--", label=f"median = {d['50%']:.3f}")
    ax.set_xlabel("Sen social welfare (RSV-renormalised Gini)")
    ax.set_ylabel("number of simulations")
    ax.set_title("Sen welfare across 50,000 simulations\n(negatives-admissible RSV Gini)")
    ax.legend(fontsize=11)
    save(fig, "sen_welfare_patched.png")

# ---------------------------------------------------------------- F3 dispersion + dip vs BC
def fig_dispersion_dip():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars_from_hist(axes[0], eda["psi_hist"], BLUE)
    axes[0].axvline(eda["psi_desc"]["mean"], color=RED, lw=2,
                    label=f"mean = {eda['psi_desc']['mean']:.3f}")
    axes[0].set_xlabel(r"belief dispersion $\psi = 4\,\mathrm{Var}$ (detector TPR belief)")
    axes[0].set_ylabel("number of simulations")
    axes[0].set_title("Belief dispersion\n(dispersion sense)")
    axes[0].legend(fontsize=11)
    fbc, fdip = eda["frac_bc_gt_0556"], eda["frac_dip_p_lt_0p05"]
    axes[1].bar(["Sarle BC > 5/9", "Hartigan dip p<0.05"], [fbc, fdip],
                color=[GREY, ORANGE], alpha=0.9, edgecolor="white")
    for i, v in enumerate([fbc, fdip]):
        axes[1].text(i, v+0.005, f"{v:.1%}", ha="center", fontsize=13)
    axes[1].set_ylabel("fraction of simulations flagged multimodal")
    axes[1].set_ylim(0, max(fbc, fdip)*1.25)
    axes[1].set_title("Multimodality of detector beliefs\nthe rigorous dip test flags fewer than Sarle's BC")
    save(fig, "dispersion_bimodality_patched.png")

# ---------------------------------------------------------------- F4 welfare radar
def fig_welfare_radar():
    rm = eda["regime_means"]
    axes_lbl = ["tpr", "fpr", "v_cost", "loss", "p_fake", "veracity_differential"]
    disp = ["TPR", "FPR", "v_cost", "loss", r"$p_{\mathrm{fake}}$", "veracity diff."]
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
        ax.set_title(f"welfare quartile {i+1}\nSen welfare = {rm[i]['sen_welfare']:.3f}", fontsize=12, pad=18)
    fig.suptitle("Parameter profile by welfare quartile (min-max scaled; RSV-Gini welfare)", fontsize=15)
    save(fig, "welfare_radar_patched.png")

# ---------------------------------------------------------------- F5 seeding forest: d_z vs d_av
def fig_seeding_forest():
    rows = [  # (label, section, key)
        ("Cascade depth (hops)", "cascade_level", "mean_max_depth"),
        ("Believed P(true)",     "node_level",    "mean_true_belief"),
        ("Belief dispersion",    "node_level",    "psi"),
        ("Mean payoff",          "sim_level",     "avg_payoff"),
        ("Sen welfare (RSV)",    "node_level",    "sen_welfare"),
        ("Veracity differential","sim_level",     "veracity_differential"),
        ("Fake-news reach",      "sim_level",     "avg_fake_cascade"),
    ]
    labels = [r[0] for r in rows]
    dz = [cmp[s][k]["cohens_dz"] for _, s, k in rows]
    dav = [cmp[s][k]["cohens_d_av"] for _, s, k in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    for thr, lab in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        for s in (1, -1):
            ax.axvline(s*thr, color=GREY, ls=":", lw=0.8, alpha=0.6)
    ax.axvline(0, color="black", lw=1)
    ax.scatter(dz, y, s=90, color=BLUE, label=r"$d_z$ (CRN-inflated)", zorder=5)
    ax.scatter(dav, y, s=90, color=ORANGE, marker="D", label=r"$d_{av}$ (correlation-free)", zorder=5)
    for i in range(len(rows)):
        ax.plot([dav[i], dz[i]], [y[i], y[i]], color=GREY, lw=1, alpha=0.5, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("standardised effect size  (influencer $-$ random seeding)")
    ax.set_title("Seeding effects shrink under the correlation-free $d_{av}$\n"
                 "shared random seeds inflate the paired $d_z$")
    ax.legend(fontsize=10, loc="lower right")
    save(fig, "seeding_effects_forest.png")

# ---------------------------------------------------------------- F6 influencer vs ordinary
def fig_influencer_vs_ordinary():
    P = inf["paired_per_sim"]; pooled = inf["pooled"]
    panels = [
        ("Raw reputation\n(payoff)", pooled["payoff"]["influencer_mean"], pooled["payoff"]["noninfluencer_mean"],
         P["payoff"]["cohens_dz"], P["payoff"]["cohens_d_av"]),
        ("Degree-fair reputation\n(captured fraction)", pooled["agent_veracity"]["influencer_mean"],
         pooled["agent_veracity"]["noninfluencer_mean"], P["agent_veracity"]["cohens_dz"], P["agent_veracity"]["cohens_d_av"]),
        ("Messages shared\n(of 1000)", pooled["total_shares"]["influencer_mean"], pooled["total_shares"]["noninfluencer_mean"],
         P["total_shares"]["cohens_dz"], P["total_shares"]["cohens_d_av"]),
        ("True-share ratio", pooled["truth_share_ratio_participants"]["influencer_mean"],
         pooled["truth_share_ratio_participants"]["noninfluencer_mean"], P["truth_share_ratio"]["cohens_dz"], P["truth_share_ratio"]["cohens_d_av"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    for ax, (title, vi, vo, dz, dav) in zip(axes, panels):
        ax.bar(["influencer", "ordinary"], [vi, vo], color=[ORANGE, BLUE], alpha=0.9, edgecolor="white")
        for j, v in enumerate([vi, vo]):
            ax.text(j, v + max(vi, vo)*0.01, f"{v:.3g}", ha="center", fontsize=12)
        ax.set_title(f"{title}\n$d_z$={dz:+.2f},  $d_{{av}}$={dav:+.2f}", fontsize=12)
        ax.set_ylim(0, max(vi, vo)*1.18)
    fig.suptitle("Influencers (15 hubs) vs ordinary (285) agents: a reach artefact, not skill\n"
                 "raw payoff differs ~5.8$\\times$ but degree-fair reputation is identical; "
                 "the $d_z$ gaps shrink to ~0 under $d_{av}$", fontsize=13)
    save(fig, "influencer_vs_ordinary.png")

# ---------------------------------------------------------------- F7 fake-reach TOST equivalence
def fig_fake_reach_equivalence():
    m = cmp["sim_level"]["avg_fake_cascade"]
    lo, hi = m["tost_equivalence"]["bounds"]; md = m["mean_diff"]
    cb = m["ci95_diff_boot"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axvspan(lo, hi, color=GREY, alpha=0.18, label="equivalence band (SESOI $\\pm$0.05)")
    ax.axvline(0, color="black", lw=1)
    ax.errorbar([md], [0], xerr=[[md-cb[0]], [cb[1]-md]], fmt="o", color=ORANGE, ms=11, capsize=6,
                lw=2, label="influencer $-$ random  (95% bootstrap CI)")
    ax.set_yticks([]); ax.set_xlim(-0.06, 0.06)
    ax.set_xlabel("difference in fake-news reach (fraction of network)")
    ax.set_title("Hub seeding does not amplify fake-news reach: formally EQUIVALENT to zero\n"
                 f"diff = {md:+.4f}, CI $\\subset$ $\\pm$0.05  (TOST p = {m['tost_equivalence']['p_tost']:.2g})")
    ax.legend(fontsize=10, loc="upper left")
    save(fig, "fake_reach_equivalence.png")

# ---------------------------------------------------------------- F8 pseudoreplication false precision
def fig_pseudoreplication():
    pp = inf["pseudoreplication_payoff"]
    naive, valid, ratio = pp["naive_independent_se_15M"], pp["valid_paired_se"], pp["false_precision_ratio"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(["naive SE\n(15M rows as independent)", "valid SE\n(per-simulation paired)"],
                  [naive, valid], color=[RED, BLUE], alpha=0.9, edgecolor="white")
    for b, v in zip(bars, [naive, valid]):
        ax.text(b.get_x()+b.get_width()/2, v+0.4, f"{v:.2f}", ha="center", fontsize=13)
    ax.set_ylabel("standard error of the influencer payoff effect")
    ax.set_title("Pooling 15M nested rows fabricates precision (pseudoreplication)\n"
                 f"the naive SE is {ratio:.1f}$\\times$ too small (Hurlbert 1984)")
    ax.annotate("", xy=(1, valid), xytext=(0, naive),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.5))
    ax.text(0.5, (naive+valid)/2, f"{ratio:.1f}$\\times$", ha="center", va="bottom", fontsize=15, color=GREY)
    ax.set_ylim(0, valid*1.2)
    save(fig, "pseudoreplication_precision.png")

if __name__ == "__main__":
    for fn in (fig_calibration, fig_sen_welfare, fig_dispersion_dip, fig_welfare_radar,
               fig_seeding_forest, fig_influencer_vs_ordinary, fig_fake_reach_equivalence,
               fig_pseudoreplication):
        fn()
    print("DONE; figures in", os.path.relpath(OUT, ROOT))
