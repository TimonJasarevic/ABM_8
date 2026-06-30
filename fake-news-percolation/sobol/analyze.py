#!/usr/bin/env python3
"""Sobol sensitivity analysis (Approach IV) for the deepfake-percolation Saltelli sweep.

Per Saltelli design point we have R stochastic replicates of each scalar output.
Following the law of total variance (Carmona-Cabrero et al. 2024, JASSS 27(1):16):

    Var(Y) = Var_X(E[Y|X])      (deterministic / structural)
           + E_X(Var[Y|X])      (stochastic)

we compute, per output, SEPARATE Sobol decompositions:
  - deterministic indices: sobol.analyze on the per-point replicate MEANS  m_i
    -> decomposes Var_X(E[Y|X]) : which factors drive the expected outcome.
  - stochastic   indices: sobol.analyze on the per-point replicate VARIANCES s_i^2
    -> decomposes Var_X(Var[Y|X]) : which factors drive the model's noisiness.
plus the global stochastic fraction E_X(Var[Y|X]) / Var(Y) (bias-corrected).

Each analysis reports S1, ST and S2 with bootstrap CIs (SALib uses the never-negative
Jansen estimator for ST). All outputs (including sen_welfare, computed in the Julia
runner) come from simulations.csv, so this runs in conda env `ABM` with no pyarrow.

    conda run -n ABM python sobol/analyze.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUTS = ["veracity_differential", "avg_verify_rate", "avg_payoff", "sen_welfare", "avg_structural_virality"]

OUTPUT_LABELS = {
    "veracity_differential": "Veracity\ndifferential",
    "avg_verify_rate": "Verification\nrate",
    "avg_payoff": "Average\npayoff",
    "sen_welfare": "SEN\nwelfare",
    "avg_structural_virality": "Structural\nvirality",
}

FACTOR_ORDER = [
    "v_cost",
    "loss",
    "tpr",
    "fpr",
    "p_fake",
    "log10_lambda",
    "loss_aversion",
]

FACTOR_LABELS = {
    "v_cost": "Verification cost",
    "loss": "Misinformation loss",
    "tpr": "True positive rate",
    "fpr": "False positive rate",
    "p_fake": "Fake-news probability",
    "log10_lambda": r"Rationality ($\log_{10}\lambda$)",
    "loss_aversion": "Loss aversion",
}


def latest_sweep() -> Path:
    sweeps = sorted(Path("data").glob("sweep_*"))
    if not sweeps:
        raise SystemExit("No data/sweep_* found. Run make_design.py, then the Julia sweep.")
    return sweeps[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=None, help="a data/sweep_* dir (default: latest)")
    ap.add_argument("--problem", default="sobol/problem.json")
    ap.add_argument("--out", default="sobol/results")
    ap.add_argument("--nboot", type=int, default=1000, help="bootstrap resamples for CIs")
    args = ap.parse_args()

    meta = json.loads(Path(args.problem).read_text())
    problem = meta["problem"]
    cso = meta["calc_second_order"]
    n_rows = meta["n_rows"]

    sweep = Path(args.sweep) if args.sweep else latest_sweep()
    sim = pd.read_csv(sweep / "simulations.csv")

    # Per design point, in design_id order: replicate mean, sample variance, count.
    g = sim.groupby("design_id")
    means = g[OUTPUTS].mean().sort_index()
    varis = g[OUTPUTS].var(ddof=1).sort_index()
    reps = g.size().sort_index()

    if len(means) != n_rows:
        raise SystemExit(
            f"Design mismatch: sweep has {len(means)} design points but problem.json "
            f"expects {n_rows}. Was this sweep run from sobol/design.csv?"
        )
    if reps.nunique() != 1:
        raise SystemExit(f"Replicate count varies across design points: {reps.value_counts().to_dict()}")
    R = int(reps.iloc[0])
    if R < 2:
        raise SystemExit("Approach IV needs R >= 2 replicates per design point to estimate within-point variance.")

    # The guards above check shape only. Verify the sweep's per-point factor values actually
    # match the current design.csv: a same-shaped but different/stale design (e.g. a re-run of
    # make_design.py with a different --seed) otherwise passes silently and is analysed against
    # the wrong Saltelli ordering. The factor values are constant within a design_id.
    design_path = Path(args.problem).with_name("design.csv")
    design = pd.read_csv(design_path)
    sim_factors = g[["v_cost", "loss", "tpr", "fpr", "p_fake", "rationality", "loss_aversion"]].first().sort_index()
    sim_factors["log10_lambda"] = np.log10(sim_factors["rationality"])  # runner stores 10**log10_lambda
    got = sim_factors[problem["names"]].to_numpy()
    want = design[problem["names"]].to_numpy()
    if got.shape != want.shape or not np.allclose(got, want, rtol=1e-6, atol=1e-9):
        raise SystemExit(
            f"Sweep/design mismatch: per-point factor values in {sweep / 'simulations.csv'} do not "
            f"match {design_path}. Was this sweep run from the current design?"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, s2_rows = [], []
    for o in OUTPUTS:
        M = means[o].to_numpy()
        Vv = varis[o].to_numpy()

        Sdet = sobol_analyze.analyze(problem, M, calc_second_order=cso,
                                     num_resamples=args.nboot, conf_level=0.95, seed=42)
        Ssto = sobol_analyze.analyze(problem, Vv, calc_second_order=cso,
                                     num_resamples=args.nboot, conf_level=0.95, seed=42)

        sto_var = float(np.nanmean(Vv))                       # ~ E_X(Var[Y|X])
        det_var = max(0.0, float(M.var()) - sto_var / R)      # ~ Var_X(E[Y|X]), bias-corrected
        stoch_frac = sto_var / (det_var + sto_var + 1e-12)

        for comp, Si in (("deterministic", Sdet), ("stochastic", Ssto)):
            for i, name in enumerate(problem["names"]):
                rows.append(dict(
                    output=o, component=comp, factor=name,
                    S1=Si["S1"][i], S1_conf=Si["S1_conf"][i],
                    ST=Si["ST"][i], ST_conf=Si["ST_conf"][i],
                    stoch_frac=stoch_frac,
                ))
            if cso:
                names = problem["names"]
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        s2_rows.append(dict(
                            output=o, component=comp, pair=f"{names[i]}*{names[j]}",
                            S2=Si["S2"][i, j], S2_conf=Si["S2_conf"][i, j],
                        ))

    res = pd.DataFrame(rows)
    res.to_csv(out / "sa_indices.csv", index=False)
    _latex(res, out / "sa_indices.tex")
    if s2_rows:
        pd.DataFrame(s2_rows).to_csv(out / "sa_indices_S2.csv", index=False)

    make_dotgrids(res, out)

    print(res.to_string(index=False))
    print("\nStochastic fraction E[V(Y|X)]/V(Y) per output:")
    for o in OUTPUTS:
        print(f"  {o}: {res.loc[res.output == o, 'stoch_frac'].iloc[0]:.3f}")
    print(f"\nReplicates per design point R = {R}; saved indices, table, and figures to {out}/")


def _label_from_dir(out_dir: Path) -> str:
    """Config label for the figure filename, derived from the results dir name."""
    name = out_dir.name
    return name.split("_sv")[0] if "_sv" in name else name


def make_dotgrids(df: pd.DataFrame, out_dir: Path) -> None:
    """Consolidated Sobol dot-grid for each variance component."""
    for component in ("deterministic", "stochastic"):
        plot_dotgrid(df, out_dir, component)


def plot_dotgrid(df: pd.DataFrame, out_dir: Path, component: str) -> None:
    """Dot-grid of S1 (circle) and ST (square) for all outputs and factors of one
    variance component: x-axis = outputs, color = factor."""
    sub = df[df["component"] == component].copy()
    label = _label_from_dir(out_dir)

    fig, ax = plt.subplots(figsize=(10, 5))

    x_base = np.arange(len(OUTPUTS))
    factor_offsets = np.linspace(-0.33, 0.33, len(FACTOR_ORDER))
    metric_offsets = {"S1": -0.018, "ST": 0.018}
    marker_map = {"S1": "o", "ST": "s"}
    conf_map = {"S1": "S1_conf", "ST": "ST_conf"}

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    factor_colors = {f: colors[i % len(colors)] for i, f in enumerate(FACTOR_ORDER)}

    for factor_index, factor in enumerate(FACTOR_ORDER):
        factor_df = sub[sub["factor"] == factor]
        for metric in ("S1", "ST"):
            xs, ys, yerrs = [], [], []
            for output_index, output in enumerate(OUTPUTS):
                row = factor_df[factor_df["output"] == output]
                if row.empty:
                    continue
                xs.append(x_base[output_index] + factor_offsets[factor_index] + metric_offsets[metric])
                ys.append(float(row[metric].iloc[0]))
                yerrs.append(float(row[conf_map[metric]].iloc[0]))
            ax.errorbar(
                xs, ys, yerr=yerrs, fmt=marker_map[metric], linestyle="none",
                markersize=5.5, capsize=3, elinewidth=1.0, markeredgewidth=0.9,
                color=factor_colors[factor], ecolor=factor_colors[factor], alpha=0.95,
            )

    ax.axhline(0, linewidth=0.8, color="black", alpha=0.8)

    ax.set_xticks(x_base)
    ax.set_xticklabels([OUTPUT_LABELS[o] for o in OUTPUTS], fontsize=10)

    if component == "deterministic":
        ax.set_ylim(-0.10, 1.05)
    else:
        # Stochastic CIs are much wider; fit the axis so error bars aren't clipped.
        lo = min(0.0, float((sub["S1"] - sub["S1_conf"]).min()))
        hi = float(max((sub["ST"] + sub["ST_conf"]).max(), (sub["S1"] + sub["S1_conf"]).max()))
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.60, len(OUTPUTS) - 0.40)

    ax.set_xlabel("Measured output", fontsize=11)
    ax.set_ylabel("Sobol index", fontsize=11)
    ax.set_title(f"{component.capitalize()} Sobol sensitivity by output", fontsize=14)

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    factor_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=factor_colors[f],
               label=FACTOR_LABELS[f], markersize=6.5)
        for f in FACTOR_ORDER
    ]
    metric_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="black", label=r"$S_1$", markersize=6.5),
        Line2D([0], [0], marker="s", linestyle="none", color="black", label=r"$S_T$", markersize=6.5),
    ]

    fig.legend(handles=factor_handles, title="Parameter", loc="lower center",
               bbox_to_anchor=(0.5, 0.045), ncol=4, frameon=False, fontsize=9, title_fontsize=10)
    fig.legend(handles=metric_handles, title="Index", loc="lower center",
               bbox_to_anchor=(0.5, -0.035), ncol=2, frameon=False, fontsize=9, title_fontsize=10)

    fig.tight_layout(rect=(0, 0.22, 1, 1))

    out_path = out_dir / f"sa_sobol_{component}_{label}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _latex(res: pd.DataFrame, path: Path):
    lines = [
        r"\begin{tabular}{lllrrrr}", r"\toprule",
        r"Output & Component & Factor & $S_1$ & $\pm$ & $S_T$ & $\pm$ \\", r"\midrule",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"{r.output} & {r.component} & {r.factor} & {r.S1:.3f} & {r.S1_conf:.3f} & "
            f"{r.ST:.3f} & {r.ST_conf:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
