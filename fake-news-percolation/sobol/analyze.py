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

OUTPUTS = ["veracity_differential", "avg_verify_rate", "avg_payoff", "sen_welfare"]


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
            _plot(o, comp, problem["names"], Si, out)
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

    print(res.to_string(index=False))
    print("\nStochastic fraction E[V(Y|X)]/V(Y) per output:")
    for o in OUTPUTS:
        print(f"  {o}: {res.loc[res.output == o, 'stoch_frac'].iloc[0]:.3f}")
    print(f"\nReplicates per design point R = {R}; saved indices, table, and figures to {out}/")


def _plot(output, component, names, Si, out: Path):
    """Course-style horizontal error-bar plot of S1 and ST for one component."""
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(Si["S1"], y - 0.12, xerr=Si["S1_conf"], fmt="o", capsize=3, label="$S_1$ first-order")
    ax.errorbar(Si["ST"], y + 0.12, xerr=Si["ST_conf"], fmt="s", capsize=3, label="$S_T$ total-order")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Sobol index")
    ax.set_title(f"{output} ({component})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"sa_sobol_{output}_{component}.png", dpi=150)
    plt.close(fig)


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
