#!/usr/bin/env python3
"""Generate a Saltelli design for the deepfake-percolation Sobol sensitivity analysis.

Writes (relative to the project root):
  sobol/design.csv   one row per Saltelli evaluation point, with a header row,
                     columns in SALib problem order. The Julia runner reads this
                     and runs the ABM at each row (x R stochastic replicates).
  sobol/problem.json the SALib problem + metadata, so analyze.py reconstructs the
                     exact sample ordering and second-order setting.

Research basis: Saltelli quasi-random (scrambled Sobol-sequence) sampling with a
power-of-two base size N; first+total(+second) order indices at a cost of N(2k+2)
model evaluations (Saxton et al. 2024; SALib docs). Sobol-sequence sampling is
preferred over Latin Hypercube for index estimation.

Verified against the project's `ABM` conda env (SALib 1.4.8):
    conda run -n ABM python sobol/make_design.py --N 1024
"""
import argparse
import json
from pathlib import Path

import numpy as np
from SALib.sample import sobol as salib_sample  # SALib 1.4.8 (saltelli.sample is deprecated)

# Factor order is load-bearing: the Julia runner reads columns 1..7 in THIS order,
# and analyze.py feeds outputs to sobol.analyze in this same factor order.
PROBLEM = {
    "num_vars": 7,
    "names": ["v_cost", "loss", "tpr", "fpr", "p_fake", "log10_lambda", "loss_aversion"],
    "bounds": [
        [0.01, 1.00],   # 1: verification cost c
        [1.00, 3.00],   # 2: reputation loss P_f
        [0.60, 0.99],   # 3: detector true-positive rate TPR
        [0.01, 0.40],   # 4: detector false-positive rate FPR
        [0.05, 0.95],   # 5: fake-news rate p_fake
        [-2.00, 2.00],  # 6: log10(rationality lambda); Julia back-transforms 10**x
        [1.00, 3.00],   # 7: loss_aversion lambda_LA (linear; 1.0 = loss-neutral)
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1024,
                    help="Saltelli base sample size (use a power of two)")
    ap.add_argument("--no-second-order", action="store_true",
                    help="skip S2 indices: cost N(k+2) instead of N(2k+2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="sobol")
    args = ap.parse_args()

    calc_second_order = not args.no_second_order
    X = salib_sample.sample(PROBLEM, args.N,
                            calc_second_order=calc_second_order, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    header = ",".join(PROBLEM["names"])
    # comments="" -> plain header line; the Julia runner reads with skipstart=1.
    np.savetxt(out / "design.csv", X, delimiter=",", header=header, comments="")

    meta = {
        "problem": PROBLEM,
        "N": args.N,
        "calc_second_order": calc_second_order,
        "seed": args.seed,
        "n_rows": int(X.shape[0]),
        "k": PROBLEM["num_vars"],
    }
    (out / "problem.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out / 'design.csv'}: {X.shape[0]} design points x {X.shape[1]} factors "
          f"(N={args.N}, calc_second_order={calc_second_order}).")
    print(f"Model runs needed = {X.shape[0]} design points x R replicates "
          f"(e.g. R=10 -> {10 * X.shape[0]:,} simulations).")


if __name__ == "__main__":
    main()
