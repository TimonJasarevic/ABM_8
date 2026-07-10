"""Rebuild <sweep_dir>/simulations.csv with a fifth Sobol output, avg_structural_virality.

The Julia runners record the structural virality of every post-burn-in cascade in cascades.arrow
(WRITE_FULL sweeps) but do not aggregate it into the per-simulation table. This script computes
the per-simulation mean over the 1000 recorded post-burn-in cascades, the same set the runner
averages for avg_payoff and avg_verify_rate (the post-burn-in recording loop in run_experiment_baseline.jl
and run_experiment_influencers.jl), so the result equals what a runner-side avg_structural_virality
would have written, to within ~1e-13 summation rounding. No re-simulation is needed.

Only two columns of cascades.arrow are projected, which still needs roughly 16-24 GB of RAM at the
full sweep size; run it on a machine that holds the sweep (a Snellius node for the cluster sweeps).
On Snellius:
    ~/fnp-venv/bin/python sobol/augment_sv.py <sweep_dir> [out_dir]
locally (needs pandas + pyarrow, e.g. the mzungu env):
    conda run -n mzungu python sobol/augment_sv.py <sweep_dir> [out_dir]

<sweep_dir> must be a WRITE_FULL sweep whose cascades.arrow carries a structural_virality column;
sweeps that predate the metric are rejected. The augmented simulations.csv is written to [out_dir]
(default: <sweep_dir>_sv, leaving the original sweep untouched; pass the sweep dir itself to
overwrite in place). avg_structural_virality is one of sobol/analyze.py's OUTPUTS, so the
follow-up is:
    conda run -n ABM python sobol/analyze.py --sweep <out_dir> --out sobol/results/<tag>
"""
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sweep = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else sweep.rstrip("/") + "_sv"

    # Refuse sweeps that predate the metric rather than producing a silently wrong result.
    names = pa.ipc.open_file(pa.memory_map(f"{sweep}/cascades.arrow", "r")).schema.names
    if "structural_virality" not in names:
        sys.exit(f"ERROR: {sweep}/cascades.arrow has no 'structural_virality' column "
                 f"(only NEW WRITE_FULL sweeps have it). Columns: {names}")
    # Fail before the expensive cascade read; scratch purging can leave a sweep incomplete.
    if not os.path.isfile(f"{sweep}/simulations.csv"):
        sys.exit(f"ERROR: {sweep}/simulations.csv not found")
    os.makedirs(out, exist_ok=True)

    # Per-cascade -> per-simulation mean over the same cascades the runner averages for avg_payoff
    # and avg_verify_rate. Projecting just the two needed columns bounds the memory of the read.
    cd = feather.read_feather(f"{sweep}/cascades.arrow", columns=["global_sim_id", "structural_virality"])
    sv = cd.groupby("global_sim_id")["structural_virality"].mean().rename("avg_structural_virality")
    del cd

    sim = pd.read_csv(f"{sweep}/simulations.csv")
    sim = sim.drop(columns=["avg_structural_virality"], errors="ignore")   # idempotent on re-runs
    sim = sim.merge(sv, on="global_sim_id", how="left")

    missing = int(sim["avg_structural_virality"].isna().sum())
    assert missing == 0, f"{missing} simulations have no structural_virality (malformed sweep?)"

    sim.to_csv(f"{out}/simulations.csv", index=False)
    print(f"WROTE {out}/simulations.csv  ({len(sim)} rows; avg_structural_virality "
          f"mean={sim['avg_structural_virality'].mean():.4f}, "
          f"min={sim['avg_structural_virality'].min():.4f}, max={sim['avg_structural_virality'].max():.4f})")


if __name__ == "__main__":
    main()
