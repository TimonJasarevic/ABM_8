"""Near-edge robustness spot-check for the l>g payoff asymmetry.

The manuscript assumes l>g with g=1.0 fixed and P_f (l) swept over [1,3]. The realised Sobol'
sample never attains the l=g edge (minimum sampled loss 1.00016), but 408 of the 16,384 design
points lie within 5% of it. This script recomputes the headline seeding comparisons on the full
design and with those near-edge design points excluded, so the manuscript can state that the
results do not hinge on the boundary region.

Writes analysis/runs/loss_edge/loss_edge_evidence.json with `full` and `excluded`
blocks (ignition means and d_av, fake-reach means and d_av, design-point-clustered TOST).

Usage:  python analysis/robustness/loss_edge_check.py
(run in the main analysis environment, see requirements.txt)
"""
import json
import os
import sys

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ANALYSIS)                        # analysis/ on sys.path
sys.path.insert(0, os.path.join(_ANALYSIS, "lib"))   # analysis/lib on sys.path

import numpy as np
import pandas as pd

import data_io
from utils import cohens_d_av, tost_clustered

BASE_CSV = data_io.BASELINE_SVD_CSV
INFL_CSV = data_io.INFLUENCER_SVD_CSV
DESIGN = data_io.DESIGN_CSV
OUT = os.path.join(data_io.RUNS_DIR, "loss_edge")

EDGE = 1.05
SESOI, ALPHA = data_io.SESOI, 0.05
N_PAIRS, N_CLUSTERS, REPS = data_io.N_PAIRS, data_io.N_DESIGNS, data_io.REPS


def load(path, value_cols):
    cols = ["global_sim_id", "design_id", "loss", *value_cols]
    df = pd.read_csv(path, usecols=cols).sort_values("global_sim_id").reset_index(drop=True)
    assert len(df) == N_PAIRS, (path, len(df))
    return df


def clustered_tost(diff, design_ids):
    """utils.tost_clustered on the design-point clusters, remapped to this file's output
    schema. Under the balanced design the CR1 SE equals the cluster-mean SE the former
    local inline copy computed; only float-operation order differs (~1e-15, below every
    reported precision)."""
    t = tost_clustered(diff, design_ids, sesoi=SESOI, alpha=ALPHA, expect_sizes={REPS})
    return {
        "n_clusters": t["n_blocks"],
        "mean_diff": t["mean_diff"],
        "se_clustered": t["se_cr1"],
        "ci90_clustered": t["ci90"],
        "p_tost": t["p_tost"],
        "equivalent": t["equivalent"],
        "smallest_passing_sesoi": t["smallest_passing_sesoi"],
    }


def block(base_reach, infl_reach, base_ign, infl_ign, design_ids, keep_mask_sim):
    br, ir = base_reach[keep_mask_sim], infl_reach[keep_mask_sim]
    bi, ii = base_ign[keep_mask_sim], infl_ign[keep_mask_sim]
    dids = design_ids[keep_mask_sim]
    return {
        "n_sims_per_config": int(keep_mask_sim.sum()),
        "ignition_mean_baseline": float(bi.mean()),
        "ignition_mean_influencer": float(ii.mean()),
        "ignition_d_av": float(cohens_d_av(ii, bi)),
        "reach_mean_baseline": float(br.mean()),
        "reach_mean_influencer": float(ir.mean()),
        "reach_d_av": float(cohens_d_av(ir, br)),
        "reach_tost": clustered_tost(ir - br, dids),
    }


def main():
    a = load(BASE_CSV, ["avg_fake_cascade", "ignition_rate"])
    b = load(INFL_CSV, ["avg_fake_cascade", "ignition_rate"])
    assert np.array_equal(a["global_sim_id"].values, b["global_sim_id"].values)
    assert np.array_equal(a["design_id"].values, b["design_id"].values)
    assert np.allclose(a["loss"].values, b["loss"].values)

    # loss is constant within each design point and matches design.csv (row i-1 for design_id i)
    per_design_loss = a.groupby("design_id")["loss"].agg(["min", "max"])
    assert (per_design_loss["min"] == per_design_loss["max"]).all()
    design = pd.read_csv(DESIGN, usecols=["loss"])
    assert len(design) == N_CLUSTERS
    assert np.allclose(per_design_loss["min"].values, design["loss"].values)

    loss_by_design = per_design_loss["min"]
    edge_designs = loss_by_design.index[loss_by_design.values < EDGE]
    keep_sims = ~a["design_id"].isin(edge_designs).values

    base_reach_dids = a["design_id"].values
    base_reach = a["avg_fake_cascade"].values
    infl_reach = b["avg_fake_cascade"].values
    base_ign = a["ignition_rate"].values
    infl_ign = b["ignition_rate"].values

    full = block(base_reach, infl_reach, base_ign, infl_ign, base_reach_dids, np.ones(N_PAIRS, dtype=bool))
    excl = block(base_reach, infl_reach, base_ign, infl_ign, base_reach_dids, keep_sims)

    # gates: the full-design numbers must reproduce the published values
    assert abs(full["ignition_mean_baseline"] - 0.39552) < 5e-4
    assert abs(full["ignition_mean_influencer"] - 0.35075) < 5e-4
    assert abs(full["reach_mean_baseline"] - data_io.PUBLISHED_MEANS[0]) < 5e-5
    assert abs(full["reach_mean_influencer"] - data_io.PUBLISHED_MEANS[1]) < 5e-5

    evidence = {
        "generated_for": "Tier A-13/U2 near-edge robustness, POSITIONING_REVIEW_2026-07-02",
        "inputs": [BASE_CSV, INFL_CSV, DESIGN],
        "g_fixed": 1.0,
        "loss_min_sampled": float(loss_by_design.min()),
        "edge_threshold": EDGE,
        "n_design_points_excluded": int(len(edge_designs)),
        "n_design_points_kept": int(N_CLUSTERS - len(edge_designs)),
        "full": full,
        "excluded": excl,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "loss_edge_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=1)

    print(f"loss min sampled = {evidence['loss_min_sampled']:.6f}; "
          f"excluded {evidence['n_design_points_excluded']} design points (< {EDGE})")
    for name, blk in (("FULL", full), ("EXCLUDED", excl)):
        t = blk["reach_tost"]
        print(f"{name}: ignition {blk['ignition_mean_baseline']:.4f}/{blk['ignition_mean_influencer']:.4f} "
              f"(d_av {blk['ignition_d_av']:+.3f}); reach {blk['reach_mean_baseline']:.4f}/"
              f"{blk['reach_mean_influencer']:.4f} (d_av {blk['reach_d_av']:+.3f}); "
              f"TOST md {t['mean_diff']:+.5f} CI90 [{t['ci90_clustered'][0]:+.5f}, {t['ci90_clustered'][1]:+.5f}] "
              f"{'EQUIVALENT' if t['equivalent'] else 'NOT equivalent'} (min SESOI {t['smallest_passing_sesoi']:.5f})")
    print("WROTE", os.path.join(OUT, "loss_edge_evidence.json"))


if __name__ == "__main__":
    main()
