"""Near-edge robustness spot-check for the l>g payoff asymmetry.

The manuscript assumes l>g with g=1.0 fixed and P_f (l) swept over [1,3]. The realised Sobol'
sample never attains the l=g edge (minimum sampled loss 1.00016), but 408 of the 16,384 design
points lie within 5% of it. This script recomputes the headline seeding comparisons on the full
design and with those near-edge design points excluded, so the manuscript can state that the
results do not hinge on the boundary region.

Writes analysis/runs/loss_edge_2026_07_03/loss_edge_evidence.json with `full` and `excluded`
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
from scipy import stats

import data_io
from utils import cohens_d_av

BASE_SV = data_io.BASELINE_SVD_CSV
INFL_SV = data_io.INFLUENCER_SVD_CSV
BASE_SVD = data_io.BASELINE_SVD_CSV
INFL_SVD = data_io.INFLUENCER_SVD_CSV
DESIGN = data_io.DESIGN_CSV
OUT = os.path.join(data_io.RUNS_DIR, "loss_edge_2026_07_03")

EDGE = 1.05
SESOI, ALPHA = 0.05, 0.05
N_PAIRS, N_CLUSTERS, REPS = 491_520, 16_384, 30


def load(path, value_col):
    cols = ["global_sim_id", "design_id", "loss", value_col]
    df = pd.read_csv(path, usecols=cols).sort_values("global_sim_id").reset_index(drop=True)
    assert len(df) == N_PAIRS, (path, len(df))
    return df


def clustered_tost(diff, design_ids):
    # equals utils.tost_clustered under this balanced design (kept as a local inline copy)
    g = pd.Series(diff).groupby(design_ids)
    cm = g.mean().values
    md = float(cm.mean())
    se = float(cm.std(ddof=1) / np.sqrt(cm.size))
    df = cm.size - 1
    p_lo = float(stats.t.sf((md + SESOI) / se, df))
    p_hi = float(stats.t.cdf((md - SESOI) / se, df))
    t95 = float(stats.t.ppf(0.95, df))
    return {
        "n_clusters": int(cm.size),
        "mean_diff": md,
        "se_clustered": se,
        "ci90_clustered": [md - t95 * se, md + t95 * se],
        "p_tost": max(p_lo, p_hi),
        "equivalent": bool(max(p_lo, p_hi) < ALPHA),
        "smallest_passing_sesoi": float(abs(md) + t95 * se),
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
    a_sv = load(BASE_SV, "avg_fake_cascade")
    b_sv = load(INFL_SV, "avg_fake_cascade")
    a_svd = load(BASE_SVD, "ignition_rate")
    b_svd = load(INFL_SVD, "ignition_rate")
    for x, y in ((a_sv, b_sv), (a_sv, a_svd), (a_sv, b_svd)):
        assert np.array_equal(x["global_sim_id"].values, y["global_sim_id"].values)
        assert np.array_equal(x["design_id"].values, y["design_id"].values)
        assert np.allclose(x["loss"].values, y["loss"].values)

    # loss is constant within each design point and matches design.csv (row i-1 for design_id i)
    per_design_loss = a_sv.groupby("design_id")["loss"].agg(["min", "max"])
    assert (per_design_loss["min"] == per_design_loss["max"]).all()
    design = pd.read_csv(DESIGN, usecols=["loss"])
    assert len(design) == N_CLUSTERS
    assert np.allclose(per_design_loss["min"].values, design["loss"].values)

    loss_by_design = per_design_loss["min"]
    edge_designs = loss_by_design.index[loss_by_design.values < EDGE]
    keep_sims = ~a_sv["design_id"].isin(edge_designs).values

    base_reach_dids = a_sv["design_id"].values
    base_reach = a_sv["avg_fake_cascade"].values
    infl_reach = b_sv["avg_fake_cascade"].values
    base_ign = a_svd["ignition_rate"].values
    infl_ign = b_svd["ignition_rate"].values

    full = block(base_reach, infl_reach, base_ign, infl_ign, base_reach_dids, np.ones(N_PAIRS, dtype=bool))
    excl = block(base_reach, infl_reach, base_ign, infl_ign, base_reach_dids, keep_sims)

    # gates: the full-design numbers must reproduce the published values
    assert abs(full["ignition_mean_baseline"] - 0.39552) < 5e-4
    assert abs(full["ignition_mean_influencer"] - 0.35075) < 5e-4
    assert abs(full["reach_mean_baseline"] - 0.22555) < 5e-5
    assert abs(full["reach_mean_influencer"] - 0.22817) < 5e-5

    evidence = {
        "generated_for": "Tier A-13/U2 near-edge robustness, POSITIONING_REVIEW_2026-07-02",
        "inputs": [BASE_SV, INFL_SV, BASE_SVD, INFL_SVD, DESIGN],
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
