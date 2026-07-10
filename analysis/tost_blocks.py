"""Saltelli-block-clustered TOST of the fake-news reach seeding contrast.

tost_clustered.py and tost_profile.py cluster on the 16,384 Saltelli design points. The
design, however, is 1,024 base samples expanded to contiguous 16-row blocks
[A, AB_1..7, BA_1..7, B] (SALib 1.4.8 sobol.sample, N=1024, calc_second_order=True,
seed=42). Rows within a block therefore share coordinates, and the design point is not the
exchangeable unit. This script recomputes the whole-design TOST and the full conditional
profile (nine fake-news-prevalence bins, two low-prevalence slices, four rationality bins)
with the base block as the cluster: block = (design_id - 1) // 16.

Estimator: design-point-weighted mean difference with a CR1 cluster-robust standard error,
se = sqrt(G/(G-1) * sum_g u_g^2) / n with u_g the within-cluster residual sums and df = G-1.
Under balanced clusters this equals the one-sample t on cluster means exactly. The overall
test therefore reduces to the tost_clustered.py construction with 1,024 blocks. In
prevalence bins the blocks contribute unbalanced cells of exactly 8 or 16 design points
(p_fake carries the A value on within-block offsets {0,1,2,3,4,6,7,12} and the B value on
{5,8,9,10,11,13,14,15}). CR1 handles this without reweighting the published point
estimates. The block-cell-mean t is reported per bin as a sensitivity. Benjamini-Yekutieli
q-values are attached over the family of all fifteen reported TOSTs (primary) and over the
nine prevalence bins alone (secondary).

Writes analysis/runs/tost_blocks/{tost_blocks_evidence.json,
fake_reach_equivalence_blocks.png, fake_reach_profile_blocks.png}. Supersedes the earlier
design-point runs (tost_clustered, tost_profile), whose values are
embedded per entry for comparison.

Usage: python analysis/tost_blocks.py, run in the main analysis environment
(see requirements.txt).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "lib"))
from utils import bh_fdr, tost_clustered
import data_io
from data_io import (N_PAIRS, N_DESIGNS, N_BLOCKS, BLOCK, REPS, PAIRS_PER_BLOCK,
                     PFAKE_EDGES, LAMBDA_EDGES)

BASE_CSV = data_io.BASELINE_SVD_CSV
INFL_CSV = data_io.INFLUENCER_SVD_CSV
BASE_CSV_PLAIN = os.path.join(data_io.BASELINE_SWEEP, "simulations.csv")
INFL_CSV_PLAIN = os.path.join(data_io.INFLUENCER_SWEEP, "simulations.csv")
DESIGN_CSV = data_io.DESIGN_CSV
OLD_TOST_JSON = os.path.join(data_io.RUNS_DIR, "tost_clustered", "tost_evidence.json")
OLD_PROFILE_JSON = os.path.join(data_io.RUNS_DIR, "tost_profile", "profile_evidence.json")
OUT = os.path.join(data_io.RUNS_DIR, "tost_blocks")

COLS = ["global_sim_id", "design_id", "rep_id", "p_fake", "rationality", "avg_fake_cascade"]
SESOI, ALPHA = data_io.SESOI, 0.05

# within-block offsets carrying the A row's value vs the B row's value, per factor
# (offset = (design_id-1) % 16; layout [A, AB_1..7, BA_1..7, B]; p_fake = factor 5,
# log10_lambda = factor 6; AB_j swaps only column j to B, BA_j swaps only column j to A)
A_PF_OFF = np.array([0, 1, 2, 3, 4, 6, 7, 12])
B_PF_OFF = np.array([5, 8, 9, 10, 11, 13, 14, 15])
A_LAM_OFF = np.array([0, 1, 2, 3, 4, 5, 7, 13])
B_LAM_OFF = np.array([6, 8, 9, 10, 11, 12, 14, 15])

# reproductions obtained from an independent external verifier;
# report-only cross-checks, never asserted
VERIFIER = {
    "overall_ci90_lo": (0.0011, 2e-4),
    "overall_ci90_hi": (0.0041, 2e-4),
    "smallest_passing_sesoi": (0.0041, 2e-4),
    "naive_se_inflation": (8.6, 0.2),
    "low_bin_ci90_lo": (0.042, 2e-3),
    "low_bin_ci90_hi": (0.058, 2e-3),
    "icc_dp_within_block": (0.28, 0.03),
    "de_block_vs_dp": (5.0, 0.5),
}


def load_pairs():
    a, b = data_io.load_seeding_pairs(COLS)                                    # A1, A2, A4
    sizes = a.groupby("design_id").size()
    assert a["design_id"].min() == 1 and a["design_id"].max() == N_DESIGNS               # A3
    assert (sizes == REPS).all()                                                         # A3
    mb, mi = float(a["avg_fake_cascade"].mean()), float(b["avg_fake_cascade"].mean())
    for col in ("p_fake", "rationality"):
        rng = a.groupby("design_id")[col].agg(["min", "max"])
        assert (rng["min"] == rng["max"]).all(), col                                     # A6
    assert np.allclose(a["p_fake"].values, b["p_fake"].values)
    assert np.allclose(a["rationality"].values, b["rationality"].values)
    d = b["avg_fake_cascade"].values - a["avg_fake_cascade"].values  # influencer - baseline
    return a, b, d, mb, mi


def crosscheck_csv_sources(a, b):
    """The _sv CSVs are the reproducibility source of record (their means are the published
    gate); the plain sweep directories hold the same table. Assert bit-identity on every used column
    so either source supports the numbers (A5)."""
    if not (os.path.exists(BASE_CSV_PLAIN) and os.path.exists(INFL_CSV_PLAIN)):
        # Plain sweep views are omitted from the trimmed reproduction package; the _sv CSVs are
        # the source of record, so skip the (optional) bit-identity crosscheck rather than error.
        print("crosscheck_csv_sources: plain sweep CSVs not bundled -- skipping A5 bit-identity check", flush=True)
        return {"skipped": "plain sweep CSVs not bundled in this package"}
    out = {}
    for label, sv_frame, plain_path in (("baseline", a, BASE_CSV_PLAIN),
                                        ("influencer", b, INFL_CSV_PLAIN)):
        p = pd.read_csv(plain_path, usecols=COLS).sort_values("global_sim_id").reset_index(drop=True)
        assert np.array_equal(p["global_sim_id"].values, sv_frame["global_sim_id"].values)
        assert np.array_equal(p["design_id"].values, sv_frame["design_id"].values)
        diffs = {col: float(np.abs(p[col].values - sv_frame[col].values).max())
                 for col in ("p_fake", "rationality", "avg_fake_cascade")}
        assert diffs["p_fake"] <= 1e-12 and diffs["avg_fake_cascade"] <= 1e-12, diffs    # A5
        assert diffs["rationality"] <= 1e-12, diffs                                      # A5
        out[label] = {"plain_csv": plain_path, "max_abs_diff": diffs}
    out["tolerance"] = 1e-12
    return out


def build_dp_table(a, d):
    key = a["design_id"].values
    dp = pd.DataFrame({
        "diff": pd.Series(d).groupby(key).mean(),
        "p_fake": a.groupby("design_id")["p_fake"].first(),
        "log10_lambda": np.log10(a.groupby("design_id")["rationality"].first()),
        "base_reach": a.groupby("design_id")["avg_fake_cascade"].mean(),
    })
    dp["block"] = (dp.index.values - 1) // BLOCK
    dp["offset"] = (dp.index.values - 1) % BLOCK
    assert len(dp) == N_DESIGNS                                                          # A7
    per_block = dp.groupby("block")["offset"].agg(["count", "min", "max", "nunique"])
    assert len(per_block) == N_BLOCKS                                                    # A7
    assert (per_block["count"] == BLOCK).all() and (per_block["nunique"] == BLOCK).all() # A7
    assert abs(float(dp["diff"].mean()) - float(d.mean())) < 1e-12                       # A9
    assert abs(float(dp.groupby("block")["diff"].mean().mean()) - float(d.mean())) < 1e-12  # A9
    return dp


def verify_saltelli_structure(dp):
    """Assert the [A, AB_1..7, BA_1..7, B] layout on the data itself: within each block the
    A-set offsets carry the A row's p_fake / log10_lambda and the B-set offsets the B row's;
    and the per-design p_fake matches sobol/design.csv row-for-row (A8)."""
    pf = dp.sort_values(["block", "offset"])["p_fake"].values.reshape(N_BLOCKS, BLOCK)
    lam = dp.sort_values(["block", "offset"])["log10_lambda"].values.reshape(N_BLOCKS, BLOCK)
    assert np.array_equal(pf[:, A_PF_OFF], np.repeat(pf[:, [0]], len(A_PF_OFF), axis=1))   # A8
    assert np.array_equal(pf[:, B_PF_OFF], np.repeat(pf[:, [15]], len(B_PF_OFF), axis=1))  # A8
    assert np.allclose(lam[:, A_LAM_OFF], np.repeat(lam[:, [0]], len(A_LAM_OFF), axis=1),
                       rtol=0, atol=1e-12)                                                 # A8
    assert np.allclose(lam[:, B_LAM_OFF], np.repeat(lam[:, [15]], len(B_LAM_OFF), axis=1),
                       rtol=0, atol=1e-12)                                                 # A8
    design = pd.read_csv(DESIGN_CSV)
    design_pf = design["p_fake"].values
    dp_pf = dp.sort_index()["p_fake"].values  # index = design_id 1..16384 -> design.csv row 0..16383
    max_err = float(np.abs(dp_pf - design_pf).max())
    assert np.allclose(dp_pf, design_pf), max_err                                          # A8
    n_equal_ab = int((pf[:, 0] == pf[:, 15]).sum())
    return {
        "verified": True,
        "a_pfake_offsets": A_PF_OFF.tolist(),
        "b_pfake_offsets": B_PF_OFF.tolist(),
        "a_lambda_offsets": A_LAM_OFF.tolist(),
        "b_lambda_offsets": B_LAM_OFF.tolist(),
        "pfake_matches_design_csv_max_abs_err": max_err,
        "n_blocks_with_equal_AB_pfake": n_equal_ab,
    }


def variance_components(d, design_ids):
    """Balanced nested ANOVA of the paired differences: replications within design points
    within blocks, plus the naive/design-point/block SE ladder."""
    dp_mean = pd.Series(d).groupby(design_ids).mean()
    blk_of_dp = (dp_mean.index.values - 1) // BLOCK
    blk_mean = dp_mean.groupby(blk_of_dp).mean()
    m = float(d.mean())

    ss_rep = float(((d - dp_mean.values[design_ids - 1]) ** 2).sum())
    df_rep = N_DESIGNS * (REPS - 1)
    ss_dp = float(REPS * ((dp_mean.values - blk_mean.values[blk_of_dp]) ** 2).sum())
    df_dp = N_BLOCKS * (BLOCK - 1)
    ss_blk = float(PAIRS_PER_BLOCK * ((blk_mean.values - m) ** 2).sum())
    df_blk = N_BLOCKS - 1
    ms_rep, ms_dp, ms_blk = ss_rep / df_rep, ss_dp / df_dp, ss_blk / df_blk
    s2_rep = ms_rep
    s2_dp = (ms_dp - ms_rep) / REPS
    s2_blk = (ms_blk - ms_dp) / PAIRS_PER_BLOCK
    total = s2_rep + max(s2_dp, 0.0) + max(s2_blk, 0.0)

    se_naive = float(d.std(ddof=1) / np.sqrt(d.size))
    se_dp = float(dp_mean.values.std(ddof=1) / np.sqrt(N_DESIGNS))
    se_blk = float(blk_mean.values.std(ddof=1) / np.sqrt(N_BLOCKS))
    de_dp = (se_dp / se_naive) ** 2
    de_blk_dp = (se_blk / se_dp) ** 2
    de_blk = (se_blk / se_naive) ** 2
    assert abs(float(blk_mean.mean()) - m) < 1e-12                                       # A9
    return {
        "se_ladder": {
            "se_naive": se_naive, "se_designpoint": se_dp, "se_block": se_blk,
            "design_effect_dp_vs_naive": de_dp,
            "design_effect_block_vs_dp": de_blk_dp,
            "design_effect_block_vs_naive": de_blk,
            "naive_se_inflation_block": float(np.sqrt(de_blk)),
            "icc_rep_within_dp": (de_dp - 1.0) / (REPS - 1),
            "icc_dp_within_block": (de_blk_dp - 1.0) / (BLOCK - 1),
        },
        "variance_components": {
            "ms_block": ms_blk, "ms_designpoint": ms_dp, "ms_replication": ms_rep,
            "sigma2_block": s2_blk, "sigma2_designpoint": s2_dp, "sigma2_replication": s2_rep,
            "icc_designpoint_level": (max(s2_blk, 0.0) + max(s2_dp, 0.0)) / total,
            "icc_block_level": max(s2_blk, 0.0) / total,
            "note": "method-of-moments on the balanced 1024x16x30 nesting; sigma2 reported raw "
                    "(negative estimates possible) with ICCs from clipped components",
        },
    }


def profile(dp, old_tost, old_profile):
    overall = tost_clustered(dp["diff"].values, dp["block"].values, sesoi=SESOI, alpha=ALPHA,
                             expect_sizes={16})
    assert abs(overall["mean_diff"] - old_tost["mean_diff"]) < 1e-12                     # A10
    # balanced-case identity: CR1 == block-mean t
    assert abs(overall["se_cr1"] - overall["cellmeans_sensitivity"]["se"]) < 1e-15 * 1e6

    old_bins = {s["bin"]: s for s in old_profile["pfake_bins"]}
    pf_bin = pd.cut(dp["p_fake"], PFAKE_EDGES, include_lowest=True)
    assert not pf_bin.isna().any()                                                       # A13
    pfake_bins = []
    for iv, grp in dp.groupby(pf_bin, observed=True):
        s = tost_clustered(grp["diff"].values, grp["block"].values, sesoi=SESOI, alpha=ALPHA,
                           expect_sizes={8, 16})
        s["bin"] = f"({iv.left:.2f}, {iv.right:.2f}]"
        s["midpoint"] = float((iv.left + iv.right) / 2)
        s["base_reach_mean"] = float(grp["base_reach"].mean())
        old = old_bins[s["bin"]]
        assert s["n_design_points"] == old["n_clusters"]                                 # A11
        assert abs(s["mean_diff"] - old["mean_diff"]) < 1e-9                             # A11
        s["se_inflation_vs_designpoint"] = s["se_cr1"] / old["se_clustered"]
        s["old_designpoint"] = old
        pfake_bins.append(s)
    wsum = sum(s["n_design_points"] * s["mean_diff"] for s in pfake_bins)
    assert abs(wsum / N_DESIGNS - overall["mean_diff"]) < 1e-12                          # A12

    slices = {}
    for cut, old_key in ((0.15, "p_fake<=0.15"), (0.30, "p_fake<=0.3")):
        sub = dp[dp["p_fake"] <= cut]
        s = tost_clustered(sub["diff"].values, sub["block"].values, sesoi=SESOI, alpha=ALPHA,
                           expect_sizes={8, 16})
        old = old_profile["pfake_slices"][old_key]
        assert s["n_design_points"] == old["n_clusters"]                                 # A11
        assert abs(s["mean_diff"] - old["mean_diff"]) < 1e-9                             # A11
        s["base_reach_mean"] = float(sub["base_reach"].mean())
        s["old_designpoint"] = old
        slices[f"p_fake<={cut}"] = s

    lam_bin = pd.cut(dp["log10_lambda"], LAMBDA_EDGES, include_lowest=True)
    assert not lam_bin.isna().any()                                                      # A13
    lambda_bins = []
    old_lam = {s["bin"]: s for s in old_profile["lambda_bins"]}
    for iv, grp in dp.groupby(lam_bin, observed=True):
        s = tost_clustered(grp["diff"].values, grp["block"].values, sesoi=SESOI, alpha=ALPHA,
                           expect_sizes={8, 16})
        s["bin"] = f"({iv.left:.0f}, {iv.right:.0f}]"
        s["midpoint"] = float((iv.left + iv.right) / 2)
        assert s["n_design_points"] == 4096                                              # A13
        old = old_lam[s["bin"]]
        assert abs(s["mean_diff"] - old["mean_diff"]) < 1e-9                             # A11
        s["old_designpoint"] = old
        lambda_bins.append(s)
    wsum_lam = sum(s["n_design_points"] * s["mean_diff"] for s in lambda_bins)
    assert abs(wsum_lam / N_DESIGNS - overall["mean_diff"]) < 1e-12                      # A12

    # BY-FDR: primary family = all fifteen reported TOSTs; secondary = the nine p_fake bins.
    # The p_fake<=0.15 slice repeats bin 1 exactly (identical design points), which only
    # makes the correction more conservative.
    fifteen = pfake_bins + lambda_bins + list(slices.values())
    q15 = bh_fdr([s["p_tost"] for s in fifteen])
    q9 = bh_fdr([s["p_tost"] for s in pfake_bins])
    qz15 = bh_fdr([s["p_zero_two_sided"] for s in fifteen])
    for s, q, qz in zip(fifteen, q15, qz15):
        s["q_tost_fdr_by_15"] = float(q)
        s["q_zero_fdr_by_15"] = float(qz)
        s["equivalent_after_fdr"] = bool(q < ALPHA)
    for s, q in zip(pfake_bins, q9):
        s["q_tost_fdr_by_9"] = float(q)
    return overall, pfake_bins, slices, lambda_bins


def verifier_crosschecks(overall, pfake_bins, ladder):
    low = pfake_bins[0]
    values = {
        "overall_ci90_lo": overall["ci90"][0],
        "overall_ci90_hi": overall["ci90"][1],
        "smallest_passing_sesoi": overall["smallest_passing_sesoi"],
        "naive_se_inflation": ladder["se_ladder"]["naive_se_inflation_block"],
        "low_bin_ci90_lo": low["ci90"][0],
        "low_bin_ci90_hi": low["ci90"][1],
        "icc_dp_within_block": ladder["se_ladder"]["icc_dp_within_block"],
        "de_block_vs_dp": ladder["se_ladder"]["design_effect_block_vs_dp"],
    }
    checks = {}
    for name, (target, tol) in VERIFIER.items():
        v = float(values[name])
        ok = abs(v - target) <= tol
        checks[name] = {"target": target, "tolerance": tol, "value": v, "pass": bool(ok)}
        line = f"  {name:26s} target {target:+.4f} +/- {tol:.4f}   value {v:+.6f}   " \
               f"{'PASS' if ok else 'FAIL'}"
        print(line if ok else "WARNING:" + line)
    checks["all_pass"] = bool(all(c["pass"] for k, c in checks.items() if k != "all_pass"))
    return checks


def main():
    with open(OLD_TOST_JSON) as f:
        old_tost = json.load(f)
    with open(OLD_PROFILE_JSON) as f:
        old_profile = json.load(f)

    a, b, d, mb, mi = load_pairs()
    csv_check = crosscheck_csv_sources(a, b)
    dp = build_dp_table(a, d)
    structure = verify_saltelli_structure(dp)
    comps = variance_components(d, a["design_id"].values)
    overall, pfake_bins, slices, lambda_bins = profile(dp, old_tost, old_profile)

    print(f"overall: md {overall['mean_diff']:+.5f}  block CI90 [{overall['ci90'][0]:+.6f}, "
          f"{overall['ci90'][1]:+.6f}]  minSESOI {overall['smallest_passing_sesoi']:.6f}  "
          f"{'EQ' if overall['equivalent'] else 'NOT-EQ'}")
    print(f"SE ladder: naive {comps['se_ladder']['se_naive']:.3e}  "
          f"dp {comps['se_ladder']['se_designpoint']:.3e}  "
          f"block {comps['se_ladder']['se_block']:.3e}  "
          f"inflation(block vs naive) {comps['se_ladder']['naive_se_inflation_block']:.2f}x  "
          f"DE(block vs dp) {comps['se_ladder']['design_effect_block_vs_dp']:.2f}  "
          f"ICC(dp within block) {comps['se_ladder']['icc_dp_within_block']:.4f}")
    for s in pfake_bins:
        print(f"p_fake {s['bin']}: n_dp={s['n_design_points']} G={s['n_blocks']}  "
              f"md {s['mean_diff']:+.5f}  CI90 [{s['ci90'][0]:+.5f}, {s['ci90'][1]:+.5f}]  "
              f"{'EQ' if s['equivalent'] else 'NOT-EQ'} q15={s['q_tost_fdr_by_15']:.3g}  "
              f"minSESOI {s['smallest_passing_sesoi']:.5f}  base reach {s['base_reach_mean']:.3f}")
    for k, s in slices.items():
        print(f"{k}: n_dp={s['n_design_points']} G={s['n_blocks']}  md {s['mean_diff']:+.5f}  "
              f"CI90 [{s['ci90'][0]:+.5f}, {s['ci90'][1]:+.5f}]  "
              f"{'EQ' if s['equivalent'] else 'NOT-EQ'} q15={s['q_tost_fdr_by_15']:.3g}")
    for s in lambda_bins:
        print(f"log10lambda {s['bin']}: n_dp={s['n_design_points']} G={s['n_blocks']}  "
              f"md {s['mean_diff']:+.5f}  CI90 [{s['ci90'][0]:+.5f}, {s['ci90'][1]:+.5f}]  "
              f"{'EQ' if s['equivalent'] else 'NOT-EQ'} q15={s['q_tost_fdr_by_15']:.3g}")

    print("verifier cross-checks (external 2026-07-03 reproductions):")
    checks = verifier_crosschecks(overall, pfake_bins, comps)

    evidence = {
        "generated_for": "block-level recompute superseding tost_clustered/tost_profile "
                         "2026_07_03 (POSITIONING_REVIEW_2026-07-03_v2, risk 1)",
        "inputs": {
            "baseline_csv": BASE_CSV, "influencer_csv": INFL_CSV,
            "design_csv": DESIGN_CSV,
            "old_tost_json": OLD_TOST_JSON, "old_profile_json": OLD_PROFILE_JSON,
        },
        "metric": "avg_fake_cascade (fake-news reach, fraction of network)",
        "alpha": ALPHA, "sesoi": SESOI,
        "sesoi_justification": old_tost["sesoi_justification"],
        "n_pairs": N_PAIRS, "n_design_points": N_DESIGNS, "n_blocks": N_BLOCKS,
        "reps_per_design_point": REPS, "design_points_per_block": BLOCK,
        "pairs_per_block": PAIRS_PER_BLOCK,
        "mean_baseline": mb, "mean_influencer": mi,
        "csv_source_crosscheck": csv_check,
        "saltelli_structure": structure,
        "unit_of_analysis": "design-point-weighted mean difference with CR1 cluster-robust SE, "
                            "cluster = Saltelli base block ((design_id-1)//16), df = G-1; "
                            "equals the one-sample t on the 1,024 block means under balance",
        "overall": overall,
        **comps,
        "pfake_bins": pfake_bins,
        "pfake_slices": slices,
        "lambda_bins": lambda_bins,
        "fdr": {
            "method": "fdr_by",
            "family_primary": "all 15 reported TOSTs (9 p_fake bins + 4 lambda bins + "
                              "2 cumulative slices), q_tost_fdr_by_15",
            "family_secondary": "9 p_fake bins alone, q_tost_fdr_by_9",
            "note_slice_duplication": "slice p_fake<=0.15 duplicates bin (0.05,0.15] exactly; "
                                      "inclusion only adds conservatism",
        },
        "all_bins_equivalent_after_fdr": bool(all(
            s["equivalent_after_fdr"] for s in pfake_bins + lambda_bins)),
        "low_pfake_equivalent": bool(all(s["equivalent"] for s in slices.values())),
        "max_abs_bin_diff": max((abs(s["mean_diff"]), s["bin"]) for s in pfake_bins),
        "old_designpoint_overall": old_tost | {"ci90_naive": old_tost["ci90_naive"]},
        "verifier_crosschecks": checks,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tost_blocks_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=1, default=float)

    print("WROTE", os.path.join(OUT, "tost_blocks_evidence.json"))


if __name__ == "__main__":
    main()
