"""Fake-only ignition-versus-conditional-reach decomposition per prevalence bin
(switchover-direction audit).

The p_fake-binned profile (tost_blocks.py) shows hub seeding raises fake-news reach by ~+0.05
where fake news is rare and mildly suppresses it where fake news is common. The epidemic
switchover phenomenon (Odor et al. 2021, PNAS, doi:10.1073/pnas.2112607118) places the hub
advantage NEAR CRITICALITY (low reproduction number, small outbreaks) via core ignition. To
locate this model's regime map on that axis, this script streams both full 491,520,000-cascade
``cascades.arrow`` files once (no sampling; exact int64 bincount accumulators over memory-mapped
zero-copy slices) and decomposes fake reach per prevalence bin and seeding configuration into
  (i)  the fake-cascade ignition probability P = P(size >= 2 | fake), and
  (ii) the ignition-weighted conditional reach C~ = E-bar[size/300 | fake, ignited],
with the exact per-sim identity  reach = A + 1/300,  A = (sum_size_ign - n_ign)/(300 n_fake)
= P (C - 1/300), so that per bin  d_reach = dP.[(C~_B+C~_I)/2 - 1/300] + dC~.(P_B+P_I)/2
exactly (midpoint identity; the 1/300 term is the size-1 reach floor). Ignition-dominant
low-bin amplification would match Odor's mechanism (translation); conditional-reach-dominant
amplification means the direction is REVERSED (hubs saturate cascades that already ignite).
Uncertainty is block-clustered on the 1,024 Saltelli base blocks (utils.tost_clustered).
True-cascade accumulators ride along as a second reconstruction gate and symmetry check.

Usage (the main analysis environment, see requirements.txt; stream the two seeding
configurations SEQUENTIALLY, then reduce):
  python analysis/robustness/switchover_audit.py stream --sweep data/sweep_<ts>_baseline   --label baseline   --out analysis/runs/switchover_audit
  python analysis/robustness/switchover_audit.py stream --sweep data/sweep_<ts>_influencer --label influencer --out analysis/runs/switchover_audit
  python analysis/robustness/switchover_audit.py reduce --out analysis/runs/switchover_audit
``stream --max-rows N`` exists only for the smoke gate; ``reduce`` refuses partial streams.
A complete acc_switchover_{label}.npz is never re-streamed unless --force is given.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ANALYSIS)                        # analysis/ on sys.path
sys.path.insert(0, os.path.join(_ANALYSIS, "lib"))   # analysis/lib on sys.path
import data_io  # noqa: E402
from data_io import PFAKE_EDGES, N_PAIRS, N_DESIGNS, N_BLOCKS, BLOCK, REPS  # noqa: E402
from utils import tost_clustered  # noqa: E402

BASE_CSV = data_io.BASELINE_SVD_CSV
INFL_CSV = data_io.INFLUENCER_SVD_CSV

N_SIMS = N_PAIRS
SIM_LO = 1
N_PER_SIM = data_io.N_CASC_PER_SIM
SIZE_MAX = data_io.N_NODES
REACH_FLOOR = 1.0 / SIZE_MAX          # reach of a size-1 (non-igniting) cascade
CHUNK = 5_000_000
ALL_CELLS = set(range(1, BLOCK + 1))  # complete-pair subsets may thin block cells to any size

ACC_KEYS = ("n_fake", "n_fake_ign", "sum_size_fake", "sum_size_fake_ign",
            "n_true", "n_true_ign", "sum_size_true", "sum_size_true_ign")


def cmd_stream(args):
    tag = "" if args.max_rows is None else "_smoke"
    out = os.path.join(args.out, f"acc_switchover_{args.label}{tag}.npz")
    if os.path.exists(out) and not args.force:
        acc = np.load(out)
        if int(acc["n_rows_streamed"]) == N_SIMS * N_PER_SIM:
            print(f"{out} already complete; use --force to re-stream", flush=True)
            return
        print(f"{out} exists but is partial; re-streaming", flush=True)

    batch, n_rows = data_io.open_arrow(os.path.join(args.sweep, "cascades.arrow"))
    if args.max_rows is not None:
        n_rows = min(n_rows, args.max_rows)

    acc = {k: np.zeros(N_SIMS, np.int64) for k in ACC_KEYS}
    t0 = time.time()
    for off, sl in data_io.stream_chunks(batch, n_rows, CHUNK, label=args.label, print_every=10):
        sid = sl.column("global_sim_id").to_numpy(zero_copy_only=True)
        size = sl.column("cascade_size").to_numpy(zero_copy_only=True)
        fake = sl.column("is_fake").to_numpy(zero_copy_only=False)  # bit-packed bool: copies
        assert sid.min() >= SIM_LO and sid.max() < SIM_LO + N_SIMS
        assert size.min() >= 1 and size.max() <= SIZE_MAX

        key = sid - SIM_LO
        ign = size >= 2
        for flag, pre in ((fake, "fake"), (~fake, "true")):
            k_all, k_ign = key[flag], key[flag & ign]
            acc[f"n_{pre}"] += np.bincount(k_all, minlength=N_SIMS)
            acc[f"n_{pre}_ign"] += np.bincount(k_ign, minlength=N_SIMS)
            acc[f"sum_size_{pre}"] += np.bincount(
                k_all, weights=size[flag], minlength=N_SIMS).astype(np.int64)
            acc[f"sum_size_{pre}_ign"] += np.bincount(
                k_ign, weights=size[flag & ign], minlength=N_SIMS).astype(np.int64)

    if args.max_rows is None:
        assert n_rows == N_SIMS * N_PER_SIM, n_rows                                 # S1 (stream)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(out, n_rows_streamed=n_rows, sweep=args.sweep, label=args.label, **acc)
    print(f"WROTE {out} ({time.time() - t0:.0f}s)", flush=True)


def _load_acc(out_dir, label):
    path = os.path.join(out_dir, f"acc_switchover_{label}.npz")
    acc = dict(np.load(path, allow_pickle=False).items())
    if int(acc["n_rows_streamed"]) != N_SIMS * N_PER_SIM:
        sys.exit(f"ERROR: {path} is a partial (smoke) stream; refusing to reduce")     # S1
    return acc


def _per_dp(acc, csv_path, mean_gate):
    """Per-sim quantities, reconstruction gates against simulations.csv, collapse to the
    16,384 design points. Returns (dp frame, gate dict)."""
    cols = ["global_sim_id", "design_id", "p_fake", "avg_fake_cascade", "avg_true_cascade"]
    csv = pd.read_csv(csv_path, usecols=cols).sort_values("global_sim_id").reset_index(drop=True)
    assert len(csv) == N_SIMS
    assert np.array_equal(csv["global_sim_id"].values, np.arange(1, N_SIMS + 1))       # S4

    nf = acc["n_fake"].astype(np.float64)
    nt = acc["n_true"].astype(np.float64)
    assert (acc["n_fake"] + acc["n_true"] == N_PER_SIM).all()                          # S2
    assert acc["n_fake"].min() > 0 and acc["n_true"].min() > 0                         # S3

    rec_fake = (acc["sum_size_fake"] / SIZE_MAX) / nf
    rec_true = (acc["sum_size_true"] / SIZE_MAX) / nt
    err_f = float(np.abs(rec_fake - csv["avg_fake_cascade"].values).max())
    err_t = float(np.abs(rec_true - csv["avg_true_cascade"].values).max())
    if err_f >= 1e-9 or err_t >= 1e-9:                                                 # S5
        print(f"RECONSTRUCTION FAILURE: max|rec-csv| fake {err_f:.3e} true {err_t:.3e}")
        print("diagnostic (first mismatching sim, fake):")
        i = int(np.abs(rec_fake - csv["avg_fake_cascade"].values).argmax())
        print(f"  sim {i+1}: n_fake {acc['n_fake'][i]} sum_size {acc['sum_size_fake'][i]} "
              f"rec {rec_fake[i]:.9f} csv {csv['avg_fake_cascade'].values[i]:.9f}")
        sys.exit(2)
    m = float(rec_fake.mean())
    if mean_gate is not None:  # published-mean gate; only the canonical sweep can satisfy it
        assert abs(m - mean_gate) < 5e-5, (m, mean_gate)                               # S5

    P = acc["n_fake_ign"] / nf
    A = (acc["sum_size_fake_ign"] - acc["n_fake_ign"]) / (SIZE_MAX * nf)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(acc["n_fake_ign"] > 0,
                     (acc["sum_size_fake_ign"] / SIZE_MAX) / acc["n_fake_ign"], np.nan)
    assert float(np.abs((A + REACH_FLOOR) - rec_fake).max()) < 1e-12   # per-sim identity
    P_t = acc["n_true_ign"] / nt
    A_t = (acc["sum_size_true_ign"] - acc["n_true_ign"]) / (SIZE_MAX * nt)

    sim = pd.DataFrame({
        "design_id": csv["design_id"].values, "p_fake": csv["p_fake"].values,
        "P": P, "A": A, "C": C, "reach": rec_fake,
        "P_true": P_t, "A_true": A_t, "reach_true": rec_true,
    })
    g = sim.groupby("design_id")
    assert (g.size() == REPS).all()
    dp = g.agg(p_fake=("p_fake", "first"), P=("P", "mean"), A=("A", "mean"),
               reach=("reach", "mean"), P_true=("P_true", "mean"), A_true=("A_true", "mean"),
               reach_true=("reach_true", "mean"))
    dp["block"] = (dp.index.values - 1) // BLOCK
    assert len(dp) == N_DESIGNS and dp["block"].nunique() == N_BLOCKS
    gates = {
        "n_rows_streamed": int(acc["n_rows_streamed"]),
        "all_sims_have_1000_cascades": True,
        "min_n_fake": int(acc["n_fake"].min()), "min_n_true": int(acc["n_true"].min()),
        "n_sims_zero_fake_ignited": int((acc["n_fake_ign"] == 0).sum()),
        "n_sims_zero_true_ignited": int((acc["n_true_ign"] == 0).sum()),
        "reconstruction_max_abs_err_fake": err_f,
        "reconstruction_max_abs_err_true": err_t,
        "mean_fake_reach": m,
    }
    return sim, dp, gates


def _bin_arm(dp_arm, mask):
    sub = dp_arm[mask]
    P_bar = float(sub["P"].mean())
    A_bar = float(sub["A"].mean())
    reach = float(sub["reach"].mean())
    P_true = float(sub["P_true"].mean())
    assert abs((A_bar + REACH_FLOOR) - reach) < 1e-12
    # C_tilde is undefined when nothing ignites in the bin; possible only for the thin
    # bins of scaled runs (every canonical bin ignites), so report NaN instead of failing
    return {
        "P_ign_fake": P_bar,
        "C_tilde_fake": A_bar / P_bar + REACH_FLOOR if P_bar > 0 else float("nan"),
        "reach_fake": reach,
        "P_ign_true": P_true,
        "C_tilde_true": (float(sub["A_true"].mean()) / P_true + REACH_FLOOR
                         if P_true > 0 else float("nan")),
        "reach_true": float(sub["reach_true"].mean()),
    }


def _decompose(base, infl):
    d_reach = infl["reach_fake"] - base["reach_fake"]
    dP = infl["P_ign_fake"] - base["P_ign_fake"]
    dC = infl["C_tilde_fake"] - base["C_tilde_fake"]
    mid_C = (base["C_tilde_fake"] + infl["C_tilde_fake"]) / 2.0
    mid_P = (base["P_ign_fake"] + infl["P_ign_fake"]) / 2.0
    ign = dP * (mid_C - REACH_FLOOR)
    cond = dC * mid_P
    resid = d_reach - ign - cond
    # the identity is exact wherever C_tilde is defined; NaN components (empty scaled
    # bins) make the residual NaN, which is reported rather than asserted
    assert not np.isfinite(resid) or abs(resid) < 1e-12, resid                         # S7
    return {
        "d_reach": d_reach, "delta_P": dP, "delta_C_tilde": dC,
        "ignition_component": ign, "conditional_component": cond,
        "ignition_share": ign / d_reach if d_reach != 0 else float("nan"),
        "conditional_share": cond / d_reach if d_reach != 0 else float("nan"),
        "residual_abs": abs(resid),
    }


def _ci(entry):
    return {k: entry[k] for k in ("mean_diff", "se_cr1", "ci90", "ci95",
                                  "p_zero_two_sided", "n_design_points", "n_blocks")}


def _ci_or_stub(values, blocks):
    """_ci(tost_clustered(...)), or a NaN-interval stub when fewer than two Saltelli blocks
    back the selection. Only scaled runs can be that thin; the canonical design backs every
    bin with >= 200 blocks."""
    values = np.asarray(values, dtype=float)
    n_blocks = int(pd.unique(np.asarray(blocks)).size)
    if data_io.CANONICAL or n_blocks >= 2:
        return _ci(tost_clustered(values, blocks, expect_sizes=ALL_CELLS))
    nan = float("nan")
    return {"mean_diff": float(values.mean()) if len(values) else nan, "se_cr1": nan,
            "ci90": [nan, nan], "ci95": [nan, nan], "p_zero_two_sided": 1.0,
            "n_design_points": int(len(values)), "n_blocks": n_blocks,
            "skipped_tost": "fewer than 2 Saltelli blocks at this scale"}


def cmd_reduce(args):
    t0 = time.time()
    acc_b = _load_acc(args.out, "baseline")
    acc_i = _load_acc(args.out, "influencer")
    sim_b, dp_b, gates_b = _per_dp(acc_b, BASE_CSV,
                                   data_io.PUBLISHED_MEANS[0] if data_io.CANONICAL else None)
    sim_i, dp_i, gates_i = _per_dp(acc_i, INFL_CSV,
                                   data_io.PUBLISHED_MEANS[1] if data_io.CANONICAL else None)
    assert np.array_equal(dp_b.index.values, dp_i.index.values)
    assert np.allclose(dp_b["p_fake"].values, dp_i["p_fake"].values)

    profile_json = os.path.join(data_io.RUNS_DIR, "tost_profile", "profile_evidence.json")
    if os.path.exists(profile_json):
        with open(profile_json) as f:
            old_profile = json.load(f)
        old_bins = {s["bin"]: s for s in old_profile["pfake_bins"]}
    else:
        # Scaled runs carry no superseded full-scale anchor analysis; the S6 anchor
        # cross-checks are vacuous there and are skipped rather than failed.
        old_profile, old_bins = None, {}
        print("tost_profile anchor absent -- skipping S6 anchor cross-checks", flush=True)

    # complete-pair rep-level conditional-reach differences -> per-dp means
    both = np.isfinite(sim_b["C"].values) & np.isfinite(sim_i["C"].values)
    dC_sim = pd.Series(sim_i["C"].values - sim_b["C"].values).where(both)
    dC_dp = dC_sim.groupby(sim_b["design_id"].values).mean()  # NaN if no valid rep in dp
    n_pairs_dropped = int(N_PAIRS - both.sum())

    pf_bin = pd.cut(dp_b["p_fake"], PFAKE_EDGES, include_lowest=True)
    assert not pf_bin.isna().any()
    blocks = dp_b["block"].values

    def bin_entry(mask, label, old_md):
        base, infl = _bin_arm(dp_b, mask), _bin_arm(dp_i, mask)
        dec = _decompose(base, infl)
        if old_md is not None:
            assert abs(dec["d_reach"] - old_md) < 1e-9, (label, dec["d_reach"], old_md)  # S6
        dP_dp = (dp_i["P"] - dp_b["P"]).values[mask]
        dA_dp = (dp_i["A"] - dp_b["A"]).values[mask]
        blk = blocks[mask]
        dc = dC_dp.values[mask.values if hasattr(mask, "values") else mask]
        dc_ok = np.isfinite(dc)
        entry = {
            "bin": label,
            "n_design_points": int(mask.sum()),
            "n_blocks": int(pd.unique(blk).size),
            "baseline": base, "influencer": infl,
            "decomposition": dec,
            "delta_ci": {
                "P_ign_fake": _ci_or_stub(dP_dp, blk),
                "A_fake": _ci_or_stub(dA_dp, blk),
                "C_completepairs": _ci_or_stub(dc[dc_ok], blk[dc_ok]) | {
                    "n_dp_used": int(dc_ok.sum()),
                    "n_dp_dropped": int((~dc_ok).sum())},
                "P_ign_true": _ci_or_stub((dp_i["P_true"] - dp_b["P_true"]).values[mask], blk),
            },
        }
        return entry

    overall = bin_entry(np.ones(N_DESIGNS, dtype=bool), "overall",
                        old_profile["overall"]["mean_diff"] if old_profile else None)
    pfake_bins = []
    for iv in pf_bin.cat.categories:
        label = f"({iv.left:.2f}, {iv.right:.2f}]"
        mask = (pf_bin == iv).values
        old_md = old_bins[label]["mean_diff"] if label in old_bins else None
        pfake_bins.append(bin_entry(mask, label, old_md))

    low = pfake_bins[0]
    dec = low["decomposition"]
    ign_ci = low["delta_ci"]["P_ign_fake"]["ci90"]
    cond_ci = low["delta_ci"]["C_completepairs"]["ci90"]
    dominant = "conditional" if abs(dec["conditional_share"]) > abs(dec["ignition_share"]) \
        else "ignition"
    ci_backed = (dominant == "conditional" and (cond_ci[0] > 0 or cond_ci[1] < 0)
                 and abs(dec["conditional_share"]) > 0.5) or \
                (dominant == "ignition" and (ign_ci[0] > 0 or ign_ci[1] < 0)
                 and abs(dec["ignition_share"]) > 0.5)
    direction = {
        "low_bin": {
            "bin": low["bin"], "d_reach": dec["d_reach"],
            "delta_P": dec["delta_P"], "delta_C_tilde": dec["delta_C_tilde"],
            "ignition_share": dec["ignition_share"],
            "conditional_share": dec["conditional_share"],
            "delta_P_ci90": ign_ci, "delta_C_ci90": cond_ci,
            "dominant_axis": dominant, "dominant_axis_ci_backed": bool(ci_backed),
        },
        "per_bin_signs": [{
            "bin": s["bin"], "midpoint_baseline_reach": s["baseline"]["reach_fake"],
            "delta_P": s["decomposition"]["delta_P"],
            "delta_C_tilde": s["decomposition"]["delta_C_tilde"],
            "d_reach": s["decomposition"]["d_reach"],
        } for s in pfake_bins],
        "bins_delta_P_positive": [s["bin"] for s in pfake_bins
                                  if s["decomposition"]["delta_P"] > 0],
        "bins_delta_C_positive": [s["bin"] for s in pfake_bins
                                  if s["decomposition"]["delta_C_tilde"] > 0],
        "interpretation_rule": "Odor et al. 2021 place the hub-seeding advantage near "
                               "criticality (small outbreaks) via core IGNITION. If the "
                               "low-prevalence amplification here is ignition-dominant "
                               "(delta_P > 0 carrying most of d_reach), the finding "
                               "translates Odor's switchover; if it is conditional-reach-"
                               "dominant while ignition stays suppressed (delta_P < 0), the "
                               "regime map is the MIRROR IMAGE: hubs gain where cascades "
                               "already spread widest (supercritical/saturating regime).",
    }

    evidence = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "generated_for": "switchover-direction audit "
                         "(POSITIONING_REVIEW_2026-07-03_v2 section 4, item 07-07)",
        "inputs": {
            "baseline_arrow": str(acc_b["sweep"]), "influencer_arrow": str(acc_i["sweep"]),
            "baseline_csv": BASE_CSV, "influencer_csv": INFL_CSV,
            "acc_files": [os.path.join(args.out, f"acc_switchover_{l}.npz")
                          for l in ("baseline", "influencer")],
        },
        "config": {"n_sims": N_SIMS, "n_per_sim": N_PER_SIM, "chunk": CHUNK,
                   "size_max": SIZE_MAX, "reach_floor": REACH_FLOOR,
                   "pfake_edges": PFAKE_EDGES.tolist(),
                   "block_definition": f"(design_id-1)//{BLOCK}, {N_BLOCKS} Saltelli base blocks"},
        "identity": "per sim: reach = A + 1/300 with A = (sum_size_ign - n_ign)/(300 n_fake); "
                    "per bin: d_reach = dP.[(C~_B+C~_I)/2 - 1/300] + dC~.(P_B+P_I)/2 with "
                    "C~ = A_bar/P_bar + 1/300 (exact, residual asserted < 1e-12)",
        "stream_gates": {"baseline": gates_b, "influencer": gates_i},
        "complete_pairs": {"n_pairs_dropped_rep_level": n_pairs_dropped,
                           "note": "rep-level pairs where either arm had no igniting fake "
                                   "cascade; dC averaged over remaining reps per design point"},
        "overall": overall,
        "pfake_bins": pfake_bins,
        "direction_relation": direction,
        "runtime_seconds": {"reduce": time.time() - t0},
    }
    out_json = os.path.join(args.out, "switchover_audit_evidence.json")
    with open(out_json, "w") as f:
        json.dump(evidence, f, indent=1, default=float)

    print(f"overall: d_reach {dec_fmt(overall)}")
    for s in pfake_bins:
        print(f"p_fake {s['bin']}: {dec_fmt(s)}")
    lb = direction["low_bin"]
    print(f"LOW BIN {lb['bin']}: d_reach {lb['d_reach']:+.5f} = ignition {lb['ignition_share']:+.1%} "
          f"+ conditional {lb['conditional_share']:+.1%}  ->  dominant axis: "
          f"{lb['dominant_axis']} (CI-backed: {lb['dominant_axis_ci_backed']})")
    print("WROTE", out_json)


def dec_fmt(s):
    d = s["decomposition"]
    b, i = s["baseline"], s["influencer"]
    return (f"P {b['P_ign_fake']:.4f}->{i['P_ign_fake']:.4f} (dP {d['delta_P']:+.4f})  "
            f"C~ {b['C_tilde_fake']:.4f}->{i['C_tilde_fake']:.4f} (dC {d['delta_C_tilde']:+.4f})  "
            f"d_reach {d['d_reach']:+.5f} = ign {d['ignition_share']:+.1%} "
            f"+ cond {d['conditional_share']:+.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stream")
    st.add_argument("--sweep", required=True)
    st.add_argument("--label", required=True, choices=("baseline", "influencer"))
    st.add_argument("--out", required=True)
    st.add_argument("--max-rows", type=int, default=None)
    st.add_argument("--force", action="store_true")
    st.set_defaults(fn=cmd_stream)
    rd = sub.add_parser("reduce")
    rd.add_argument("--out", required=True)
    rd.set_defaults(fn=cmd_reduce)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
