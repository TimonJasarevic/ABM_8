"""Build the evidence JSONs for the seeding-comparison analysis: stream + reduce in one script.

The raw sweeps are large (nodes.arrow ~17 GB, cascades.arrow ~37 GB per seeding
configuration), so the script has two tiers.

STREAM (bounded memory; run once per raw sweep) memory-maps the arrows and walks them in
simulation-aligned chunks into a small per-simulation cache (~80 MB of acc_*.npz files).
Each simulation is a contiguous 300-node block, guaranteed by the Julia ``vcat(node_dfs...)``
write order, so the per-chunk groupby reductions equal the whole-array reductions
bit-for-bit. Peak RAM stays a few GB.

    python analysis/build_evidence.py stream            --sweep <sweep_dir> --label baseline   --out <cache>
    python analysis/build_evidence.py stream-cascades   --sweep <sweep_dir> --label baseline   --out <cache>
    python analysis/build_evidence.py sample-cascade-sizes                                     --out <cache>
    python analysis/build_evidence.py stream-influencer --sweep <influencer_sweep>             --out <cache>

REDUCE (minutes, < 2 GB RAM) turns the cache plus the small per-sim simulations.arrow into
the three evidence JSONs that make_figures.py and the manuscript tables read. Three
subcommands write one JSON each:

    python analysis/build_evidence.py eda         -> analysis/eda_evidence.json
    python analysis/build_evidence.py compare     -> analysis/compare_evidence.json
    python analysis/build_evidence.py influencer  -> analysis/influencer_node_evidence.json

``compare`` pairs a baseline sweep (uniform-random cascade seeding) against an influencer
sweep (seeding from the highest-degree hubs). The two sweeps share the same Saltelli design
and per-simulation seeds, so simulations are matched by global_sim_id and the only
difference is the cascade seed pool. Sweep directories and the cache default to the
published sweeps and analysis/runs/nodecache_* (override with --baseline / --influencer /
--cache; the stream subcommands take an explicit --out so a partial stream can never
silently overwrite the verified cache). The paired-statistics conventions live in
lib/paired_stats.py. The superseded whole-arrow reducer (~55-60 GiB peak, the reason this
step once needed an HPC node) is retained in the development archive (not distributed).
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))
import data_io
from data_io import open_arrow, stream_chunks
from paired_stats import g3, _clustered_family, paired_cmp, paired_inf
from utils import sen_welfare, belief_dispersion, dip_test, bimodality_coefficient, \
    flexible_calibration, bh_fdr, tost_paired

OUT_DIR = data_io.EVIDENCE_DIR                 # analysis/ canonically; the run root for scaled runs
DEFAULT_BASELINE = data_io.BASELINE_SWEEP      # uniform-random seeding (run_experiment_baseline.jl)
DEFAULT_INFLUENCER = data_io.INFLUENCER_SWEEP  # top-degree hub seeding

N_PER_SIM = data_io.N_NODES                    # nodes per simulation (Julia N_AGENTS)
N_RECORDED = data_io.N_CASC_PER_SIM            # recorded cascades per simulation
N_SIMS = data_io.N_PAIRS
NODE_COLS = data_io.NODE_COLS                  # nodes.arrow projection
PERSIM_METRICS = ["sen_welfare", "discard_rate", "psi", "bc", "dip_p", "mean_true_belief"]
PAIRM = ["payoff", "agent_veracity", "total_shares", "node_veracity_diff", "participation",
         "truth_share_ratio"]


# ------------------------------------------------------------------ shared helpers
def agent_veracity(rep, best, worst):
    """Degree-fair normalised payoff in [-1, 1]: losses scaled by the worst attainable payoff,
    gains by the best attainable, 0 when there was no opportunity."""
    return np.where((rep < 0) & (worst != 0), -(rep / np.where(worst == 0, 1, worst)),
                    np.where((rep > 0) & (best != 0), rep / np.where(best == 0, 1, best), 0.0))


def describe_dict(s):
    """Round-4 dict of a pandas ``describe()`` with the house percentiles."""
    return {k: round(float(v), 4)
            for k, v in s.describe(percentiles=[.05, .25, .5, .75, .95]).to_dict().items()}


def hist_dict(x, bins):
    """Histogram block: round-3 edges, integer counts."""
    c, e = np.histogram(x, bins=bins)
    return {"edges": [round(float(v), 3) for v in e], "counts": [int(v) for v in c]}


# ================================================================== reduce tier
def cmd_eda(baseline, cache):
    D = Path(baseline)
    sim = feather.read_feather(D / "simulations.arrow")
    sim = sim.drop(columns=["sen_welfare"], errors="ignore")  # simulations.arrow carries a precomputed sen_welfare; dropped so that eda recomputes it from the cache
    out = {"n_sims": int(len(sim))}

    # Node-derived stats from the streamed per-sim cache (byte-identical to the archived
    # whole-arrow reduction: the same pandas/utils code produced the cache, only chunked).
    c = np.load(os.path.join(cache, "acc_nodes_baseline.npz"), allow_pickle=True)
    p = json.loads(str(c["eda_pooled_json"]))
    out["n_nodes"] = p["n_nodes"]
    out["agent_veracity_desc"] = p["agent_veracity_desc"]
    out["agent_veracity_hist"] = p["agent_veracity_hist"]
    out["frac_zero_opportunity"] = p["frac_zero_opportunity"]
    out["frac_zero_payoff"] = p["frac_zero_payoff"]
    out["frac_av_eq_0"] = p["frac_av_eq_0"]
    sim = sim.merge(pd.DataFrame({"global_sim_id": c["gsid"], **{k: c[k] for k in PERSIM_METRICS}}),
                    on="global_sim_id")

    out["sen_welfare_desc"] = describe_dict(sim["sen_welfare"]); out["sen_welfare_hist"] = hist_dict(sim["sen_welfare"].values, 10)
    out["discard_rate_desc"] = describe_dict(sim["discard_rate"]); out["discard_rate_hist"] = hist_dict(sim["discard_rate"].values, 10)
    out["psi_desc"] = describe_dict(sim["psi"]); out["bc_desc"] = describe_dict(sim["bc"])
    out["psi_hist"] = hist_dict(sim["psi"].values, 10); out["bc_hist"] = hist_dict(sim["bc"].dropna().values, 10)
    out["frac_bc_gt_0556"] = round(float((sim["bc"] > 5/9).mean()), 4)
    out["frac_dip_p_lt_0p05"] = round(float((sim["dip_p"] < 0.05).mean()), 4)

    for col in ["avg_true_cascade", "avg_fake_cascade", "veracity_differential", "avg_verify_rate", "avg_payoff"]:
        out[col + "_desc"] = describe_dict(sim[col])
    out["true_cascade_hist"] = hist_dict(sim["avg_true_cascade"].values, 10)
    out["fake_cascade_hist"] = hist_dict(sim["avg_fake_cascade"].values, 10)

    m = sim[["global_sim_id", "mean_true_belief", "p_fake"]].rename(columns={"mean_true_belief": "true_prev"})
    m["p_true"] = 1 - m["p_fake"]; m["bin"] = pd.cut(m["p_true"], bins=np.linspace(0, 1, 11))
    cal = m.groupby("bin", observed=True).agg(p_true=("p_true", "mean"), belief=("true_prev", "mean"), n=("true_prev", "size")).reset_index(drop=True)
    out["calibration"] = [{k: (round(float(v), 4) if k != "n" else int(v)) for k, v in r.items()} for r in cal.to_dict(orient="records")]
    out["calibration_corr"] = round(float(m["p_true"].corr(m["true_prev"])), 4)
    b1, b0 = np.polyfit(m["p_true"], m["true_prev"], 1); out["calibration_slope"] = round(float(b1), 4); out["calibration_intercept"] = round(float(b0), 4)
    out["calibration_flexible"] = flexible_calibration(m["p_true"].values, m["true_prev"].values)

    params = data_io.FACTORS
    outc = ["veracity_differential", "sen_welfare", "avg_verify_rate", "avg_true_cascade", "avg_fake_cascade", "discard_rate", "psi", "bc", "avg_payoff"]
    out["pearson_param_outcome"] = {o: {p: round(float(sim[p].corr(sim[o])), 3) for p in params} for o in outc}
    out["corr_verify_vs_vd"] = round(float(sim["avg_verify_rate"].corr(sim["veracity_differential"])), 3)
    out["corr_verify_vs_discard"] = round(float(sim["avg_verify_rate"].corr(sim["discard_rate"])), 3)
    out["corr_vd_vs_sw"] = round(float(sim["veracity_differential"].corr(sim["sen_welfare"])), 3)
    out["corr_bcpsi_vs_vd"] = round(float((sim["bc"] * sim["psi"]).corr(sim["veracity_differential"])), 3)

    sim["regime"] = pd.qcut(sim["sen_welfare"], 4, labels=[0, 1, 2, 3])
    reg = sim.groupby("regime", observed=True)[["tpr", "fpr", "v_cost", "loss", "p_fake", "rationality", "loss_aversion", "veracity_differential", "avg_verify_rate", "discard_rate", "sen_welfare"]].mean()
    out["regime_means"] = [{k: round(float(v), 4) for k, v in r.items()} for r in reg.reset_index(drop=True).to_dict(orient="records")]

    sim["pfb"] = pd.cut(sim["p_fake"], bins=7)
    out["verify_by_pfake_bin"] = [{"bin": str(k), "verify": round(float(v), 4)} for k, v in sim.groupby("pfb", observed=True)["avg_verify_rate"].mean().items()]

    with open(os.path.join(OUT_DIR, "eda_evidence.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE eda_evidence.json with", len(out), "keys")


def cmd_compare(baseline, influencer, cache):
    A_DIR, B_DIR = baseline, influencer
    out = {}

    # ---------- sim-level (cheap) ----------
    cols = ["global_sim_id", "design_id", *data_io.FACTORS,
            "avg_payoff", "avg_fake_cascade", "avg_true_cascade", "veracity_differential", "avg_verify_rate"]
    simA = feather.read_feather(f"{A_DIR}/simulations.arrow", columns=cols).sort_values("global_sim_id").reset_index(drop=True)
    simB = feather.read_feather(f"{B_DIR}/simulations.arrow", columns=cols).sort_values("global_sim_id").reset_index(drop=True)
    assert (simA.global_sim_id.values == simB.global_sim_id.values).all()
    assert (simA.design_id.values == simB.design_id.values).all()
    # one-time check of the gsid layout that the node/cascade block vectors below rely on
    assert np.array_equal(simA.design_id.values, (simA.global_sim_id.values - 1) // data_io.REPS + 1)
    blocks_sim = (simA.design_id.values - 1) // data_io.BLOCK
    out["n_pairs"] = int(len(simA))
    out["pairing_validity_max_abs_param_diff"] = {
        p: float(np.abs(simA[p].values - simB[p].values).max()) for p in data_io.FACTORS}
    out["sim_level"] = {}
    for m in ["avg_payoff", "avg_fake_cascade", "avg_true_cascade", "veracity_differential", "avg_verify_rate"]:
        out["sim_level"].update(paired_cmp(simA[m].values, simB[m].values, m, blocks=blocks_sim))

    # ---------- node-derived per-sim (from the streamed cache) ----------
    def node_per_sim(label):
        c = np.load(os.path.join(cache, f"acc_nodes_{label}.npz"), allow_pickle=True)
        return pd.DataFrame({m: c[m] for m in PERSIM_METRICS},
                            index=pd.Index(c["gsid"], name="global_sim_id")).sort_index()
    nodeA = node_per_sim("baseline"); gc.collect()
    nodeB = node_per_sim("influencer"); gc.collect()
    assert (nodeA.index.values == nodeB.index.values).all()
    # block per gsid: (gsid-1)//480 == ((gsid-1)//REPS)//BLOCK, valid by the layout assert above
    blocks_node = (nodeA.index.values - 1) // data_io.PAIRS_PER_BLOCK
    out["node_level"] = {}
    for m in PERSIM_METRICS:
        out["node_level"].update(paired_cmp(nodeA[m].values, nodeB[m].values, m, blocks=blocks_node))

    # calibration (believed P(true) vs actual = 1 - p_fake), per sweep
    def calib(node_tb, sim):
        p_true = 1 - sim["p_fake"].values; bel = node_tb["mean_true_belief"].values
        s, i = np.polyfit(p_true, bel, 1)
        fc = flexible_calibration(p_true, bel)
        return {"slope": round(float(s), 4), "intercept": round(float(i), 4),
                "corr": round(float(np.corrcoef(p_true, bel)[0, 1]), 4), "flexible": fc}
    out["calibration"] = {"baseline": calib(nodeA, simA), "influencer": calib(nodeB, simB)}

    # ---------- cascade-level per-sim means (from the streamed cache) ----------
    # structural virality is present only when both configurations' streams carried the column
    casc_metrics = [str(x) for x in np.load(os.path.join(cache, "acc_casc_baseline.npz"),
                                            allow_pickle=True)["metrics"]]

    def casc_per_sim(label):
        c = np.load(os.path.join(cache, f"acc_casc_{label}.npz"), allow_pickle=True)
        cnt = c["count"]
        res = pd.DataFrame({m: c[f"sum_{m}"] / cnt for m in casc_metrics},
                           index=pd.Index(np.arange(len(cnt)) + 1, name="global_sim_id"))
        samp = np.load(os.path.join(cache, "cascade_size_samples.npz"))[f"cascade_size_sample_{label}"]
        return res, np.asarray(samp, float)
    cascA, csA = casc_per_sim("baseline"); gc.collect()
    cascB, csB = casc_per_sim("influencer"); gc.collect()
    assert (cascA.index.values == cascB.index.values).all()
    blocks_casc = (cascA.index.values - 1) // data_io.PAIRS_PER_BLOCK
    out["cascade_level"] = {}
    for m in casc_metrics:
        out["cascade_level"].update(paired_cmp(cascA[m].values, cascB[m].values, "mean_" + m, blocks=blocks_casc))
    ks = stats.ks_2samp(csA, csB)
    wdist = float(stats.wasserstein_distance(csA, csB))
    edist = float(stats.energy_distance(csA, csB))
    out["cascade_size_KS"] = {"statistic": round(float(ks.statistic), 4), "p": g3(ks.pvalue),
                              "wasserstein_distance": round(wdist, 4), "energy_distance": round(edist, 4),
                              "mean_baseline": round(float(csA.mean()), 3), "mean_influencer": round(float(csB.mean()), 3),
                              "median_baseline": round(float(np.median(csA)), 1), "median_influencer": round(float(np.median(csB)), 1)}

    # ---- multiple-comparison control across all paired tests (Benjamini-Yekutieli) ----
    _keys, _ps = [], []
    for _section in ("sim_level", "node_level", "cascade_level"):
        for _m, _d in out.get(_section, {}).items():
            if isinstance(_d, dict) and "p_ttest" in _d:
                _keys.append((_section, _m)); _ps.append(_d["p_ttest"])
    if _ps:
        _q = bh_fdr(_ps, method="fdr_by")
        for (_section, _m), _qi in zip(_keys, _q):
            out[_section][_m]["p_ttest_fdr_by"] = g3(_qi)

    # ---- the same family on CR1 block-clustered t-tests (the sweep's exchangeable unit is the
    # Saltelli base block, so these q values are the pseudoreplication-honest family verdicts).
    # Full-precision p feeds the FDR; g3 rounding happens only after. ----
    _keys2, _ps2 = [], []
    for _section in ("sim_level", "node_level", "cascade_level"):
        for _m, _d in out.get(_section, {}).items():
            if isinstance(_d, dict) and "p_ttest_clustered" in _d:
                _keys2.append((_section, _m)); _ps2.append(_d["p_ttest_clustered"])
    if _ps2:
        _q2 = bh_fdr(_ps2, method="fdr_by")
        for (_section, _m), _qi in zip(_keys2, _q2):
            out[_section][_m]["q_fdr_by_clustered"] = g3(float(_qi))
            out[_section][_m]["p_ttest_clustered"] = g3(out[_section][_m]["p_ttest_clustered"])

    # ---- equivalence test (TOST) for fake-news reach; the SESOI band is data_io.SESOI ----
    # Record any failure instead of silently dropping the result (matches the mixed_model_payoff
    # pattern); the success path is unchanged.
    try:
        out["sim_level"]["avg_fake_cascade"]["tost_equivalence"] = tost_paired(
            simB["avg_fake_cascade"].values, simA["avg_fake_cascade"].values,
            -data_io.SESOI, data_io.SESOI)
    except Exception as e:
        out["sim_level"]["avg_fake_cascade"]["tost_equivalence"] = {"error": repr(e)}
        print("WARN: tost_equivalence failed:", repr(e))

    with open(os.path.join(OUT_DIR, "compare_evidence.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE compare_evidence.json  (n_pairs=%d)" % out["n_pairs"])
    print(json.dumps(out, indent=1))


def cmd_influencer(cache):
    """Reduce acc_infl.npz -> influencer_node_evidence.json. The descriptive blocks were computed
    in-stream via the exact original operations; here they are only reloaded, and paired_inf is
    rerun on the per-sim inf/non means."""
    out = {}
    c = np.load(os.path.join(cache, "acc_infl.npz"), allow_pickle=True)
    desc = json.loads(str(c["desc_json"]))
    out["influencer_id_check"] = desc["influencer_id_check"]
    out["influencer_mean_seed_degree"] = desc["influencer_mean_seed_degree"]
    out["group_counts"] = desc["group_counts"]
    out["pooled"] = desc["pooled"]
    # the stream writes the per-sim means gsid-dense (row i = gsid i+1); assert the layout the
    # block vector relies on against a table that carries both id columns
    n = len(c["inf_mean_payoff"])
    assert n == data_io.N_PAIRS, n
    _ids = pd.read_csv(data_io.INFLUENCER_SVD_CSV, usecols=["global_sim_id", "design_id"])
    assert np.array_equal(_ids["design_id"].values, (_ids["global_sim_id"].values - 1) // data_io.REPS + 1)
    blocks_inf = np.arange(n) // data_io.PAIRS_PER_BLOCK
    out["paired_per_sim"] = {}
    for m in PAIRM:
        out["paired_per_sim"].update(paired_inf(c[f"inf_mean_{m}"], c[f"non_mean_{m}"], m, blocks=blocks_inf))
    _clustered_family(out["paired_per_sim"])
    out["_methods_note"] = desc["_methods_note"]
    out["pseudoreplication_payoff"] = desc["pseudoreplication_payoff"]
    out["mixed_model_payoff"] = desc["mixed_model_payoff"]

    with open(os.path.join(OUT_DIR, "influencer_node_evidence.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE influencer_node_evidence.json")
    print(json.dumps(out, indent=1))


# ================================================================== stream tier
def cmd_stream_nodes(args):
    batch, n_rows = open_arrow(os.path.join(args.sweep, "nodes.arrow"))
    if args.max_rows is not None:
        n_rows = min(n_rows, args.max_rows)
    assert n_rows % N_PER_SIM == 0, f"n_rows {n_rows} not a multiple of {N_PER_SIM}"
    n_sims = n_rows // N_PER_SIM
    chunk_rows = args.chunk_sims * N_PER_SIM

    # per-sim outputs (indexed by global_sim_id - 1); NaN until filled
    per = {k: np.full(N_SIMS, np.nan) for k in PERSIM_METRICS}
    seen = np.zeros(N_SIMS, dtype=bool)
    # pooled agent_veracity: hold the full node vector for the exact describe() percentiles
    av_buf = np.empty(n_rows, dtype=np.float64)
    n_zero_opportunity = n_zero_payoff = n_av_eq_0 = 0

    t0 = time.time()
    for off, sl in stream_chunks(batch, n_rows, chunk_rows, label="nodes"):
        assert off % N_PER_SIM == 0
        node = sl.to_pandas()[NODE_COLS]

        rep = node["payoff"].values; best = node["best_payoff"].values; worst = node["worst_payoff"].values
        av = agent_veracity(rep, best, worst)
        av_buf[off:off + len(av)] = av
        n_zero_opportunity += int(((best == 0) & (worst == 0)).sum())
        n_zero_payoff += int((rep == 0).sum())
        n_av_eq_0 += int((av == 0).sum())

        node["agent_veracity"] = av
        node["D"] = (N_RECORDED - node["true_shares"] - node["fake_shares"]) / N_RECORDED
        node["tpr_mean"] = node["tpr_alpha"] / (node["tpr_alpha"] + node["tpr_beta"])
        node["true_prev"] = node["true_alpha"] / (node["true_alpha"] + node["true_beta"])
        g = node.groupby("global_sim_id")
        # the per-simulation reductions, applied per sim-aligned chunk (bit-identical to the
        # whole-array groupby because each sim lies wholly inside one chunk)
        res = {
            "sen_welfare": g["agent_veracity"].apply(lambda x: sen_welfare(x.values)),
            "discard_rate": g["D"].mean(),
            "psi": g["tpr_mean"].apply(lambda x: belief_dispersion(x.values)),
            "bc": g["tpr_mean"].apply(lambda x: bimodality_coefficient(x.values)),
            "dip_p": g["tpr_mean"].apply(lambda x: dip_test(x.values)["p"]),
            "mean_true_belief": g["true_prev"].mean(),
        }
        idx = res["sen_welfare"].index.values.astype(np.int64) - 1   # global_sim_id -> 0-based
        for k, s in res.items():
            per[k][idx] = s.values
        seen[idx] = True
        del node, g, res, av

    filled = seen[:n_sims]
    assert filled.all(), f"{(~filled).sum()} of first {n_sims} sims unfilled"

    # pooled agent_veracity block (exact describe/hist/fracs over the full node vector)
    pooled = {
        "n_nodes": int(n_rows),
        "agent_veracity_desc": describe_dict(pd.Series(av_buf)),
        "agent_veracity_hist": hist_dict(av_buf, np.linspace(-1, 1, 11)),
        "frac_zero_opportunity": round(n_zero_opportunity / n_rows, 4),
        "frac_zero_payoff": round(n_zero_payoff / n_rows, 4),
        "frac_av_eq_0": round(n_av_eq_0 / n_rows, 4),
    }

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"acc_nodes_{args.label}.npz")
    np.savez_compressed(
        dst, label=args.label, sweep=args.sweep, n_rows_streamed=n_rows, n_sims=n_sims,
        gsid=(np.arange(N_SIMS) + 1).astype(np.int64),
        eda_pooled_json=json.dumps(pooled),
        **{k: per[k] for k in per})
    print(f"WROTE {dst}  ({n_sims} sims; {time.time()-t0:.1f}s; "
          f"sen_welfare[:3]={per['sen_welfare'][:3]}, dip_p[:3]={per['dip_p'][:3]})", flush=True)


def cmd_stream_cascades(args):
    """Per-simulation cascade sums (means = sum/count) -> acc_casc_<label>.npz. Additive, bincount."""
    batch, n_rows = open_arrow(os.path.join(args.sweep, "cascades.arrow"))
    if args.max_rows is not None:
        n_rows = min(n_rows, args.max_rows)
    metrics = list(data_io.CASC_METRICS) \
        + (["structural_virality"] if "structural_virality" in batch.schema.names else [])
    CHUNK = 5_000_000
    sums = {m: np.zeros(N_SIMS, np.float64) for m in metrics}
    count = np.zeros(N_SIMS, np.int64)
    t0 = time.time()
    for off, sl in stream_chunks(batch, n_rows, CHUNK, label="casc"):
        key = sl.column("global_sim_id").to_numpy(zero_copy_only=True) - 1
        count += np.bincount(key, minlength=N_SIMS)
        for m in metrics:
            sums[m] += np.bincount(key, weights=sl.column(m).to_numpy(zero_copy_only=True), minlength=N_SIMS)
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"acc_casc_{args.label}.npz")
    np.savez_compressed(dst, label=args.label, n_rows_streamed=n_rows, metrics=np.array(metrics),
                        count=count, **{f"sum_{m}": sums[m] for m in metrics})
    print(f"WROTE {dst}  ({time.time()-t0:.1f}s)", flush=True)


def cmd_sample_cascade_sizes(args):
    """Shared-RNG A-then-B 300k cascade_size samples (the compare KS/Wasserstein inputs)."""
    def full_cs(d):
        batch, n = open_arrow(os.path.join(d, "cascades.arrow"))
        cs = np.empty(n, dtype=np.int64)
        for off, sl in stream_chunks(batch, n, 20_000_000):
            cs[off:off + sl.num_rows] = sl.column("cascade_size").to_numpy(zero_copy_only=True)
        return cs
    rng = np.random.default_rng(0)                      # ONE shared rng, A then B (exact)
    csa = full_cs(args.baseline)
    sa = csa if len(csa) <= 300000 else rng.choice(csa, 300000, replace=False)
    del csa; gc.collect()
    csb = full_cs(args.influencer)
    sb = csb if len(csb) <= 300000 else rng.choice(csb, 300000, replace=False)
    del csb; gc.collect()
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "cascade_size_samples.npz")
    np.savez_compressed(dst, cascade_size_sample_baseline=np.asarray(sa, float),
                        cascade_size_sample_influencer=np.asarray(sb, float))
    print(f"WROTE {dst}", flush=True)


def cmd_stream_influencer(args):
    """Two-pass influencer/ordinary reducer -> acc_infl.npz (for the influencer reduce). The
    descriptive blocks (id_check/seed_degree/group_counts/pooled/pseudoreplication/mixed_model)
    are computed IN-STREAM on the held per-group node arrays; the reduce runs paired_inf on the
    stored per-sim inf/non means for paired_per_sim."""
    from utils import pooled_sd, cohens_d_pooled
    B = args.sweep
    # ---- pass 1: cascades -> seed bitmap [N_SIMS, N_PER_SIM+1], pooled seed_degree ----
    cbatch, nc = open_arrow(os.path.join(B, "cascades.arrow"))
    if args.max_rows is not None:
        nc = min(nc, args.max_rows)
    seed_bitmap = np.zeros((N_SIMS, N_PER_SIM + 1), dtype=bool)
    seed_deg_sum = 0.0; seed_deg_cnt = 0
    t0 = time.time()
    for off, sl in stream_chunks(cbatch, nc, 5_000_000):
        sid = sl.column("global_sim_id").to_numpy(zero_copy_only=True)
        sn = sl.column("seed_node").to_numpy(zero_copy_only=True)
        sd = sl.column("seed_degree").to_numpy(zero_copy_only=True)
        seed_bitmap[sid - 1, sn] = True
        seed_deg_sum += float(sd.sum()); seed_deg_cnt += len(sd)
    unique_seeds_per_sim = seed_bitmap.sum(axis=1).astype(np.int64)   # all unique seed_nodes per sim
    print(f"  pass1 cascades done {time.time()-t0:.1f}s; seeds/sim min/mean/max="
          f"{unique_seeds_per_sim.min()}/{unique_seeds_per_sim.mean():.3f}/{unique_seeds_per_sim.max()}", flush=True)

    # ---- pass 2: nodes -> per-(sim,group) means (paired) + held per-group arrays (pooled) + mm rows ----
    keep = np.random.default_rng(0).choice(np.arange(1, N_SIMS + 1), min(300, N_SIMS),
                                            replace=False)   # mixed-model subsample
    nbatch, nn = open_arrow(os.path.join(B, "nodes.arrow"))
    if args.max_rows is not None:
        nn = min(nn, args.max_rows)
    inf_mean = {m: np.full(N_SIMS, np.nan) for m in PAIRM}; non_mean = {m: np.full(N_SIMS, np.nan) for m in PAIRM}
    # held per-group node values for the pooled block (group 0=non, 1=infl); preallocate + cursor (exactly 15 seeds/sim)
    cap = {1: 16 * N_SIMS, 0: 286 * N_SIMS}
    HELD = {"payoff": np.float64, "agent_veracity": np.float64, "total_shares": np.int16,
            "node_veracity_diff": np.int16, "truth_share_ratio": np.float64}   # ints <=2000; f64-promoted in reductions
    arr = {(m, g): np.empty(cap[g], dt) for m, dt in HELD.items() for g in (0, 1)}
    pos = {k: 0 for k in arr}
    mm_pay, mm_inf, mm_gsid = [], [], []
    for off, sl in stream_chunks(nbatch, nn, args.chunk_sims * N_PER_SIM, label="infl-nodes"):
        sid = sl.column("global_sim_id").to_numpy(zero_copy_only=True)
        nid = sl.column("node_id").to_numpy(zero_copy_only=True)
        rep = sl.column("payoff").to_numpy(zero_copy_only=True)
        best = sl.column("best_payoff").to_numpy(zero_copy_only=True)
        worst = sl.column("worst_payoff").to_numpy(zero_copy_only=True)
        tsh = sl.column("true_shares").to_numpy(zero_copy_only=True).astype(np.int64)   # signed, matches pandas int64
        fsh = sl.column("fake_shares").to_numpy(zero_copy_only=True).astype(np.int64)
        av = agent_veracity(rep, best, worst)
        tot = tsh + fsh                                             # int64 (matches nd["total_shares"])
        nvd = tsh - fsh                                             # int64
        pm = tot > 0                                               # participants (truth_share_ratio)
        tsr = np.where(pm, tsh / np.where(tot == 0, 1, tot), np.nan)
        grp = seed_bitmap[sid - 1, nid].astype(np.int64)           # 1=influencer, 0=ordinary
        # per-sim group means via pandas groupby.mean -- byte-identical to a whole-array groupby
        # (each sim lies wholly inside one sim-aligned chunk, Kahan summation included)
        df = pd.DataFrame({"gsid": sid, "inf": grp.astype(bool), "payoff": rep.astype(np.float64),
                           "agent_veracity": av, "total_shares": tot, "node_veracity_diff": nvd,
                           "participation": pm.astype(np.float64), "truth_share_ratio": tsr})
        ag = df.groupby(["gsid", "inf"]).agg(
            payoff=("payoff", "mean"), agent_veracity=("agent_veracity", "mean"),
            total_shares=("total_shares", "mean"), node_veracity_diff=("node_veracity_diff", "mean"),
            participation=("participation", "mean")).reset_index()
        gs = ag["gsid"].values - 1; im = ag["inf"].values
        for m in ["payoff", "agent_veracity", "total_shares", "node_veracity_diff", "participation"]:
            v = ag[m].values; inf_mean[m][gs[im]] = v[im]; non_mean[m][gs[~im]] = v[~im]
        pt = df[pm].groupby(["gsid", "inf"])["truth_share_ratio"].mean().reset_index()
        gp = pt["gsid"].values - 1; ip = pt["inf"].values; vt = pt["truth_share_ratio"].values
        inf_mean["truth_share_ratio"][gp[ip]] = vt[ip]; non_mean["truth_share_ratio"][gp[~ip]] = vt[~ip]
        for g in (0, 1):
            gm = grp == g
            for m, col in (("payoff", rep), ("agent_veracity", av), ("total_shares", tot), ("node_veracity_diff", nvd)):
                a = col[gm]; k = (m, g); arr[k][pos[k]:pos[k] + a.size] = a; pos[k] += a.size
            t = tsr[gm & pm]; k = ("truth_share_ratio", g); arr[k][pos[k]:pos[k] + t.size] = t; pos[k] += t.size
        km = np.isin(sid, keep)
        if km.any():
            mm_pay.append(rep[km].astype(np.float64)); mm_inf.append(grp[km].copy()); mm_gsid.append(sid[km].copy())

    # ---- descriptive blocks IN-STREAM on the held per-group arrays ----
    A = {m: arr[(m, 1)][:pos[(m, 1)]] for m in HELD}               # influencer-group values (node order)
    Bn = {m: arr[(m, 0)][:pos[(m, 0)]] for m in HELD}              # ordinary-group values (node order)

    def _pooled(m):
        a, b = A[m], Bn[m]
        return {"influencer_mean": round(float(a.mean()), 5), "noninfluencer_mean": round(float(b.mean()), 5),
                "influencer_median": round(float(np.median(a)), 5), "noninfluencer_median": round(float(np.median(b)), 5),
                "cohens_d_pooled_DESCRIPTIVE_ONLY": round(float(cohens_d_pooled(a, b)), 4),
                "_warning": "pseudoreplicated (Hurlbert 1984); NOT an inferential effect size - use paired_per_sim"}
    desc = {}
    desc["influencer_id_check"] = {"n_sims": int(len(unique_seeds_per_sim)),
        "influencers_per_sim_min": int(unique_seeds_per_sim.min()), "mean": round(float(unique_seeds_per_sim.mean()), 3),
        "max": int(unique_seeds_per_sim.max()),
        "note": "find_influencers returns exactly 15 (floor(0.05*300)); unique seeds should be <=15 and ~15 if all hubs got seeded"}
    desc["influencer_mean_seed_degree"] = round(seed_deg_sum / seed_deg_cnt, 3)
    desc["group_counts"] = {"influencer_node_obs": int(pos[("payoff", 1)]), "noninfluencer_node_obs": int(pos[("payoff", 0)])}
    desc["pooled"] = {"agent_veracity": _pooled("agent_veracity"), "payoff": _pooled("payoff"),
        "node_veracity_diff": _pooled("node_veracity_diff"),
        "truth_share_ratio_participants": _pooled("truth_share_ratio"), "total_shares": _pooled("total_shares"),
        "participation_rate": {"influencer": round(float((A["total_shares"] > 0).mean()), 4),
                               "noninfluencer": round(float((Bn["total_shares"] > 0).mean()), 4)}}

    desc["_methods_note"] = ("paired_per_sim is the valid estimator; the pooled block is DESCRIPTIVE "
        "ONLY (pseudoreplicated). pseudoreplication_payoff shows a SE built on ~15M independent rows is "
        "far smaller than the valid per-simulation paired SE (false precision).")
    try:
        gi, gn = A["payoff"], Bn["payoff"]
        naive_se_indep = float(pooled_sd(gi, gn) * np.sqrt(1.0 / len(gi) + 1.0 / len(gn)))
        dpair = inf_mean["payoff"] - non_mean["payoff"]
        valid_paired_se = float(dpair.std(ddof=1) / np.sqrt(len(dpair)))
        desc["pseudoreplication_payoff"] = {"n_influencer_rows": int(len(gi)), "n_ordinary_rows": int(len(gn)),
            "naive_independent_se_15M": round(naive_se_indep, 4), "valid_paired_se": round(valid_paired_se, 4),
            "false_precision_ratio": round(valid_paired_se / naive_se_indep, 2) if naive_se_indep > 0 else None}
    except Exception as e:
        desc["pseudoreplication_payoff"] = {"error": repr(e)}
    try:
        import statsmodels.formula.api as smf
        sub = pd.DataFrame({"payoff": np.concatenate(mm_pay), "is_influencer": np.concatenate(mm_inf).astype(int),
                            "global_sim_id": np.concatenate(mm_gsid)})
        mm = smf.mixedlm("payoff ~ is_influencer", sub, groups=sub["global_sim_id"]).fit()
        desc["mixed_model_payoff"] = {"n_sims_subsample": int(len(keep)),
            "is_influencer_coef": round(float(mm.params["is_influencer"]), 4),
            "cluster_aware_se": round(float(mm.bse["is_influencer"]), 4),
            "note": "within-sim contrast; corroborates the per-sim paired estimate"}
    except Exception as e:
        desc["mixed_model_payoff"] = {"error": repr(e)}

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "acc_infl.npz")
    save = {"desc_json": json.dumps(desc)}
    for m in PAIRM:
        save[f"inf_mean_{m}"] = inf_mean[m]; save[f"non_mean_{m}"] = non_mean[m]
    np.savez_compressed(dst, **save)
    print(f"WROTE {dst}  ({time.time()-t0:.1f}s total)", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Build the evidence JSONs for the seeding-comparison analysis (stream + reduce).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ed = sub.add_parser("eda", help="reduce the cache -> eda_evidence.json")
    ed.add_argument("--baseline", default=DEFAULT_BASELINE, help="baseline sweep directory")
    ed.add_argument("--cache", default=data_io.NODECACHE_DIR, help="per-sim node/cascade cache directory")
    cp = sub.add_parser("compare", help="reduce the cache -> compare_evidence.json")
    cp.add_argument("--baseline", default=DEFAULT_BASELINE, help="baseline sweep directory")
    cp.add_argument("--influencer", default=DEFAULT_INFLUENCER, help="influencer sweep directory")
    cp.add_argument("--cache", default=data_io.NODECACHE_DIR, help="per-sim node/cascade cache directory")
    inf = sub.add_parser("influencer", help="reduce the cache -> influencer_node_evidence.json")
    inf.add_argument("--cache", default=data_io.NODECACHE_DIR, help="per-sim node/cascade cache directory")

    sn = sub.add_parser("stream", help="stream nodes.arrow -> acc_nodes_<label>.npz")
    sn.add_argument("--sweep", required=True)
    sn.add_argument("--label", required=True)
    sn.add_argument("--out", required=True)
    sn.add_argument("--chunk-sims", type=int, default=16_384)
    sn.add_argument("--max-rows", type=int, default=None, help="slice-check only")
    sc = sub.add_parser("stream-cascades", help="stream cascades.arrow -> acc_casc_<label>.npz")
    for a in ("--sweep", "--label", "--out"):
        sc.add_argument(a, required=True)
    sc.add_argument("--max-rows", type=int, default=None)
    ss = sub.add_parser("sample-cascade-sizes", help="shared-RNG 300k cascade_size samples -> npz")
    ss.add_argument("--baseline", default=DEFAULT_BASELINE)
    ss.add_argument("--influencer", default=DEFAULT_INFLUENCER)
    ss.add_argument("--out", required=True)
    si = sub.add_parser("stream-influencer", help="cascades+nodes -> acc_infl.npz")
    si.add_argument("--sweep", required=True)
    si.add_argument("--out", required=True)
    si.add_argument("--chunk-sims", type=int, default=8_192)   # smaller: holds ~3.8 GB + a per-chunk groupby frame
    si.add_argument("--max-rows", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "eda":
        cmd_eda(args.baseline, args.cache)
    elif args.cmd == "compare":
        cmd_compare(args.baseline, args.influencer, args.cache)
    elif args.cmd == "influencer":
        cmd_influencer(args.cache)
    elif args.cmd == "stream":
        cmd_stream_nodes(args)
    elif args.cmd == "stream-cascades":
        cmd_stream_cascades(args)
    elif args.cmd == "sample-cascade-sizes":
        cmd_sample_cascade_sizes(args)
    elif args.cmd == "stream-influencer":
        cmd_stream_influencer(args)


if __name__ == "__main__":
    main()
