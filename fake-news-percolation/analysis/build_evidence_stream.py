"""Bounded-memory streaming reducer for nodes.arrow / cascades.arrow -> a small per-simulation CACHE.

This "stream" tier allows build_evidence.py to run without HPC resources. It memory-maps the
~16.5 GB nodes.arrow (and ~39 GB cascades.arrow) and walks them in simulation-aligned chunks.
Each simulation is a contiguous 300-node block, guaranteed by the Julia `vcat(node_dfs...)` write
order. The EXACT original per-simulation reductions (`build_evidence.cmd_eda`'s `groupby.apply`)
run chunk-by-chunk. The result is byte-identical to the full-load path because the same
pandas/utils code runs on the same data, only chunked. Peak RAM stays a few GB instead of
~55-60 GiB.

Output cache (per configuration, in --out): acc_nodes_<label>.npz. build_evidence.py
{eda,compare,influencer} --cache <out> then reduces from these instead of loading nodes.arrow.

Usage (the main analysis environment, see requirements.txt; from repo root):
  python analysis/build_evidence_stream.py stream --sweep data/sweep_..._baseline --label baseline \
      --out analysis/runs/nodecache_2026_07_08
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))
import data_io                                                              # noqa: E402
from data_io import N_PAIRS, ram_guard, trim_working_set                   # noqa: E402
from utils import sen_welfare, belief_dispersion, bimodality_coefficient, dip_test  # noqa: E402

N_PER_SIM = 300          # nodes per simulation (Julia N_AGENTS)
N_RECORDED = 1000        # recorded cascades per simulation (build_evidence.N_RECORDED)
N_SIMS = N_PAIRS         # 491_520
MIN_AVAIL_GB = 1.0
NODE_COLS = ["global_sim_id", "payoff", "best_payoff", "worst_payoff", "true_shares", "fake_shares",
             "tpr_alpha", "tpr_beta", "true_alpha", "true_beta"]


def agent_veracity(rep, best, worst):   # verbatim from build_evidence.py:56-60
    return np.where((rep < 0) & (worst != 0), -(rep / np.where(worst == 0, 1, worst)),
                    np.where((rep > 0) & (best != 0), rep / np.where(best == 0, 1, best), 0.0))


def cmd_stream_nodes(args):
    path = os.path.join(args.sweep, "nodes.arrow")
    batch = pa.ipc.open_file(pa.memory_map(path, "r")).get_batch(0)
    n_rows = batch.num_rows if args.max_rows is None else min(batch.num_rows, args.max_rows)
    assert n_rows % N_PER_SIM == 0, f"n_rows {n_rows} not a multiple of {N_PER_SIM}"
    n_sims = n_rows // N_PER_SIM
    chunk_rows = args.chunk_sims * N_PER_SIM

    # per-sim outputs (indexed by global_sim_id - 1); NaN until filled
    per = {k: np.full(N_SIMS, np.nan) for k in
           ("sen_welfare", "discard_rate", "psi", "bc", "dip_p", "mean_true_belief")}
    seen = np.zeros(N_SIMS, dtype=bool)
    # pooled agent_veracity: hold the full node vector for the exact describe() percentiles
    av_buf = np.empty(n_rows, dtype=np.float64)
    n_zero_opportunity = n_zero_payoff = n_av_eq_0 = 0

    t0 = time.time()
    for off in range(0, n_rows, chunk_rows):
        ram_guard(MIN_AVAIL_GB)
        assert off % N_PER_SIM == 0
        n = min(chunk_rows, n_rows - off)
        sl = batch.slice(off, n)
        node = sl.to_pandas()[NODE_COLS]

        rep = node["payoff"].values; best = node["best_payoff"].values; worst = node["worst_payoff"].values
        av = agent_veracity(rep, best, worst)
        av_buf[off:off + n] = av
        n_zero_opportunity += int(((best == 0) & (worst == 0)).sum())
        n_zero_payoff += int((rep == 0).sum())
        n_av_eq_0 += int((av == 0).sum())

        node["agent_veracity"] = av
        node["D"] = (N_RECORDED - node["true_shares"] - node["fake_shares"]) / N_RECORDED
        node["tpr_mean"] = node["tpr_alpha"] / (node["tpr_alpha"] + node["tpr_beta"])
        node["true_prev"] = node["true_alpha"] / (node["true_alpha"] + node["true_beta"])
        g = node.groupby("global_sim_id")
        # EXACT original cmd_eda reductions, applied per chunk (bit-identical, bounded memory)
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
        trim_working_set()
        rate = (off + n) / 1e6 / max(time.time() - t0, 1e-9)
        print(f"  nodes {off + n:>12,}/{n_rows:,}  {rate:5.1f} M rows/s  "
              f"avail {psutil.virtual_memory().available / 2**30:.1f} GiB", flush=True)

    filled = seen[:n_sims]
    assert filled.all(), f"{(~filled).sum()} of first {n_sims} sims unfilled"

    # pooled agent_veracity block (exact describe/hist/fracs over the full node vector = original)
    def _round(v, d):
        return round(float(v), d)
    desc = {k: _round(v, 4) for k, v in
            pd.Series(av_buf).describe(percentiles=[.05, .25, .5, .75, .95]).to_dict().items()}
    c, e = np.histogram(av_buf, bins=np.linspace(-1, 1, 11))
    av_hist = {"edges": [_round(v, 3) for v in e], "counts": [int(v) for v in c]}
    pooled = {
        "n_nodes": int(n_rows),
        "agent_veracity_desc": desc,
        "agent_veracity_hist": av_hist,
        "frac_zero_opportunity": _round(n_zero_opportunity / n_rows, 4),
        "frac_zero_payoff": _round(n_zero_payoff / n_rows, 4),
        "frac_av_eq_0": _round(n_av_eq_0 / n_rows, 4),
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


CASC_METRICS = ["cascade_size", "max_depth", "seed_degree", "shares", "verifications"]


def cmd_stream_cascades(args):
    """Per-simulation cascade sums (means = sum/count) -> acc_casc_<label>.npz. Additive, bincount."""
    reader = pa.ipc.open_file(pa.memory_map(os.path.join(args.sweep, "cascades.arrow"), "r"))
    batch = reader.get_batch(0)
    n_rows = batch.num_rows if args.max_rows is None else min(batch.num_rows, args.max_rows)
    metrics = CASC_METRICS + (["structural_virality"] if "structural_virality" in reader.schema.names else [])
    CHUNK = 5_000_000
    sums = {m: np.zeros(N_SIMS, np.float64) for m in metrics}
    count = np.zeros(N_SIMS, np.int64)
    t0 = time.time()
    for off in range(0, n_rows, CHUNK):
        ram_guard(MIN_AVAIL_GB)
        n = min(CHUNK, n_rows - off)
        sl = batch.slice(off, n)
        key = sl.column("global_sim_id").to_numpy(zero_copy_only=True) - 1
        count += np.bincount(key, minlength=N_SIMS)
        for m in metrics:
            sums[m] += np.bincount(key, weights=sl.column(m).to_numpy(zero_copy_only=True), minlength=N_SIMS)
        trim_working_set()
        print(f"  casc {off + n:>12,}/{n_rows:,}  {(off+n)/1e6/max(time.time()-t0,1e-9):5.1f} M rows/s  "
              f"avail {psutil.virtual_memory().available/2**30:.1f} GiB", flush=True)
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"acc_casc_{args.label}.npz")
    np.savez_compressed(dst, label=args.label, n_rows_streamed=n_rows, metrics=np.array(metrics),
                        count=count, **{f"sum_{m}": sums[m] for m in metrics})
    print(f"WROTE {dst}  ({time.time()-t0:.1f}s)", flush=True)


def cmd_sample_cascade_sizes(args):
    """Shared-RNG A-then-B 300k cascade_size samples (matches build_evidence.cmd_compare L280-289)."""
    def full_cs(d):
        batch = pa.ipc.open_file(pa.memory_map(os.path.join(d, "cascades.arrow"), "r")).get_batch(0)
        n = batch.num_rows
        cs = np.empty(n, dtype=np.int64)
        for off in range(0, n, 20_000_000):
            ram_guard(MIN_AVAIL_GB)
            m = min(20_000_000, n - off)
            cs[off:off + m] = batch.slice(off, m).column("cascade_size").to_numpy(zero_copy_only=True)
            trim_working_set()
        return cs
    import gc
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
    """Two-pass influencer/ordinary reducer -> acc_infl.npz (for cmd_influencer --cache). The descriptive
    blocks (id_check/seed_degree/group_counts/pooled/pseudoreplication/mixed_model) are computed IN-STREAM
    by the exact original ops on the held per-group node arrays (reuse of cmd_influencer, not re-derived);
    the reduce runs paired_inf on the stored per-sim inf/non means for paired_per_sim."""
    from utils import pooled_sd, cohens_d_pooled
    B = args.sweep
    # ---- pass 1: cascades -> seed bitmap [N_SIMS,301], pooled seed_degree, unique seeds/sim ----
    cbatch = pa.ipc.open_file(pa.memory_map(os.path.join(B, "cascades.arrow"), "r")).get_batch(0)
    nc = cbatch.num_rows if args.max_rows is None else min(cbatch.num_rows, args.max_rows)
    seed_bitmap = np.zeros((N_SIMS, 301), dtype=bool)
    seed_deg_sum = 0.0; seed_deg_cnt = 0
    t0 = time.time()
    for off in range(0, nc, 5_000_000):
        ram_guard(MIN_AVAIL_GB)
        n = min(5_000_000, nc - off); sl = cbatch.slice(off, n)
        sid = sl.column("global_sim_id").to_numpy(zero_copy_only=True)
        sn = sl.column("seed_node").to_numpy(zero_copy_only=True)
        sd = sl.column("seed_degree").to_numpy(zero_copy_only=True)
        seed_bitmap[sid - 1, sn] = True
        seed_deg_sum += float(sd.sum()); seed_deg_cnt += len(sd)
        trim_working_set()
    unique_seeds_per_sim = seed_bitmap.sum(axis=1).astype(np.int64)   # all unique seed_nodes per sim
    print(f"  pass1 cascades done {time.time()-t0:.1f}s; seeds/sim min/mean/max="
          f"{unique_seeds_per_sim.min()}/{unique_seeds_per_sim.mean():.3f}/{unique_seeds_per_sim.max()}", flush=True)

    # ---- pass 2: nodes -> per-(sim,group) sums/counts (paired means) + held per-group arrays (pooled) + mm rows ----
    keep = np.random.default_rng(0).choice(np.arange(1, N_SIMS + 1), 300, replace=False)   # matches cmd_influencer
    nbatch = pa.ipc.open_file(pa.memory_map(os.path.join(B, "nodes.arrow"), "r")).get_batch(0)
    nn = nbatch.num_rows if args.max_rows is None else min(args.max_rows, nbatch.num_rows)
    PAIRM = ["payoff", "agent_veracity", "total_shares", "node_veracity_diff", "participation", "truth_share_ratio"]
    inf_mean = {m: np.full(N_SIMS, np.nan) for m in PAIRM}; non_mean = {m: np.full(N_SIMS, np.nan) for m in PAIRM}
    # held per-group node values for the pooled block (group 0=non, 1=infl); preallocate + cursor (exactly 15 seeds/sim)
    cap = {1: 16 * N_SIMS, 0: 286 * N_SIMS}
    HELD = {"payoff": np.float64, "agent_veracity": np.float64, "total_shares": np.int16,
            "node_veracity_diff": np.int16, "truth_share_ratio": np.float64}   # ints <=2000; f64-promoted in reductions
    arr = {(m, g): np.empty(cap[g], dt) for m, dt in HELD.items() for g in (0, 1)}
    pos = {k: 0 for k in arr}
    mm_pay, mm_inf, mm_gsid = [], [], []
    t1 = time.time()
    for off in range(0, nn, args.chunk_sims * N_PER_SIM):
        ram_guard(MIN_AVAIL_GB)
        n = min(args.chunk_sims * N_PER_SIM, nn - off); sl = nbatch.slice(off, n)
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
        # per-sim group means via pandas groupby.mean -- byte-identical to cmd_influencer (each sim lies
        # wholly inside one sim-aligned chunk, so this per-chunk groupby == the whole-array groupby, Kahan summation included)
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
        trim_working_set()
        print(f"  infl-nodes {off + n:>12,}/{nn:,}  {(off+n)/1e6/max(time.time()-t1,1e-9):5.1f} M rows/s  "
              f"avail {psutil.virtual_memory().available/2**30:.1f} GiB", flush=True)

    # ---- descriptive blocks IN-STREAM via the exact original ops (byte-identical reuse of cmd_influencer) ----
    A = {m: arr[(m, 1)][:pos[(m, 1)]] for m in HELD}               # influencer-group values (nd order)
    Bn = {m: arr[(m, 0)][:pos[(m, 0)]] for m in HELD}              # ordinary-group values (nd order)

    def _pooled(m):                                                # mirrors cmd_influencer.pooled()
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
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
    for a in ("--baseline", "--influencer", "--out"):
        ss.add_argument(a, required=True)
    si = sub.add_parser("stream-influencer", help="cascades+nodes -> acc_infl.npz")
    si.add_argument("--sweep", required=True)
    si.add_argument("--out", required=True)
    si.add_argument("--chunk-sims", type=int, default=8_192)   # smaller: holds ~3.8 GB + a per-chunk groupby frame
    si.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "stream":
        cmd_stream_nodes(args)
    elif args.cmd == "stream-cascades":
        cmd_stream_cascades(args)
    elif args.cmd == "sample-cascade-sizes":
        cmd_sample_cascade_sizes(args)
    elif args.cmd == "stream-influencer":
        cmd_stream_influencer(args)


if __name__ == "__main__":
    main()
