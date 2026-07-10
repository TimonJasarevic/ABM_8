"""Post-hoc justification numbers for the replication count, the burn-in length, and the Sen
negative-mean regime.

Three blocks -> runs/replication_justification_2026_07_09/replication_evidence.json:
  mser5             MSER-5 warm-up truncation (Hoad et al. 2010 family) on replicate-averaged
                    per-cascade-index series from the dedicated 2000-cascade probe
                    (run_experiment_mser_probe.jl; the production sweeps never log the burn-in).
  required_runs     Secchi & Seri (2017) run-count check (their approximation is calibrated at
                    alpha=0.01, power=0.95) at the SESOI-implied effect size and two benchmarks.
  sen_negative_mean Fraction of simulations (and design points) whose mean normalised payoff is
                    negative: sign(sen_welfare) == sign(mean(agent_veracity)) because the RSV
                    Gini is bounded in [0, 1], so W = mean * (1 - G) carries the mean's sign.

Usage: python analysis/replication_justification.py
(run in the main analysis environment, see requirements.txt)
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
import data_io
import utils

PROBE_DIR = os.path.join(HERE, "runs", "mser_probe_2026_07_09")
OUT_DIR = os.path.join(HERE, "runs", "replication_justification_2026_07_09")
CACHE_DIR = data_io.NODECACHE_DIR
BURN_IN_USED = 1000
N_AGENTS = data_io.N_NODES

# SESOI 0.05 raw reach units over the average condition SD 0.255782 -> d_av 0.1955; the
# two-group Cohen's f equivalent is d/2.
SESOI_DAV = 0.1955
ES_GRID = {"sesoi_implied_f_0.098": SESOI_DAV / 2, "cohen_small_f_0.10": 0.10,
           "cohen_medium_f_0.25": 0.25}


def mser_block():
    cfg = json.load(open(os.path.join(PROBE_DIR, "probe_config.json")))
    df = pd.read_csv(os.path.join(PROBE_DIR, "mser_probe_cascades.csv"))
    expected = len(cfg["design_ids"]) * cfg["reps_per_point"] * cfg["cascades_per_sim"]
    assert len(df) == expected, (len(df), expected)
    df["reach"] = df["cascade_size"] / N_AGENTS
    df["fake_reach"] = np.where(df["is_fake"] == 1, df["reach"], np.nan)
    df["verify_rate"] = df["verifications"] / df["cascade_size"]

    points, truncs = [], []
    for did, sub in df.groupby("design_id"):
        g = sub.groupby("cascade_id")
        series = {
            "reach": g["reach"].mean().sort_index().values,
            "fake_reach": g["fake_reach"].mean().sort_index().values,   # mean over fake draws only
            "verify_rate": g["verify_rate"].mean().sort_index().values,
        }
        row = {"design_id": int(did),
               "p_fake": float(sub["p_fake"].iloc[0]) if "p_fake" in sub else None}
        for name, s in series.items():
            s = np.asarray(s, float)
            s = np.where(np.isfinite(s), s, np.nanmean(s))  # rare all-true cascade indices
            r = utils.mser5(s, batch=5)
            row[f"mser_truncation_{name}"] = int(r["truncation"])
            truncs.append(int(r["truncation"]))
        points.append(row)
    pts = [p["design_id"] for p in points]
    return {
        "probe": {k: cfg[k] for k in ("design_ids", "reps_per_point", "cascades_per_sim",
                                      "seed_rule", "note")},
        "series": "per-cascade-index series averaged over the 30 replicates of each design point",
        "metrics": ["reach", "fake_reach", "verify_rate"],
        "per_point": points,
        "max_truncation": int(max(truncs)),
        "burn_in_used": BURN_IN_USED,
        "burn_in_adequate_for_all_probed_points": bool(max(truncs) <= BURN_IN_USED),
        "n_points": len(pts),
    }


def required_runs_block():
    sds = {}
    for lab, csv in (("baseline", data_io.BASELINE_SVD_CSV), ("influencer", data_io.INFLUENCER_SVD_CSV)):
        df = pd.read_csv(csv, usecols=["design_id", "avg_fake_cascade"])
        sds[lab] = float(df.groupby("design_id")["avg_fake_cascade"].std(ddof=1).mean())
    out = {
        "method": "Secchi & Seri (2017) required-runs approximation 14.091 * J^-0.640 * es^-1.986; "
                  "calibrated at alpha=0.01, power=0.95 (report as an order-of-magnitude check)",
        "J_design_points": data_io.N_DESIGNS,
        "reps_used": data_io.REPS,
        "required_runs": {label: float(utils.required_runs(data_io.N_DESIGNS, es))
                          for label, es in ES_GRID.items()},
        "es_grid_note": "f = d/2; sesoi-implied d_av = 0.05 raw reach / 0.255782 avg SD = 0.1955",
        "within_design_point_sd_avg_fake_cascade": sds,
    }
    return out


def sen_negative_block():
    res = {"sign_note": "RSV Gini is bounded in [0, 1], so sign(sen_welfare) == "
                        "sign(mean per-agent normalised payoff) except at G == 1 exactly"}
    for lab in ("baseline", "influencer"):
        c = np.load(os.path.join(CACHE_DIR, f"acc_nodes_{lab}.npz"), allow_pickle=True)
        w = np.asarray(c["sen_welfare"], float)
        assert len(w) == data_io.N_PAIRS, len(w)
        dp_mean = w.reshape(data_io.N_DESIGNS, data_io.REPS).mean(axis=1)
        res[lab] = {
            "n_sims": int(len(w)),
            "frac_sims_sen_welfare_negative": float((w < 0).mean()),
            "frac_design_points_mean_sen_welfare_negative": float((dp_mean < 0).mean()),
        }
    return res


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"generated": "2026-07-09",
           "required_runs": required_runs_block(),
           "sen_negative_mean": sen_negative_block()}
    try:
        out["mser5"] = mser_block()
    except FileNotFoundError as e:
        out["mser5"] = {"error": f"probe output missing: {e!r} - run run_experiment_mser_probe.jl first"}
        print("WARN:", out["mser5"]["error"])
    path = os.path.join(OUT_DIR, "replication_evidence.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", path)
    print(json.dumps({k: v for k, v in out.items() if k != "mser5"}, indent=1))
    if isinstance(out.get("mser5"), dict) and "max_truncation" in out["mser5"]:
        print("MSER-5 max truncation:", out["mser5"]["max_truncation"],
              "| burn-in used:", BURN_IN_USED)
