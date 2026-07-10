"""Minimal heterogeneous mean-field benchmark for the cascade model.

Maps the share/verify/discard cascade onto degree-based percolation in the canonical
network-epidemics tradition. The percolation mapping with generating functions follows
Newman (2002, PRE 66 016128). The excess-degree machinery follows Newman, Strogatz &
Watts (2001, PRE 64 026118). The degree-class mean-field treatment follows
Pastor-Satorras & Vespignani (2001, PRL 86 3200). The take-off-probability /
outbreak-size distinction follows Kenah & Robins (2007, PRE 76 036113). The tree
approximation on this clustered graph is used in the spirit of Melnik et al.
(2011, PRE 83 036112).

Structure (all per Saltelli design point, per seeding configuration, vectorised):

  belief layer   Beta-mean beliefs at their learned steady state. Global truth belief
                 and neighbour trust converge to the EXPOSURE-weighted truth rate
                 theta_exp. Fake cascades die earlier, so exposure over-represents true
                 items (the model's over-optimism mechanism). Both are shrunk toward
                 the prior (mean 0.5, strength 2) by the expected number of Beta
                 updates an agent/edge accumulates by the measurement window. Detector
                 beliefs converge to the actual TPR/FPR the same way. theta_exp itself
                 depends on cascade sizes, so the whole layer is a damped outer fixed
                 point.
  decision layer Softmax over U_share/U_verify/U_discard exactly as in the model
                 (utilities scale with the decider's own degree k), giving per-degree
                 forwarding probabilities T_true(k), T_fake(k); the seed uses its
                 never-updated base prior trust (mean 0.5) instead of learned trust.
  percolation    On the annealed degree ensemble of the Julia Holme-Kim generator
  layer          (ported below): excess-degree branching ratio R, edge-extinction
                 fixed point eps, giant exposed fraction f, ignition I (the seed
                 forwards), take-off probability P_g, and mean cascade reach.

Seeding enters ONLY through the seed factor (ignition/take-off); R, eps and f are
seed-invariant given beliefs. That is the analytic form of the manuscript's
ignition-led seeding result; the belief shift through theta_exp is the second-order
channel.

Reads the paired per-simulation sweep CSVs (all 16,384 design points x 30 reps per
configuration; no sampling) and writes
  runs/mean_field_2026_07_09/mean_field_evidence.json   (summary + binned profiles)
  runs/mean_field_2026_07_09/mf_points.npz              (per-design-point arrays for
                                                         make_figures.py)

Usage: python analysis/mean_field.py, run in the main analysis environment
(see requirements.txt).
"""
import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
import data_io

OUT_DIR = data_io.MEANFIELD_DIR

# fixed model constants (Table tab:params; run_sobol_sweep in the Julia runners)
N_AGENTS = data_io.N_NODES
M_EDGES = 3
P_TRIAD = 0.5
G_REP = 1.0                  # reputational gain g
PRIOR_STRENGTH = 2.0         # Beta pseudo-observations
PRIOR_MEAN = 0.5             # mean of the U[0.3, 0.7] prior-mean draw
TOP_FRACTION = 0.05          # influencer pool = top-5% degrees = 15 nodes
# Expected belief-update count is evaluated at the midpoint of the measured window
# (burn-in 1000 + half of the 1000 measured cascades); beliefs keep updating during
# measurement, so the midpoint is the natural single-point summary.
C_EFF = 1500.0

N_GRAPHS = 500               # degree-ensemble size (annealed mean field)
ENSEMBLE_SEED = 42

DAMP = 0.5                   # outer fixed-point damping (per-point adaptive below it)
# Near the percolation transition the belief-size feedback can be steep enough that no
# fixed step converges (the mean-field counterpart of the model's own die-or-flood
# bimodality there). The per-point damping halves on every residual sign flip, never
# recovers, and bottoms out at DAMP_MIN, so such points freeze on a value between the
# two branches; their residuals are counted and reported in the evidence JSON rather
# than hidden.
DAMP_MIN = 1e-5
OUTER_MAX, OUTER_TOL = 100, 1e-9
STEP_TOL = 1e-9              # stop when the largest damped state step is below this
EPS_BISECT_STEPS = 45        # extinction fixed point via bisection (exact to ~3e-14)


# ------------------------------------------------------------------ degree ensemble
def powerlaw_cluster_graph_degrees(n, m, p, rng):
    """Degree sequence + top-degree pool of one Holme-Kim graph.

    Line-by-line port of `powerlaw_cluster_graph` in run_experiment_baseline.jl
    (initial (m+1)-clique; first edge preferential, subsequent edges triadic with
    probability p around the FIRST target u, else preferential; `pref` extended with
    (v, partner) per placed edge). Only degrees are needed here.
    """
    adj = [set() for _ in range(n)]

    def add_edge(a, b):
        adj[a].add(b)
        adj[b].add(a)

    seed = m + 1
    for i in range(seed):
        for j in range(i + 1, seed):
            add_edge(i, j)

    pref = []
    for v in range(seed):
        pref.extend([v] * len(adj[v]))

    for v in range(seed, n):
        connected = []
        u = pref[rng.integers(len(pref))]
        add_edge(v, u)
        connected.append(u)
        while len(connected) < m:
            if rng.random() < p:
                nbrs = tuple(adj[u])
                w = nbrs[rng.integers(len(nbrs))]
            else:
                w = pref[rng.integers(len(pref))]
            if w != v and w not in connected and w not in adj[v]:
                add_edge(v, w)
                connected.append(w)
        for u2 in connected:
            pref.append(v)
            pref.append(u2)

    deg = np.array([len(a) for a in adj])
    n_hubs = max(1, int(np.floor(TOP_FRACTION * n)))
    hub_deg = np.sort(deg)[::-1][:n_hubs]
    return deg, hub_deg


def degree_ensemble():
    """Pooled degree distribution p(k), excess distribution q(k), and hub-seed
    distribution over the N_GRAPHS-graph ensemble (fresh network per simulation in
    the ABM, so the annealed ensemble matches the average over replications)."""
    rng = np.random.default_rng(ENSEMBLE_SEED)
    all_deg, all_hub = [], []
    for _ in range(N_GRAPHS):
        deg, hub = powerlaw_cluster_graph_degrees(N_AGENTS, M_EDGES, P_TRIAD, rng)
        all_deg.append(deg)
        all_hub.append(hub)
    all_deg = np.concatenate(all_deg)
    all_hub = np.concatenate(all_hub)

    kmax = int(all_deg.max())
    k = np.arange(1, kmax + 1)
    pk = np.bincount(all_deg, minlength=kmax + 1)[1:].astype(float)
    pk /= pk.sum()
    qk = k * pk
    qk /= qk.sum()
    hub_pk = np.bincount(all_hub, minlength=kmax + 1)[1:].astype(float)
    hub_pk /= hub_pk.sum()

    stats = {
        "n_graphs": N_GRAPHS, "ensemble_seed": ENSEMBLE_SEED,
        "mean_degree": float(all_deg.mean()), "var_degree": float(all_deg.var()),
        "max_degree": int(all_deg.max()),
        "mean_excess_kminus1": float((qk * (k - 1)).sum()),
        "hub_pool_mean_degree": float(all_hub.mean()),
        "hub_pool_min_degree": int(all_hub.min()),
    }
    return k, pk, qk, hub_pk, stats


# ------------------------------------------------------------------ mean-field solver
def softmax_pi(lam, c, a_s, a_v, k):
    """Share/verify softmax probabilities on the degree grid.

    a_s, a_v: (P,) per-unit-degree utility slopes; lam, c: (P,); k: (K,).
    Returns pi_share, pi_verify with shape (P, K). Max-subtracted exactly as in
    AgentLogic.decide.
    """
    U_s = a_s[:, None] * k[None, :]
    U_v = -c[:, None] + a_v[:, None] * k[None, :]
    U_max = np.maximum(np.maximum(U_s, U_v), 0.0)
    lamc = lam[:, None]
    w_s = np.exp(lamc * (U_s - U_max))
    w_v = np.exp(lamc * (U_v - U_max))
    w_d = np.exp(lamc * (0.0 - U_max))
    tot = w_s + w_v + w_d
    return w_s / tot, w_v / tot


def solve_eps(qk, T, R):
    """Edge-extinction fixed point eps = g(eps) with g(eps) = sum_k q(k)
    [1 - T(k) (1 - eps^(k-1))] (Newman 2002 percolation mapping with
    transmitter-degree-dependent transmissibility). g is increasing and convex with
    g(1) = 1. For branching ratio R = g'(1) <= 1 the only root is eps = 1. For
    R > 1 the stable root is the unique zero of g(eps) - eps in [0, 1), found here by
    vectorised bisection. T: (P, K); returns (P,)."""
    A = (qk[None, :] * (1.0 - T)).sum(axis=1)
    B = qk[None, :] * T

    def g(eps):
        # powers[:, j] = eps**j (j = k-1 on the grid k = 1..kmax); the cumulative
        # product is ~30x cheaper than a broadcast float power at this array size
        powers = np.empty_like(B)
        powers[:, 0] = 1.0
        for j in range(1, powers.shape[1]):
            powers[:, j] = powers[:, j - 1] * eps
        return A + (B * powers).sum(axis=1)

    lo = np.zeros_like(A)
    hi = np.ones_like(A)
    for _ in range(EPS_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        pos = g(mid) - mid > 0.0                # root lies above mid
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    eps = 0.5 * (lo + hi)
    return np.where(R <= 1.0, 1.0, eps)


def percolation_outputs(k, pk, qk, seed_pk, T, T_seed):
    """Branching ratio, extinction, ignition, take-off, giant fraction, mean size."""
    R = (qk[None, :] * T * (k - 1)[None, :]).sum(axis=1)
    eps = solve_eps(qk, T, R)
    epsk = eps[:, None] ** k[None, :]
    f = (pk[None, :] * (1.0 - epsk)).sum(axis=1)
    ignition = (seed_pk[None, :] * T_seed).sum(axis=1)
    takeoff = (seed_pk[None, :] * T_seed * (1.0 - epsk)).sum(axis=1)
    # Mean cascade size in nodes, continuous across the transition. The take-off part
    # contributes P_takeoff * N * f (giant exposed fraction). The finite part uses the
    # extinction-conditioned (dual) branching process, whose per-edge subtree mean is
    # nu = 1 / (1 - R_dual) with R_dual = g'(eps) < 1 in both regimes (Kenah & Robins
    # 2007). With eps = 1 (subcritical) this reduces exactly to the Newman-Strogatz-
    # Watts mean component size 1 + <first generation> / (1 - R). The formula diverges
    # only in a narrow band at R = 1, where the total is capped at N.
    km2pow = eps[:, None] ** np.maximum(k - 2, 0)[None, :]
    R_dual = (qk[None, :] * T * (k - 1)[None, :] * km2pow).sum(axis=1)
    nu = 1.0 / np.maximum(1.0 - R_dual, 1e-12)
    finite_part = ((seed_pk[None, :] * (1.0 - T_seed)).sum(axis=1)
                   + (seed_pk[None, :] * T_seed * epsk
                      * (1.0 + k[None, :] * nu[:, None])).sum(axis=1))
    size = np.minimum(takeoff * f * N_AGENTS + finite_part, float(N_AGENTS))
    return {"eps": eps, "R": R, "f": f, "ignition": ignition,
            "takeoff": takeoff, "size": size}


def solve_config(design, k, pk, qk, seed_pk, verbose=True):
    """Outer belief/percolation fixed point for one seeding configuration.

    design: DataFrame with one row per design point (columns v_cost, loss, tpr, fpr,
    p_fake, rationality, loss_aversion; rationality already back-transformed to
    lambda). Returns per-point arrays.
    """
    P = len(design)
    c = design["v_cost"].values
    ell = design["loss"].values
    tpr = design["tpr"].values
    fpr = design["fpr"].values
    p_fake = design["p_fake"].values
    lam = design["rationality"].values
    lam_la = design["loss_aversion"].values

    mean_k = (pk * k).sum()
    loss_w = lam_la * ell                      # perceived loss weight in the utilities

    # state: mean sizes per item type, exposure-weighted verify probability.
    # Per-point adaptive damping: near the transition the belief-size feedback is
    # steep and a fixed step oscillates, so the step shrinks wherever the residual
    # sign flips and recovers slowly where iteration is smooth.
    E_St = np.full(P, 10.0)
    E_Sf = np.full(P, 10.0)
    pv_bar = np.full(P, 1.0 / 3.0)
    damp = np.full(P, DAMP)
    prev_dSt = prev_dSf = None

    out_t = out_f = None
    theta_bar = np.full(P, PRIOR_MEAN)
    n_iter = OUTER_MAX
    for it in range(OUTER_MAX):
        Ebar = (1.0 - p_fake) * E_St + p_fake * E_Sf
        theta_exp = (1.0 - p_fake) * E_St / Ebar
        # expected Beta-update counts by the measurement midpoint
        n_theta = C_EFF * Ebar / N_AGENTS
        n_tau = C_EFF * np.maximum(Ebar - 1.0, 0.0) / (N_AGENTS * mean_k)
        n_v = C_EFF * (Ebar / N_AGENTS) * pv_bar
        phi_f = p_fake * E_Sf / Ebar
        n_tpr, n_fpr = n_v * phi_f, n_v * (1.0 - phi_f)

        shrink = lambda target, n: (PRIOR_STRENGTH * PRIOR_MEAN + n * target) / (PRIOR_STRENGTH + n)
        theta_new = shrink(theta_exp, n_theta)
        tau = shrink(theta_exp, n_tau)
        tpr_hat = shrink(tpr, n_tpr)
        fpr_hat = shrink(fpr, n_fpr)

        # posteriors: learned trust for non-seeds, never-updated base prior (mean 0.5)
        # for the seed, which reduces the Bayes posterior to theta_bar itself
        p_star = tau * theta_new / (tau * theta_new + (1.0 - tau) * (1.0 - theta_new))
        p_seed = theta_new

        a_s = p_star * G_REP - (1.0 - p_star) * loss_w
        a_v = p_star * (1.0 - fpr_hat) * G_REP - (1.0 - p_star) * (1.0 - tpr_hat) * loss_w
        pi_s, pi_v = softmax_pi(lam, c, a_s, a_v, k)
        a_s_seed = p_seed * G_REP - (1.0 - p_seed) * loss_w
        a_v_seed = p_seed * (1.0 - fpr_hat) * G_REP - (1.0 - p_seed) * (1.0 - tpr_hat) * loss_w
        pi_s_seed, pi_v_seed = softmax_pi(lam, c, a_s_seed, a_v_seed, k)

        # forwarding probabilities: share outright, or verify and the detector raises no flag
        T_t = pi_s + pi_v * (1.0 - fpr[:, None])
        T_f = pi_s + pi_v * (1.0 - tpr[:, None])
        Ts_t = pi_s_seed + pi_v_seed * (1.0 - fpr[:, None])
        Ts_f = pi_s_seed + pi_v_seed * (1.0 - tpr[:, None])

        out_t = percolation_outputs(k, pk, qk, seed_pk, T_t, Ts_t)
        out_f = percolation_outputs(k, pk, qk, seed_pk, T_f, Ts_f)

        dSt = out_t["size"] - E_St
        dSf = out_f["size"] - E_Sf
        if prev_dSt is not None:
            flip = (np.sign(dSt) * np.sign(prev_dSt) < 0) | (np.sign(dSf) * np.sign(prev_dSf) < 0)
            damp = np.where(flip, np.maximum(damp * 0.5, DAMP_MIN), damp)
        prev_dSt, prev_dSf = dSt, dSf

        E_St_new = E_St + damp * dSt
        E_Sf_new = E_Sf + damp * dSf
        pv_bar_new = pv_bar + damp * ((qk[None, :] * pi_v).sum(axis=1) - pv_bar)

        resid_pts = np.maximum(np.abs(dSt), np.abs(dSf)) / N_AGENTS
        delta = max(float(resid_pts.max()), np.max(np.abs(theta_new - theta_bar)))
        step = float((damp * resid_pts).max())
        E_St, E_Sf, pv_bar, theta_bar = E_St_new, E_Sf_new, pv_bar_new, theta_new
        if verbose and (it + 1) % 10 == 0:
            print(f"  outer {it + 1}: delta={delta:.2e} step={step:.2e}", flush=True)
        if delta < OUTER_TOL or step < STEP_TOL:
            n_iter = it + 1
            break

    p_pool = p_fake
    return {
        "true_reach": E_St / N_AGENTS,
        "fake_reach": E_Sf / N_AGENTS,
        "R_true": out_t["R"], "R_fake": out_f["R"],
        "f_true": out_t["f"], "f_fake": out_f["f"],
        "ignition_true": out_t["ignition"], "ignition_fake": out_f["ignition"],
        "ignition_pooled": (1.0 - p_pool) * out_t["ignition"] + p_pool * out_f["ignition"],
        "takeoff_fake": out_f["takeoff"],
        "theta_believed": theta_bar,
        "outer_iterations": n_iter,
        "final_max_delta": float(resid_pts.max()),
        "n_points_residual_gt_1e-6": int((resid_pts > 1e-6).sum()),
        "n_points_residual_gt_1e-3": int((resid_pts > 1e-3).sum()),
    }


def _solve_chunk(payload):
    design, k, pk, qk, seed_pk = payload
    return solve_config(design, k, pk, qk, seed_pk, verbose=False)


N_CHUNKS = 16                # fixed, so results do not depend on the machine's core count


def solve_config_parallel(design, k, pk, qk, seed_pk, n_jobs):
    """Split the design points into N_CHUNKS fixed chunks solved in worker processes
    (the fixed point is per-point independent; a fixed chunking keeps the frozen
    near-critical values machine-independent). Per-point arrays concatenate; the
    convergence scalars merge as max (iterations, delta) and sum (counts)."""
    if n_jobs <= 1:
        return solve_config(design, k, pk, qk, seed_pk)
    from concurrent.futures import ProcessPoolExecutor
    idx_chunks = np.array_split(np.arange(len(design)), N_CHUNKS)
    payloads = [(design.iloc[idx], k, pk, qk, seed_pk) for idx in idx_chunks]
    with ProcessPoolExecutor(max_workers=min(n_jobs, N_CHUNKS)) as ex:
        parts = list(ex.map(_solve_chunk, payloads))
    merged = {}
    for key, v in parts[0].items():
        if isinstance(v, np.ndarray):
            merged[key] = np.concatenate([p[key] for p in parts])
        elif key.startswith("n_points"):
            merged[key] = sum(p[key] for p in parts)
        else:
            merged[key] = max(p[key] for p in parts)
    return merged


# ------------------------------------------------------------------ comparison
def compare(mf, obs_fake, obs_true):
    """Correlation/bias summary of MF predictions against observed design-point means."""
    def block(pred, obs):
        pred, obs = np.asarray(pred), np.asarray(obs)
        r = float(np.corrcoef(pred, obs)[0, 1])
        rho = float(pd.Series(pred).corr(pd.Series(obs), method="spearman"))
        return {"pearson_r": r, "spearman_rho": rho,
                "mae": float(np.mean(np.abs(pred - obs))),
                "bias_mf_minus_abm": float(np.mean(pred - obs)),
                "mean_mf": float(np.mean(pred)), "mean_abm": float(np.mean(obs))}

    res = {"fake_reach": block(mf["fake_reach"], obs_fake),
           "true_reach": block(mf["true_reach"], obs_true),
           "veracity_differential": block(mf["true_reach"] - mf["fake_reach"],
                                          obs_true - obs_fake)}
    sub = mf["R_fake"] < 1.0
    res["fake_reach_subcritical_R<1"] = block(mf["fake_reach"][sub], obs_fake[sub])
    res["fake_reach_supercritical_R>=1"] = block(mf["fake_reach"][~sub], obs_fake[~sub])
    res["n_subcritical"] = int(sub.sum())
    return res


def threshold_profile(R, obs, edges):
    """Observed mean reach within branching-ratio bins (the transition check)."""
    idx = np.digitize(R, edges)
    prof = []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() == 0:
            continue
        prof.append({"R_lo": float(edges[b - 1]), "R_hi": float(edges[b]),
                     "n": int(m.sum()), "obs_mean_reach": float(obs[m].mean()),
                     "mf_R_mean": float(R[m].mean())})
    return prof


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("degree ensemble ...", flush=True)
    k, pk, qk, hub_pk, deg_stats = degree_ensemble()
    print(f"  <k>={deg_stats['mean_degree']:.3f}  kmax={deg_stats['max_degree']}  "
          f"hub pool <k>={deg_stats['hub_pool_mean_degree']:.2f}")

    cols = ["global_sim_id", "design_id", *data_io.FACTORS, "avg_fake_cascade", "avg_true_cascade"]
    base, infl = data_io.load_seeding_pairs(cols)
    factors = data_io.FACTORS
    gb = base.groupby("design_id")
    design = gb[factors].first().reset_index()
    assert len(design) == data_io.N_DESIGNS
    # design factors are identical across the 30 reps and across configurations
    assert (gb[factors].nunique().values == 1).all()
    obs = {
        "base_fake": gb["avg_fake_cascade"].mean().values,
        "base_true": gb["avg_true_cascade"].mean().values,
        "infl_fake": infl.groupby("design_id")["avg_fake_cascade"].mean().values,
        "infl_true": infl.groupby("design_id")["avg_true_cascade"].mean().values,
    }

    n_jobs = max(1, min(14, (os.cpu_count() or 2) - 2))
    print(f"solving mean field: baseline configuration ({n_jobs} workers) ...", flush=True)
    mf_base = solve_config_parallel(design, k, pk, qk, pk, n_jobs)     # uniform seed = degree dist
    print(f"  done in <= {mf_base['outer_iterations']} outer iterations")
    print(f"solving mean field: influencer configuration ({n_jobs} workers) ...", flush=True)
    mf_infl = solve_config_parallel(design, k, pk, qk, hub_pk, n_jobs)  # hub seed pool
    print(f"  done in <= {mf_infl['outer_iterations']} outer iterations")

    cmp_base = compare(mf_base, obs["base_fake"], obs["base_true"])
    cmp_infl = compare(mf_infl, obs["infl_fake"], obs["infl_true"])

    # seeding differential (influencer - baseline) per point and per prevalence bin
    d_mf = mf_infl["fake_reach"] - mf_base["fake_reach"]
    d_abm = obs["infl_fake"] - obs["base_fake"]
    pbin = np.digitize(design["p_fake"].values, data_io.PFAKE_EDGES)
    diff_profile = []
    for b in range(1, len(data_io.PFAKE_EDGES)):
        m = pbin == b
        diff_profile.append({
            "p_fake_lo": float(data_io.PFAKE_EDGES[b - 1]),
            "p_fake_hi": float(data_io.PFAKE_EDGES[b]), "n": int(m.sum()),
            "mf_diff": float(d_mf[m].mean()), "abm_diff": float(d_abm[m].mean()),
            "mf_ignition_diff": float((mf_infl["ignition_fake"] - mf_base["ignition_fake"])[m].mean()),
        })

    # calibration: believed truth rate vs environment truth rate (slope, intercept)
    def calib(mf):
        x = 1.0 - design["p_fake"].values
        y = mf["theta_believed"]
        slope, intercept = np.polyfit(x, y, 1)
        return {"slope": float(slope), "intercept": float(intercept),
                "mean_overoptimism": float(np.mean(y - x))}

    # design-average ignition vs the exact decomposition's published rates
    sv_ev_path = os.path.join(data_io.SVDECOMP_DIR, "sv_evidence.json")
    sv_ignition = None
    if os.path.exists(sv_ev_path):
        with open(sv_ev_path) as fh:
            sv = json.load(fh)
        ign = sv.get("paired", {}).get("ignition_rate", {})
        sv_ignition = {kk: ign[kk] for kk in ("mean_baseline", "mean_influencer")
                       if kk in ign} or None

    evidence = {
        "meta": {
            "script": "analysis/mean_field.py", "date": "2026-07-09",
            "n_design_points": int(len(design)), "reps_per_point": data_io.REPS,
            "c_eff_cascades": C_EFF, "prior_strength": PRIOR_STRENGTH,
            "outer_damping": DAMP,
            "grounding": ["newman_2002", "newman_strogatz_watts_2001",
                          "pastor_satorras_vespignani_2001", "melnik_2011",
                          "kenah_robins_2007"],
        },
        "degree_ensemble": deg_stats,
        "convergence": {cfg: {kk: mf[kk] for kk in
                              ("outer_iterations", "final_max_delta",
                               "n_points_residual_gt_1e-6", "n_points_residual_gt_1e-3")}
                        for cfg, mf in (("baseline", mf_base), ("influencer", mf_infl))},
        "comparison_baseline": cmp_base,
        "comparison_influencer": cmp_infl,
        "threshold_profile_baseline_fake": threshold_profile(
            mf_base["R_fake"], obs["base_fake"],
            np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 20.0])),
        "seeding_differential_profile": diff_profile,
        "seeding_differential_corr": float(np.corrcoef(d_mf, d_abm)[0, 1]),
        "ignition": {
            "mf_design_avg_pooled_baseline": float(mf_base["ignition_pooled"].mean()),
            "mf_design_avg_pooled_influencer": float(mf_infl["ignition_pooled"].mean()),
            "mf_design_avg_fake_baseline": float(mf_base["ignition_fake"].mean()),
            "mf_design_avg_fake_influencer": float(mf_infl["ignition_fake"].mean()),
            "abm_published_pooled": sv_ignition,
        },
        "calibration_baseline": calib(mf_base),
        "calibration_influencer": calib(mf_infl),
        "seed_invariance": {
            "note": "R, eps and f depend on beliefs and the non-seed decision layer "
                    "only; given beliefs they are identical across seeding "
                    "configurations. The residual difference below is the "
                    "belief-feedback (theta_exp) channel alone.",
            "max_abs_f_fake_diff": float(np.max(np.abs(mf_infl["f_fake"] - mf_base["f_fake"]))),
            "mean_abs_f_fake_diff": float(np.mean(np.abs(mf_infl["f_fake"] - mf_base["f_fake"]))),
        },
    }

    with open(os.path.join(OUT_DIR, "mean_field_evidence.json"), "w") as fh:
        json.dump(evidence, fh, indent=2)

    np.savez_compressed(
        os.path.join(OUT_DIR, "mf_points.npz"),
        design_id=design["design_id"].values, p_fake=design["p_fake"].values,
        rationality=design["rationality"].values,
        obs_base_fake=obs["base_fake"], obs_base_true=obs["base_true"],
        obs_infl_fake=obs["infl_fake"], obs_infl_true=obs["infl_true"],
        mf_base_fake=mf_base["fake_reach"], mf_base_true=mf_base["true_reach"],
        mf_infl_fake=mf_infl["fake_reach"], mf_infl_true=mf_infl["true_reach"],
        R_base_fake=mf_base["R_fake"], R_base_true=mf_base["R_true"],
        mf_base_theta=mf_base["theta_believed"], mf_infl_theta=mf_infl["theta_believed"],
        mf_base_ignition_fake=mf_base["ignition_fake"],
        mf_infl_ignition_fake=mf_infl["ignition_fake"],
    )

    print(json.dumps({kk: evidence[kk] for kk in
                      ("comparison_baseline", "ignition", "calibration_baseline",
                       "seeding_differential_corr")}, indent=2))
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
