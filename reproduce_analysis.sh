#!/usr/bin/env bash
# reproduce_analysis.sh -- regenerate every manuscript figure, table, and quantitative claim
# from the bundled frozen intermediate data (NOT from the raw cascade sweeps).
#
# Run from the repository root:        bash reproduce_analysis.sh
# All outputs are written inside the repository (analysis/figures/, analysis/runs/*,
# sobol/results/*), and every output location is gitignored, so a reproduction run leaves
# the working tree clean.
#
# Two Python environments are needed (see requirements.txt):
#   PY_MAIN  : the main pinned stack (numpy>=2, pandas, scipy, statsmodels,
#              matplotlib, seaborn, networkx, pyarrow, diptest, psutil).
#   PY_SOBOL : the SALib environment (SALib==1.4.8, numpy==1.26.4, pandas==1.5.3,
#              matplotlib==3.9.4). SALib 1.4.8 predates NumPy 2, hence the split.
# Defaults assume `python` on PATH points at the main stack. With conda:
#   PY_MAIN="conda run -n <main-env> --no-capture-output python" \
#   PY_SOBOL="conda run -n <salib-env> --no-capture-output python" bash reproduce_analysis.sh
# If PY_SOBOL is unset, the Sobol step (the last section) is skipped with a notice.
#
# Pipeline shape: every analysis script is pure-compute (writes an evidence JSON / .npz);
# analysis/make_figures.py then renders the figures from those in one pass. The compute
# steps therefore run first and make_figures runs last (before the independent Sobol step).

set -euo pipefail
cd "$(dirname "$0")"

PY_MAIN="${PY_MAIN:-python}"
PY_SOBOL="${PY_SOBOL:-}"

hdr() { echo; echo "=================================================================="; echo "== $1"; echo "=================================================================="; }

# ------------------------------------------------------------------------------
# SCOPE: every stage here is a pure `reduce` from bundled bounded-memory accumulators --
# no HPC, no multi-GB arrow loads. build_evidence.py produces the 3 evidence JSONs and the
# descriptive-figure inputs by reducing from the bundled node/cascade cache
# (analysis/runs/nodecache/, ~80 MB) plus the small per-sim simulations.arrow
# (bundled, ~63 MB per configuration). The ~174 GB raw nodes/cascades.arrow are needed
# ONLY when the cache is re-streamed locally via build_evidence.py's stream subcommands
# (STREAM_FROM_ARROW=1, see step 5a).
# ------------------------------------------------------------------------------

# OPTIONAL -- full end-to-end regeneration from the Julia model (HPC-scale; OFF by default).
# Re-runs the model to regenerate the raw sweeps the bundled intermediates were derived from.
# Needs Julia 1.12.x; WRITE_FULL emits ~174 GB of *.arrow and the full design is 16,384x30 sims.
# Regenerated sweeps reproduce the published statistics to reported precision, not bit-identically
# (byte-identity comes from the bundled data below). Enable with RUN_JULIA_SWEEP=1.
if [ "${RUN_JULIA_SWEEP:-}" = "1" ]; then
  hdr "0  (optional) Julia model -> raw sweeps + identification check  [needs Julia 1.12.x]"
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  WRITE_FULL=true OUT_TAG=baseline   julia --project=. -t auto run_experiment_baseline.jl     # -> data/sweep_<ts>_baseline
  WRITE_FULL=true OUT_TAG=influencer julia --project=. -t auto run_experiment_influencers.jl  # -> data/sweep_<ts>_influencer
  julia --project=. analysis/influence_identification.jl   # identification-robustness check
  echo "NOTE: fresh sweeps land in new data/sweep_<timestamp>_* dirs; point the stream steps"
  echo "      (STREAM_FROM_ARROW=1 + ARROW_BASELINE/ARROW_INFLUENCER) at them, then re-run."
else
  echo "(optional Julia end-to-end sweep regeneration skipped; set RUN_JULIA_SWEEP=1 to enable -- HPC-scale, needs Julia)"
fi

# ---- Compute steps (pure-compute; each writes an evidence JSON that make_figures reads) -------

hdr "1/7  tost_blocks.py  (block-clustered fake-news reach equivalence + profile;
      VERIFIER cross-checks must report all_pass:true)
      [main env]  -> analysis/runs/tost_blocks/tost_blocks_evidence.json"
$PY_MAIN analysis/tost_blocks.py

hdr "2/7  sv_decomposition.py reduce  (two-part structural-virality decomposition
      from the bundled exact accumulators; pass criteria V1-V7 must PASS)
      [main env]  -> analysis/runs/svdecomp/sv_evidence.json"
$PY_MAIN analysis/sv_decomposition.py reduce --out analysis/runs/svdecomp \
    --baseline-csv data/baseline_svd/simulations.csv.gz \
    --influencer-csv data/influencer_svd/simulations.csv.gz

hdr "3/7  robustness/switchover_audit.py reduce  (exact ignition-vs-conditional
      decomposition of the seeding contrast per prevalence bin, from the bundled
      accumulators; gates S1-S7 asserted)
      [main env]  -> analysis/runs/switchover_audit/switchover_audit_evidence.json"
$PY_MAIN analysis/robustness/switchover_audit.py reduce \
    --out analysis/runs/switchover_audit

hdr "4/7  robustness/loss_edge_check.py  (near-edge robustness of the l>g payoff
      asymmetry; reproduces the published means by assertion)
      [main env]  -> analysis/runs/loss_edge/"
$PY_MAIN analysis/robustness/loss_edge_check.py

hdr "4b   mean_field.py  (degree-based mean-field benchmark: belief steady state +
      percolation per design point; backs the mean-field subsection and Appendix A)
      [main env]  -> analysis/runs/mean_field/"
$PY_MAIN analysis/mean_field.py

hdr "4c   replication_justification.py  (MSER-5 warm-up truncation from the bundled probe
      run + required-replications check; backs the in-text replication numbers)
      [main env]  -> analysis/runs/replication_justification/"
$PY_MAIN analysis/replication_justification.py

# ---- build_evidence.py: the 3 seeding-comparison evidence JSONs (eda/compare/influencer) --------
# DEFAULT: reduce from the bundled bounded-memory node/cascade cache -- reads only the small
# simulations.arrow + the ~80 MB cache (analysis/runs/nodecache/); < 2 GB RAM, no HPC.
# OPTIONAL full re-stream: set STREAM_FROM_ARROW=1 (and ARROW_BASELINE / ARROW_INFLUENCER to the
# local raw WRITE_FULL sweep dirs) to regenerate the cache from the ~174 GB nodes/cascades.arrow
# first via build_evidence.py's bounded-memory stream subcommands, then reduce. The raw arrows
# are NOT bundled.
CACHE=analysis/runs/nodecache
BASE_SWEEP="${ARROW_BASELINE:-data/baseline}"
INFL_SWEEP="${ARROW_INFLUENCER:-data/influencer}"
if [ "${STREAM_FROM_ARROW:-}" = "1" ]; then
  hdr "5a (optional) re-stream the node/cascade cache from raw *.arrow  [bounded memory, no HPC]"
  $PY_MAIN analysis/build_evidence.py stream            --sweep "$BASE_SWEEP" --label baseline   --out "$CACHE"
  $PY_MAIN analysis/build_evidence.py stream            --sweep "$INFL_SWEEP" --label influencer --out "$CACHE"
  $PY_MAIN analysis/build_evidence.py stream-cascades   --sweep "$BASE_SWEEP" --label baseline   --out "$CACHE"
  $PY_MAIN analysis/build_evidence.py stream-cascades   --sweep "$INFL_SWEEP" --label influencer --out "$CACHE"
  $PY_MAIN analysis/build_evidence.py sample-cascade-sizes --baseline "$BASE_SWEEP" --influencer "$INFL_SWEEP" --out "$CACHE"
  $PY_MAIN analysis/build_evidence.py stream-influencer --sweep "$INFL_SWEEP" --out "$CACHE"
else
  echo "(cache re-streaming skipped; reducing from the bundled cache. Set STREAM_FROM_ARROW=1 + ARROW_BASELINE/ARROW_INFLUENCER to regenerate from raw *.arrow.)"
fi
hdr "5/7  build_evidence.py  (eda/compare/influencer evidence JSONs from the node cache)
      [main env]  -> analysis/{eda,compare,influencer_node}_evidence.json"
$PY_MAIN analysis/build_evidence.py eda        --baseline "$BASE_SWEEP"                            --cache "$CACHE"
$PY_MAIN analysis/build_evidence.py compare    --baseline "$BASE_SWEEP" --influencer "$INFL_SWEEP" --cache "$CACHE"
$PY_MAIN analysis/build_evidence.py influencer                                                     --cache "$CACHE"

# ---- Figures (single entry point; renders every figure whose inputs are present) ----
hdr "6/7  make_figures.py  (renders the manuscript figures from the evidence built above)
      [main env]  -> analysis/figures/"
$PY_MAIN analysis/make_figures.py

hdr "7/7  Sobol sensitivity analysis  [SALib env -- requires PY_SOBOL]
      analyze.py (per-arm indices) + analyze_svd.py (ignition-rate indices; the cond_sv
      rows are empty by construction, matching the reference results)
      -> sobol/results/{baseline_sv,influencer_sv,baseline_svd,influencer_svd}/"
if [ -n "$PY_SOBOL" ]; then
  # --welfare-npz: the bundled sweep CSVs carry a superseded min-shift sen_welfare;
  # the flag replaces it with the RSV values from the bundled node cache.
  $PY_SOBOL sobol/analyze.py --sweep data/baseline_svd \
      --welfare-npz "$CACHE/acc_nodes_baseline.npz" \
      --problem sobol/problem.json --out sobol/results/baseline_sv
  $PY_SOBOL sobol/analyze.py --sweep data/influencer_svd \
      --welfare-npz "$CACHE/acc_nodes_influencer.npz" \
      --problem sobol/problem.json --out sobol/results/influencer_sv
  $PY_SOBOL sobol/analyze_svd.py --sweep data/baseline_svd \
      --problem sobol/problem.json --out sobol/results/baseline_svd
  $PY_SOBOL sobol/analyze_svd.py --sweep data/influencer_svd \
      --problem sobol/problem.json --out sobol/results/influencer_svd
else
  echo "SKIPPED: set PY_SOBOL to the SALib/numpy<2 environment's python to run this step."
fi

echo
echo "reproduce_analysis.sh: all requested steps completed."
