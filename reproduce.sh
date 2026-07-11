#!/usr/bin/env bash
# reproduce.sh -- resource-adaptive entry point for reproducing the manuscript.
#
# Tiers:
#   render      re-render every figure from the evidence already present   (minutes)
#   verify      rebuild all evidence from the bundled data, byte-checked   (tens of minutes)
#               (delegates to reproduce_analysis.sh; set PY_SOBOL for the Sobol tables)
#   smoke       re-SIMULATE at N=8    x R=2  and run the full analysis on it (pipeline check)
#   quick       re-simulate  at N=128  x R=2  (flagship TOST + factor ranking)
#   overnight   re-simulate  at N=1024 x R=2  (full design resolution, all bins, S2)
#   stochastic  re-simulate  at N=1024 x R=8  (adds stochastic-Sobol factor attributions)
#
# The model is never changed: scaled tiers shrink only the sweep breadth (Saltelli base
# sample N x replicates R); network size, cascade counts, and burn-in stay at the
# published values. Scaled runs therefore give statistically valid but wider-CI numbers;
# they are a directional replication check, not the manuscript's designed power. Every
# scaled run is self-contained under scaled_runs/<tier>_<tag>/ and touches nothing else.
#
# Usage:  bash reproduce.sh                 # detect resources, suggest a tier, confirm
#         bash reproduce.sh --tier quick    # run a specific tier
#         bash reproduce.sh --tier smoke --yes --tag myrun
# Env:    PY_MAIN / PY_SOBOL as in reproduce_analysis.sh; TIME_BUDGET (seconds, default 7200).
# On Windows run from Git Bash (ships with Git for Windows).

set -euo pipefail
cd "$(dirname "$0")"

PY_MAIN="${PY_MAIN:-python}"
PY_SOBOL="${PY_SOBOL:-}"
TIME_BUDGET="${TIME_BUDGET:-7200}"

# Seconds per simulation per thread, measured on the fully loaded 192-thread production
# nodes (snellius_logs/: 491,520 sims in ~4.6 h). Deliberately conservative: a 16-thread
# desktop completed the smoke tier's 512 sims in under a minute including Julia start-up,
# i.e. several times faster per thread than a saturated cluster node. Every scaled run
# prints its own measured constant (wall_seconds * threads / total_sims) after step 1;
# treat the table below as upper bounds.
SPT=6.6

THREADS=$( (command -v nproc >/dev/null && nproc) || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4 )

TIER=""; YES=0; TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --yes)  YES=1; shift ;;
    --tag)  TAG="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

eta_h() {  # eta_h <N> <R>  -> hours for both configurations on $THREADS threads
  awk -v n="$1" -v r="$2" -v t="$THREADS" -v s="$SPT" 'BEGIN{printf "%.1f", 32*n*r*s/t/3600}'
}

preset() {  # preset <tier> -> "N R" (empty for non-simulation tiers)
  case "$1" in
    smoke)      echo "8 2" ;;
    quick)      echo "128 2" ;;
    overnight)  echo "1024 2" ;;
    stochastic) echo "1024 8" ;;
    *)          echo "" ;;
  esac
}

show_table() {
  echo "Detected $THREADS hardware threads. ETAs at the measured $SPT s/sim/thread:"
  echo
  echo "  tier        N     R   sims(total)  est. wall     checks"
  echo "  render      -     -   -            minutes       figures from present evidence"
  echo "  verify      -     -   -            ~1 h          every manuscript number, exact"
  printf "  smoke       8     2   %-12s ~%s h         pipeline end-to-end only\n"      "$((32*8*2))"     "$(eta_h 8 2)"
  printf "  quick       128   2   %-12s ~%s h         flagship TOST, S1/ST ranking\n"  "$((32*128*2))"   "$(eta_h 128 2)"
  printf "  overnight   1024  2   %-12s ~%s h         + all bins, S2, switchover\n"    "$((32*1024*2))"  "$(eta_h 1024 2)"
  printf "  stochastic  1024  8   %-12s ~%s h         + stochastic-Sobol attributions\n" "$((32*1024*8))" "$(eta_h 1024 8)"
  echo
}

suggest_tier() {
  local best="smoke"
  for t in quick overnight stochastic; do
    set -- $(preset "$t")
    local secs; secs=$(awk -v n="$1" -v r="$2" -v tt="$THREADS" -v s="$SPT" 'BEGIN{printf "%d", 32*n*r*s/tt}')
    [ "$secs" -le "$TIME_BUDGET" ] && best="$t"
  done
  echo "$best"
}

if [ -z "$TIER" ]; then
  show_table
  SUGGEST=$(suggest_tier)
  echo "Suggested simulation tier within TIME_BUDGET=${TIME_BUDGET}s: $SUGGEST"
  echo "(tiers render/verify need no simulation; pick them for exact manuscript numbers)"
  if [ "$YES" = 1 ]; then
    TIER="$SUGGEST"
  else
    printf "Tier to run [%s]: " "$SUGGEST"
    read -r TIER
    TIER="${TIER:-$SUGGEST}"
  fi
fi

case "$TIER" in
  render)
    exec $PY_MAIN analysis/make_figures.py
    ;;
  verify)
    exec bash reproduce_analysis.sh
    ;;
  smoke|quick|overnight|stochastic)
    set -- $(preset "$TIER"); N=$1; R=$2
    ;;
  *) echo "unknown tier: $TIER"; exit 2 ;;
esac

RUN="scaled_runs/${TIER}_${TAG:-$(date +%Y%m%d_%H%M%S)}"
[ -e "$RUN" ] && { echo "$RUN already exists; pass a fresh --tag"; exit 2; }
mkdir -p "$RUN"
ABS_RUN=$(cd "$RUN" && pwd)
echo "== scaled tier '$TIER' (N=$N, R=$R, ~$(eta_h "$N" "$R") h on $THREADS threads) -> $RUN"
if [ "$YES" != 1 ]; then
  printf "Proceed? [y/N] "
  read -r ok; [ "$ok" = y ] || [ "$ok" = Y ] || exit 0
fi
T0=$SECONDS

hdr() { echo; echo "== $1"; }

hdr "1/6 Julia scaled sweeps (production model, WRITE_FULL; one process per configuration)"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
FNP_RUN_ROOT="$ABS_RUN" FNP_N="$N" FNP_REPS="$R" julia --project=. -t auto run_experiment_scaled.jl baseline
FNP_RUN_ROOT="$ABS_RUN" FNP_N="$N" FNP_REPS="$R" julia --project=. -t auto run_experiment_scaled.jl influencer
SIM_SECS=$((SECONDS - T0))
TOTAL_SIMS=$((32 * N * R))
echo "== simulation wall ${SIM_SECS}s for $TOTAL_SIMS sims on $THREADS threads" \
     "($(awk -v w="$SIM_SECS" -v t="$THREADS" -v n="$TOTAL_SIMS" 'BEGIN{printf "%.2f", w*t/n}') s/sim/thread)"

hdr "2/6 avg_structural_virality column (sobol/augment_sv.py per configuration)"
$PY_MAIN sobol/augment_sv.py "$RUN/data/baseline"   "$RUN/data/baseline_svd"
$PY_MAIN sobol/augment_sv.py "$RUN/data/influencer" "$RUN/data/influencer_svd"

export FNP_RUN_ROOT="$ABS_RUN"   # every analysis path/constant now derives from the run

hdr "3/6 stream the accumulators from the scaled sweeps"
CACHE="$RUN/runs/nodecache"
$PY_MAIN analysis/build_evidence.py stream            --sweep "$RUN/data/baseline"   --label baseline   --out "$CACHE"
$PY_MAIN analysis/build_evidence.py stream            --sweep "$RUN/data/influencer" --label influencer --out "$CACHE"
$PY_MAIN analysis/build_evidence.py stream-cascades   --sweep "$RUN/data/baseline"   --label baseline   --out "$CACHE"
$PY_MAIN analysis/build_evidence.py stream-cascades   --sweep "$RUN/data/influencer" --label influencer --out "$CACHE"
$PY_MAIN analysis/build_evidence.py sample-cascade-sizes --baseline "$RUN/data/baseline" --influencer "$RUN/data/influencer" --out "$CACHE"
$PY_MAIN analysis/build_evidence.py stream-influencer --sweep "$RUN/data/influencer" --out "$CACHE"
$PY_MAIN analysis/sv_decomposition.py stream --sweep "$RUN/data/baseline"   --label baseline   --out "$RUN/runs/svdecomp"
$PY_MAIN analysis/sv_decomposition.py stream --sweep "$RUN/data/influencer" --label influencer --out "$RUN/runs/svdecomp"
$PY_MAIN analysis/robustness/switchover_audit.py stream --sweep "$RUN/data/baseline"   --label baseline   --out "$RUN/runs/switchover_audit"
$PY_MAIN analysis/robustness/switchover_audit.py stream --sweep "$RUN/data/influencer" --label influencer --out "$RUN/runs/switchover_audit"

hdr "4/6 evidence (same pure-compute chain as reproduce_analysis.sh, scaled inputs)"
$PY_MAIN analysis/build_evidence.py eda        --baseline "$RUN/data/baseline" --cache "$CACHE"
$PY_MAIN analysis/build_evidence.py compare    --baseline "$RUN/data/baseline" --influencer "$RUN/data/influencer" --cache "$CACHE"
$PY_MAIN analysis/build_evidence.py influencer --cache "$CACHE"
$PY_MAIN analysis/tost_blocks.py
$PY_MAIN analysis/sv_decomposition.py reduce --out "$RUN/runs/svdecomp" \
    --baseline-csv "$RUN/data/baseline_svd/simulations.csv" \
    --influencer-csv "$RUN/data/influencer_svd/simulations.csv" \
    --write-sobol-csv   # appends ignition_rate/cond_sv, as bundled in the canonical CSVs
$PY_MAIN analysis/robustness/switchover_audit.py reduce --out "$RUN/runs/switchover_audit"
$PY_MAIN analysis/robustness/loss_edge_check.py
$PY_MAIN analysis/mean_field.py
$PY_MAIN analysis/replication_justification.py   # MSER block skips: the probe is a full-scale artifact

hdr "5/6 figures"
$PY_MAIN analysis/make_figures.py "$RUN"

hdr "6/6 Sobol indices [needs PY_SOBOL]"
if [ -n "$PY_SOBOL" ]; then
  # scaled sweeps are generated post-RSV, so sen_welfare is already correct (no --welfare-npz)
  $PY_SOBOL sobol/analyze.py     --sweep "$RUN/data/baseline_svd"   --problem "$RUN/problem.json" --out "$RUN/sobol_results/baseline"
  $PY_SOBOL sobol/analyze.py     --sweep "$RUN/data/influencer_svd" --problem "$RUN/problem.json" --out "$RUN/sobol_results/influencer"
  $PY_SOBOL sobol/analyze_svd.py --sweep "$RUN/data/baseline_svd"   --problem "$RUN/problem.json" --out "$RUN/sobol_results/baseline_svd"
  $PY_SOBOL sobol/analyze_svd.py --sweep "$RUN/data/influencer_svd" --problem "$RUN/problem.json" --out "$RUN/sobol_results/influencer_svd"
else
  echo "SKIPPED: set PY_SOBOL to the SALib/numpy<2 environment's python for the Sobol tables."
fi

echo
echo "reproduce.sh tier '$TIER' complete in $(( (SECONDS-T0) / 60 )) min; everything under $RUN"
echo "Scaled-run caveat: valid but wider-CI statistics; compare qualitatively against the"
echo "manuscript (factor ranking, TOST direction, switchover shape), not digit-for-digit."