#!/bin/bash
# =============================================================================
# Snellius (SURF) SLURM job: full 7-factor analysis of a paired baseline+influencer sweep.
# Pure Python (no Julia). The job runs the evidence pipeline and the figures:
#   1. eda        --baseline   <BASELINE>                          -> eda_evidence_7f.json  (uniform-sweep descriptives)
#   2. influencer --influencer <INFLUENCER>                        -> influencer_node_evidence_7f.json
#   3. compare    --baseline <BASELINE> --influencer <INFLUENCER>  -> compare_evidence_7f.json  (asserts the 1:1 pairing)
#   4. make_figures.py <run_dir>  -> every figure whose inputs are present: the nine evidence-JSON
#      figures (including the structural-virality row of the seeding forest); the five sweep-table figures are
#      skipped where the local sweep copies are absent.
# The three JSONs and the figures are then archived per run under analysis/runs/<tag>/.
#
# Paths. Sweep data are read from /scratch-shared/$USER/...; scripts, venv, and outputs live in
# ~/fake-news-percolation/. build_evidence.py resolves its output directory from __file__ (the
# JSONs land in ~/.../analysis/) and takes the sweeps via --baseline/--influencer, so the ~88 GB
# of sweep data is never copied into $HOME.
#
# Sizing. The pipeline is RAM-bound and effectively single-threaded, so parallelism cannot lower
# the cost. The heaviest step is compare, which reads both sweeps' nodes and cascades tables (the
# cascade projection is 7 columns including structural_virality, ~27.5 GB, for a ~55-60 GiB peak).
# --mem=96G buys ~55 billed cores on genoa (96/1.75) and covers that peak: MALLOC_TRIM returns
# freed memory between the two cascade reads, and 96G already carried the influencer step's
# ~55 GiB. If compare is OOM-killed, raise to 128G. Expected runtime is ~25-40 min, i.e. ~23-37
# SBU (roughly 1-2% of the ~1780 SBU that generated the two sweeps).
#
# ---- One-time setup (login node; has internet) ------------------------------
#   cd ~/fake-news-percolation
#   module load 2024 && module load Python/3.12.3-GCCcore-13.3.0   # confirm with: module avail Python (the toolchain suffix is required)
#   python -m venv ~/fnp-venv && source ~/fnp-venv/bin/activate
#   pip install --upgrade pip && pip install -r requirements.txt && deactivate
#   Submit (from ~/fake-news-percolation):  sbatch run_snellius_analysis.sh [BASELINE_DIR] [INFLUENCER_DIR]
#   Monitor: squeue -u $USER       Log: slurm-analysis-<jobid>.out
# =============================================================================

#SBATCH --job-name=fnp_analysis_7f
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G                 # ~55 billed cores on genoa (96/1.75); covers the ~55-60 GiB compare peak. Raise to 128G if OOM-killed.
#SBATCH --time=04:00:00           # generous cap; SBU charges actual elapsed time
#SBATCH --output=slurm-analysis-%j.out
#SBATCH --error=slurm-analysis-%j.err
# #SBATCH -A <YOUR_PROJECT>       # uncomment if your account requires a budget

set -euo pipefail

PROJ="$HOME/fake-news-percolation"; VENV="$HOME/fnp-venv"          # scripts, venv, and outputs ($HOME)
SCRATCH="/scratch-shared/$USER/fake-news-percolation/data"          # sweep data (scratch-shared)
# The two paired sweeps (same sobol/design.csv and seeds); override by passing directories as arguments.
BASELINE="${1:-$SCRATCH/sweep_2026_06_27_1641_baseline}"           # uniform-random seeding
INFLUENCER="${2:-$SCRATCH/sweep_2026_06_27_1633_influencer}"       # top-degree hub seeding
RUN_TAG="${SLURM_JOB_ID:-local}_$(date +%m-%d-%y_%H%M)"
RUN_OUT="$PROJ/analysis/runs/$RUN_TAG"

for d in "$BASELINE" "$INFLUENCER"; do
    [[ -f "$d/simulations.arrow" && -f "$d/nodes.arrow" && -f "$d/cascades.arrow" ]] \
        || { echo "ERROR: $d missing simulations/nodes/cascades.arrow (need a WRITE_FULL sweep)"; exit 1; }
done
[[ -f "$VENV/bin/activate" ]] || { echo "ERROR: venv missing: $VENV (run the one-time setup above)"; exit 1; }
echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK run_tag=$RUN_TAG start=$(date)"
echo "BASELINE=$BASELINE"
echo "INFLUENCER=$INFLUENCER"

# ---- Evidence pipeline (single-threaded pandas; RAM-bound) ------------------
module purge 2>/dev/null || true
module load 2024 && module load Python/3.12.3-GCCcore-13.3.0   # must match the module the venv was built with
source "$VENV/bin/activate"
# Cap the thread pools to the allocation and return freed memory to the OS between the two large
# cascade reads.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=0

BE="$PROJ/analysis/build_evidence.py"
# The three steps run sequentially in separate processes; each peaks at tens of GiB, so they must
# not be backgrounded.
python "$BE" eda        --baseline   "$BASELINE"                              # -> eda_evidence_7f.json
python "$BE" influencer --influencer "$INFLUENCER"                           # -> influencer_node_evidence_7f.json
python "$BE" compare    --baseline   "$BASELINE" --influencer "$INFLUENCER"  # -> compare_evidence_7f.json (asserts pairing)

# ---- Archive this run's three JSONs and figures under analysis/runs/$RUN_TAG/ ----
# The archive holds copies; the canonical JSONs stay in analysis/.
mkdir -p "$RUN_OUT"
cp -f "$PROJ/analysis/"{eda,influencer_node,compare}_evidence_7f.json "$RUN_OUT/"
{ echo "run_tag=$RUN_TAG"; echo "host=$(hostname)"; echo "date=$(date)";
  echo "baseline=$BASELINE"; echo "influencer=$INFLUENCER"; } > "$RUN_OUT/run_info.txt"
# make_figures.py writes every figure whose inputs are present into $RUN_OUT/figures_7factor/;
# failure is non-fatal because the JSONs are already archived.
python "$PROJ/analysis/make_figures.py" "$RUN_OUT" || echo "WARN: make_figures.py failed; JSONs are archived in $RUN_OUT"

echo "done=$(date)"
echo "Run archive (3 JSONs + figures_7factor/ + run_info.txt): $RUN_OUT/"
echo "Latest canonical evidence: $PROJ/analysis/{eda,influencer_node,compare}_evidence_7f.json"

# Not part of this job: the data-independent identification check.
#   module load EESSI/2025.06 && module load Julia/1.12.2
#   julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia --project=. analysis/influence_identification.jl
