#!/bin/bash
# =============================================================================
# Snellius (SURF) SLURM job: fake-news-percolation BASELINE (uniform-seeding) sweep
# Runs run_experiment_baseline.jl (the Sobol/Saltelli sweep; uniform-random cascade seeding)
# on ONE full Genoa node (192 cores / 384 GiB), shared-memory multithreaded.
#
# ---- ONE-TIME SETUP (run on a LOGIN node, once) -----------------------------
#   1. Copy the model + project to your home, e.g.:
#        ~/fake-news-percolation/run_experiment_influencers.jl
#        ~/fake-news-percolation/Project.toml      (optional; created by step 3)
#        ~/fake-news-percolation/run_snellius.sh   (this file)
#        ~/fake-news-percolation/sobol/            (scripts + design.csv from step 1b)
#   1b. The Julia runner READS sobol/design.csv (the SALib Saltelli design). Generate it
#       once with Python + SALib (e.g. locally in the `ABM` conda env, then copy sobol/):
#          python sobol/make_design.py --N 1024     # -> sobol/design.csv (+ problem.json)
#       It is small and deterministic (seed=42), so no Python is needed on the node.
#   2. Make Julia available. CONFIRM what your Snellius offers first:
#        module avail Julia
#      then pick ONE of the options in the "Julia environment" block below.
#   3. Install the dependencies into the project ONCE (login node has internet):
#        cd ~/fake-news-percolation
#        julia --project=. -e 'using Pkg; Pkg.add(["Graphs","DataStructures",\
#          "StatsBase","ProgressMeter","DataFrames","Arrow","DelimitedFiles"]); Pkg.precompile()'
#      (Random, Statistics, Dates are stdlib; DelimitedFiles is a bundled stdlib on
#       Julia 1.12.x but is declared here so the project stays self-contained.)
#   4. Submit:   sbatch run_snellius.sh <true> if per-node/edge/cascade files should be written (~83GB)
#      Monitor:  squeue -u $USER       Log: slurm-<jobid>.out
#
# ---- COST / RUNTIME (estimate) ----------------------------------------------
#   1 SBU = 1 core-hour; a full Genoa node bills all 192 cores x ACTUAL runtime
#   (not the --time cap). The default Sobol sweep is ~430,080 sims, but write_full=false skips the heavy per-node/edge/cascade Arrow files
#   (no ~9 GB write) and emits only a small simulations.{arrow,csv}. Runtime is not
#   yet calibrated; the --time cap below is deliberately generous (it is only a cap;
#   SBU bills actual runtime). Check the elapsed time in slurm-<jobid>.out.
# =============================================================================

#SBATCH --job-name=deepfake_baseline
# Genoa thin node = 192 cores / 384 GiB; take the whole node (exclusive) for max RAM.
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ---- Julia environment: -----
module load EESSI/2025.06 && module load Julia/1.12.2        # EESSI stack (SURF-documented on Snellius); run 'module avail Julia' to confirm the version
julia --version

PROJ="$HOME/fake-news-percolation"
# Large sweep output (~84 GB with WRITE_FULL) goes to scratch, not $HOME: the SURF home quota is
# only 200 GiB and SURF recommends scratch for "voluminous job I/O"; scratch-shared = 8 TiB.
# CAVEAT: scratch-shared auto-deletes files unmodified for 14 days and is NOT backed up -- run the
# analysis (or copy results back to $HOME) promptly. (https://servicedesk.surf.nl Snellius filesystems)
RUN_DIR="/scratch-shared/$USER/fake-news-percolation"

# one Julia thread per allocated core (reads JULIA_NUM_THREADS automatically)
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "host=$(hostname) nproc=$(nproc) JULIA_NUM_THREADS=$JULIA_NUM_THREADS start=$(date)"

# fail fast if the Saltelli design is missing (see ONE-TIME SETUP step 1b)
if [[ ! -f "$PROJ/sobol/design.csv" ]]; then
    echo "ERROR: $PROJ/sobol/design.csv not found. Generate it first with:"
    echo "       python sobol/make_design.py --N 1024   (needs Python + SALib)"
    exit 1
fi

# Stage the design on scratch and cd there: the runner reads sobol/design.csv and writes
# data/<sweep> RELATIVE TO CWD, so this keeps the large output on scratch instead of $HOME.
mkdir -p "$RUN_DIR/sobol"
cp -f "$PROJ/sobol/design.csv" "$RUN_DIR/sobol/design.csv"
cd "$RUN_DIR"                        # output -> $RUN_DIR/data/sweep_<timestamp>_<tag>

# WRITE_FULL: enable the heavy per-cascade/node/edge Arrow files. Accept it from a positional
# arg ($1, always delivered to the job script) or the environment, and export it so the julia
# child sees it. Robust submit forms:  `sbatch run_snellius.sh true`  or  `WRITE_FULL=true sbatch run_snellius.sh`.
export WRITE_FULL="${1:-${WRITE_FULL:-false}}"
echo "WRITE_FULL (job script) = '$WRITE_FULL'"

# OUT_TAG: label the output dir (data/sweep_<ts>_<tag>) so a parallel baseline+influencer pair
# never collides on the minute-resolution timestamp. The runner reads it from the environment.
export OUT_TAG="baseline"
echo "OUT_TAG (job script) = '$OUT_TAG'"

julia --project="$PROJ" "$PROJ/run_experiment_baseline.jl"

echo "done=$(date)  WRITE_FULL=$WRITE_FULL  output in $RUN_DIR/data/sweep_*/"
echo "Next (off-node, needs Python + SALib): python sobol/analyze.py"
