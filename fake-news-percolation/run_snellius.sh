#!/bin/bash
# =============================================================================
# Snellius (SURF) SLURM job: faken-news-percolation influencer sweep
# Runs run_experiment_influencers.jl (the Sobol/Saltelli sweep; default ~430,080 sims)
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
#   4. Submit:   sbatch run_snellius.sh
#      Monitor:  squeue -u $USER       Log: slurm-<jobid>.out
#
# ---- COST / RUNTIME (estimate) ----------------------------------------------
#   1 SBU = 1 core-hour; a full Genoa node bills all 192 cores x ACTUAL runtime
#   (not the --time cap). The default Sobol sweep is ~430,080 sims (8.6x the old 50k
#   LHS run), but write_full=false skips the heavy per-node/edge/cascade Arrow files
#   (no ~9 GB write) and emits only a small simulations.{arrow,csv}. Runtime is not
#   yet calibrated; the --time cap below is deliberately generous (it is only a cap;
#   SBU bills actual runtime). Check the elapsed time in slurm-<jobid>.out.
# =============================================================================

#SBATCH --job-name=deepfake_influencers
# Genoa thin node = 192 cores / 384 GiB; take the whole node (exclusive) for max RAM.
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# ---- Julia environment: -----
module load EESSI/2025.06 && module load Julia/1.12.2        # EESSI stack (SURF-documented on Snellius); run 'module avail Julia' to confirm the version
julia --version

PROJ="$HOME/fake-news-percolation"
cd "$PROJ"                           # run + write here; output -> $PROJ/data/sweep_<timestamp>

# one Julia thread per allocated core (reads JULIA_NUM_THREADS automatically)
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "host=$(hostname) nproc=$(nproc) JULIA_NUM_THREADS=$JULIA_NUM_THREADS start=$(date)"

# fail fast if the Saltelli design is missing (see ONE-TIME SETUP step 1b)
if [[ ! -f "$PROJ/sobol/design.csv" ]]; then
    echo "ERROR: $PROJ/sobol/design.csv not found. Generate it first with:"
    echo "       python sobol/make_design.py --N 1024   (needs Python + SALib)"
    exit 1
fi

julia --project="$PROJ" "$PROJ/run_experiment_influencers.jl"

echo "done=$(date)  output in $PROJ/data/ (simulations.arrow + simulations.csv)"
echo "Next (off-node, needs Python + SALib): python sobol/analyze.py"
