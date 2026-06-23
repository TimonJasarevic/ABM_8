#!/bin/bash
# =============================================================================
# Snellius (SURF) SLURM job: deepfake-percolation influencer sweep
# Runs run_experiment_influencers.jl (the full 50,000-simulation LHS sweep)
# on ONE full Genoa node (192 cores / 384 GiB), shared-memory multithreaded.
#
# ---- ONE-TIME SETUP (run on a LOGIN node, once) -----------------------------
#   1. Copy the model + project to your home, e.g.:
#        ~/fake-news-percolation/run_experiment_influencers.jl
#        ~/fake-news-percolation/Project.toml      (optional; created by step 3)
#        ~/fake-news-percolation/run_snellius.sh   (this file)
#   2. Make Julia available. CONFIRM what your Snellius offers first:
#        module avail Julia
#      then pick ONE of the options in the "Julia environment" block below.
#   3. Install the dependencies into the project ONCE (login node has internet):
#        cd ~/fake-news-percolation
#        julia --project=. -e 'using Pkg; Pkg.add(["Graphs","DataStructures",\
#          "StatsBase","ProgressMeter","DataFrames","Arrow"]); Pkg.precompile()'
#      (Random, Statistics, Dates are stdlib — no install needed.)
#   4. Submit:   sbatch run_snellius.sh
#      Monitor:  squeue -u $USER       Log: slurm-<jobid>.out
#
# ---- COST / RUNTIME (estimate) ----------------------------------------------
#   1 SBU = 1 core-hour; a full Genoa node bills all 192 cores x ACTUAL runtime
#   (not the --time cap). Estimated runtime ~10-20 min (dominated by the serial
#   ~9 GB Arrow write at the end), so ~32-64 SBU. The first run is the real
#   calibration — check the elapsed time in slurm-<jobid>.out.
# =============================================================================

#SBATCH --job-name=deepfake_influencers
# Genoa thin node = 192 cores / 384 GiB; take the whole node (exclusive) for max RAM.
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
# wall-time CAP only; SBU is billed on ACTUAL runtime (~10-20 min), not this limit.
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
# Uncomment + fill in if your account requires an explicit budget:
# #SBATCH -A <YOUR_PROJECT>

set -euo pipefail

# ---- Julia environment: confirm with `module avail Julia`, then pick ONE -----
# module load EESSI/2025.06 && module load Julia/1.12.2        # RECOMMENDED: EESSI stack (SURF-documented on Snellius); run 'module avail Julia' to confirm the version
# module load 2025 && module load Julia/<version>             # native stack: NO Julia module documented by SURF; check 'module load 2025; module avail Julia'
# export PATH="$HOME/.juliaup/bin:$PATH"                       # fallback: self-installed via juliaup
julia --version

PROJ="$HOME/fake-news-percolation"
cd "$PROJ"                           # run + write here; output -> $PROJ/data/sweep_<timestamp>

# one Julia thread per allocated core (reads JULIA_NUM_THREADS automatically)
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "host=$(hostname) nproc=$(nproc) JULIA_NUM_THREADS=$JULIA_NUM_THREADS start=$(date)"

julia --project="$PROJ" "$PROJ/run_experiment_influencers.jl"

echo "done=$(date)  output in $PROJ/data/"
