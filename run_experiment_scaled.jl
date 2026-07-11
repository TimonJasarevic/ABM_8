# run_experiment_scaled.jl -- scaled reviewer sweep: the production model at reduced
# sweep breadth (fewer Saltelli base samples and replicates), fully contained in one
# run directory.
#
# The model itself is untouched: this driver includes the production runner as a module
# (the run_experiment_mser_probe.jl pattern) and only passes a smaller design and
# replicate count through run_sobol_sweep's existing keyword arguments. The scaled
# design is the FIRST 16*N rows of the committed sobol/design.csv, copied line-for-line;
# a Sobol-sequence prefix at a power-of-two base sample is itself the Saltelli sample
# that SALib would generate for that N with the same seed (verified byte-identical for
# N = 8, 32, 128), so the scaled run visits a verbatim subset of the published design.
# Replicate subsets keep the common-random-number pairing: thread_seed depends only on
# (design_id, rep_id, 42).
#
# Usage (once per seeding configuration, each in a fresh Julia process):
#   FNP_RUN_ROOT=<run-dir> FNP_N=<base-sample> FNP_REPS=<reps> \
#       julia --project=. -t auto run_experiment_scaled.jl <baseline|influencer>
# Writes into <run-dir> only: design.csv, problem.json, run_meta.json, and
# data/<configuration>/ (simulations + WRITE_FULL cascade/node/edge tables; small at
# scaled sizes). reproduce.sh drives the full tier-3 chain around this.

const ARM = length(ARGS) == 1 ? ARGS[1] : error("usage: run_experiment_scaled.jl <baseline|influencer>")
ARM in ("baseline", "influencer") || error("configuration must be baseline or influencer, got $ARM")

const RUN_ROOT = abspath(get(ENV, "FNP_RUN_ROOT") do
    error("FNP_RUN_ROOT must point at the scaled-run directory")
end)
const N_BASE = parse(Int, get(ENV, "FNP_N") do
    error("FNP_N (Saltelli base sample, power of two <= 1024) is required")
end)
const N_REPS = parse(Int, get(ENV, "FNP_REPS", "2"))
const BLOCK = 16                       # 2k+2 for the fixed k = 7 factors
N_REPS >= 2 || error("FNP_REPS must be >= 2 (the stochastic decomposition needs replicates)")
(N_BASE >= 1 && N_BASE <= 1024) || error("FNP_N must be in 1..1024")
ispow2(N_BASE) || @warn "FNP_N=$N_BASE is not a power of two; Sobol-sequence balance is best at powers of two"

const REPO = @__DIR__
const DESIGN_SRC = joinpath(REPO, "sobol", "design.csv")
const PROBLEM_SRC = joinpath(REPO, "sobol", "problem.json")

# ---------------------------------------------------------------- run-root inputs (idempotent)
mkpath(joinpath(RUN_ROOT, "data"))
let rows = BLOCK * N_BASE
    lines = readlines(DESIGN_SRC)
    length(lines) >= rows + 1 || error("design.csv holds $(length(lines)-1) rows < $rows")
    open(joinpath(RUN_ROOT, "design.csv"), "w") do io
        for ln in @view lines[1:rows+1]   # header + the verbatim design prefix
            println(io, ln)
        end
    end
    prob = read(PROBLEM_SRC, String)
    p1 = replace(prob, "\"N\": 1024" => "\"N\": $N_BASE")
    p2 = replace(p1, "\"n_rows\": 16384" => "\"n_rows\": $rows")
    (p1 != prob && p2 != p1) || error("problem.json rewrite failed; unexpected formatting")
    write(joinpath(RUN_ROOT, "problem.json"), p2)
    write(joinpath(RUN_ROOT, "run_meta.json"),
          "{\"num_reps\": $N_REPS, \"n_base\": $N_BASE, \"design\": \"first $rows rows of sobol/design.csv\"}\n")
end

# ---------------------------------------------------------------- production model, unchanged
_INCLUDED_AS_MODULE = true
include(joinpath(REPO, "run_experiment_" * (ARM == "baseline" ? "baseline" : "influencers") * ".jl"))

dest = joinpath(RUN_ROOT, "data", ARM)
isdir(dest) && error("$dest already exists; use a fresh run directory per configuration")
cd(RUN_ROOT) do
    run_sobol_sweep(design_path=joinpath(RUN_ROOT, "design.csv"),
                    num_reps=N_REPS, write_full=true, out_tag=ARM)
end
sweeps = filter(d -> startswith(d, "sweep_") && endswith(d, "_" * ARM),
                readdir(joinpath(RUN_ROOT, "data")))
length(sweeps) == 1 || error("expected exactly one sweep_*_$ARM under $RUN_ROOT/data, found $(length(sweeps))")
mv(joinpath(RUN_ROOT, "data", sweeps[1]), dest)
println("scaled $ARM sweep complete -> $dest")
