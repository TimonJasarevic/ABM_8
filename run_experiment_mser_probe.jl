# run_experiment_mser_probe.jl -- MSER-5 warm-up probe.
#
# The production sweeps never log the first 1000 (burn-in) cascades, so no data exists to
# check the burn-in length. This probe reuses the production model unchanged (module include
# of run_experiment_baseline.jl) and logs every cascade, 1..2000, for a small stratified set
# of design points x 30 reps. Dynamics and RNG stream are identical to the production sweep:
# after_burn_in gates only payoff bookkeeping and the structural-virality computation (no
# random draws, no belief updates), and it is passed exactly as production does (false for
# cascades 1..1000, true after). The replicate-averaged per-cascade-index series feed
# analysis/replication_justification.py -> utils.mser5.
#
# Run:  julia --project=. -t auto run_experiment_mser_probe.jl
# Out:  analysis/runs/mser_probe/{mser_probe_cascades.csv, probe_config.json}

_INCLUDED_AS_MODULE = true
include(joinpath(@__DIR__, "run_experiment_baseline.jl"))

using DelimitedFiles: readdlm
using Printf

const PROBE_OUT = joinpath(@__DIR__, "analysis", "runs", "mser_probe")
const PROBE_REPS = 30
const PROBE_CASCADES = 2000
const PROBE_BURN_IN = 1000          # mirrors the production loop split; logging ignores it
const PROBE_RNG_NUMBER = 42         # production seed constant (thread_seed derivation)

# 3x3 grid over the two dominant drivers: p_fake (col 5) x log10(lambda) (col 6).
const GRID = [(p, l) for p in (0.10, 0.50, 0.90) for l in (-1.5, 0.0, 1.5)]

function pick_design_points(design)
    ids = Int[]
    for (pt, lt) in GRID
        best, bestd = 0, Inf
        for i in 1:size(design, 1)
            d = ((design[i, 5] - pt) / 0.9)^2 + ((design[i, 6] - lt) / 4.0)^2
            if d < bestd && !(i in ids)
                best, bestd = i, d
            end
        end
        push!(ids, best)
    end
    return ids
end

function run_probe()
    mkpath(PROBE_OUT)
    design = readdlm(joinpath(@__DIR__, "sobol", "design.csv"), ',', Float64; skipstart=1)
    design_ids = pick_design_points(design)
    println("Probing design points: ", design_ids)

    jobs = [(did, rep) for did in design_ids for rep in 1:PROBE_REPS]
    rows = Vector{Vector{NTuple{7,Int}}}(undef, length(jobs))

    Threads.@threads for j in 1:length(jobs)
        design_id, rep_id = jobs[j]
        cost, loss = design[design_id, 1], design[design_id, 2]
        tpr, fpr   = design[design_id, 3], design[design_id, 4]
        prev       = design[design_id, 5]
        rat        = 10.0 ^ design[design_id, 6]
        loss_av    = design[design_id, 7]
        thread_seed = Int(mod(hash((design_id, rep_id, PROBE_RNG_NUMBER)), typemax(Int)))

        cfg = SimConfig(
            num_agents=300, num_connections=3, p_triadic_closure=0.5,
            global_fake_prob=prev, verification_cost=cost, rep_gain=1.0, rep_loss=loss,
            verification_tpr=tpr, verification_fpr=fpr, rationality=rat,
            loss_aversion=loss_av, random_seed=thread_seed)
        sim = Simulation(cfg)

        recs = Vector{NTuple{7,Int}}(undef, PROBE_CASCADES)
        for c_id in 1:PROBE_CASCADES
            stats = run_single_cascade!(sim, c_id > PROBE_BURN_IN)
            recs[c_id] = (design_id, rep_id, c_id, Int(stats.is_fake),
                          stats.size, stats.verifications, stats.shares)
        end
        rows[j] = recs
    end

    csv_path = joinpath(PROBE_OUT, "mser_probe_cascades.csv")
    open(csv_path, "w") do io
        println(io, "design_id,rep_id,cascade_id,is_fake,cascade_size,verifications,shares")
        for recs in rows, r in recs
            @printf(io, "%d,%d,%d,%d,%d,%d,%d\n", r...)
        end
    end

    open(joinpath(PROBE_OUT, "probe_config.json"), "w") do io
        pts = join(("{\"design_id\": $(did), \"p_fake\": $(design[did, 5]), " *
                    "\"log10_lambda\": $(design[did, 6])}" for did in design_ids), ", ")
        println(io, "{\"design_ids\": [", join(design_ids, ", "), "], ",
                "\"grid_targets_pfake_log10lambda\": \"3x3: p_fake {0.10,0.50,0.90} x log10lambda {-1.5,0,+1.5}\", ",
                "\"points\": [", pts, "], ",
                "\"reps_per_point\": ", PROBE_REPS, ", ",
                "\"cascades_per_sim\": ", PROBE_CASCADES, ", ",
                "\"burn_in_production\": ", PROBE_BURN_IN, ", ",
                "\"seed_rule\": \"hash((design_id, rep_id, 42)) as in run_experiment_baseline.jl\", ",
                "\"julia_threads\": ", Threads.nthreads(), ", ",
                "\"note\": \"all 2000 cascades logged; after_burn_in passed exactly as production (gates only payoff bookkeeping + SV, no RNG draws), so dynamics match the sweep\"}")
    end
    println("WROTE ", csv_path, "  (", length(jobs) * PROBE_CASCADES, " rows)")
end

run_probe()
