# deepfake_percolation.jl
using Graphs
using DataStructures
using StatsBase
using Random
using ProgressMeter
using DataFrames
using Arrow
using Statistics
using Dates

# configuration
Base.@kwdef struct SimConfig
    random_seed::Int = 42
    num_agents::Int = 300
    num_connections::Int = 3
    p_triadic_closure::Float64 = 0.25
    global_fake_prob::Float64 = 0.5
    verification_cost::Float64 = 1.0 
    rep_gain::Float64 = 1.0
    rep_loss::Float64 = 2.0
    verification_tpr::Float64 = 0.90
    verification_fpr::Float64 = 0.10
    prior_min::Float64 = 0.3
    prior_max::Float64 = 0.7
    prior_strength::Float64 = 2.0
end

mutable struct BetaDist
    alpha::Float64
    beta::Float64
end

# agent logic
module AgentLogic
    using ..Main: SimConfig, BetaDist
    compute_beta_mean(b::BetaDist) = b.alpha / (b.alpha + b.beta)
    compute_beta_mean(alpha::Float64, beta::Float64) = alpha / (alpha + beta)
    function calculate_posterior_truth_prob(trust::Float64, global_true_belief::Float64)
        p_false_global = 1.0 - global_true_belief
        numerator = trust * global_true_belief
        denominator = (trust * global_true_belief) + ((1.0 - trust) * p_false_global)
        return denominator == 0.0 ? 0.0 : numerator / denominator
    end
    function decide(p_true::Float64, reach::Int, est_tpr::Float64, est_fpr::Float64, cfg::SimConfig)
        k = Float64(reach)
        u_share_if_true = cfg.rep_gain * k
        u_share_if_fake = -cfg.rep_loss * k
        u_share = (p_true * u_share_if_true) + ((1.0 - p_true) * u_share_if_fake)
        eu_given_real = (1.0 - est_fpr) * (cfg.rep_gain * k)
        eu_given_fake = (1.0 - est_tpr) * (-cfg.rep_loss * k)
        u_verify = -cfg.verification_cost + (p_true * eu_given_real) + ((1.0 - p_true) * eu_given_fake)
        if u_share > 0.0
            return u_share > u_verify ? :SHARE : :VERIFY
        else
            return 0.0 >= u_verify ? :DISCARD : :VERIFY
        end
    end
end

# environment
mutable struct SocialNetwork
    cfg::SimConfig
    rng::AbstractRNG
    G::SimpleGraph{Int}
    payoff::Vector{Float64}
    true_belief::Vector{BetaDist}
    tech_tpr_belief::Vector{BetaDist}
    tech_fpr_belief::Vector{BetaDist}
    neighbor_trust::Vector{Dict{Int, BetaDist}}
    base_neighbor_alpha::Vector{Float64}
    base_neighbor_beta::Vector{Float64}
    max_possible_payoff::Vector{Float64}
    min_possible_payoff::Vector{Float64}
    shares_true::Vector{Int}
    shares_fake::Vector{Int}
    verify_count::Vector{Int}
    exposure_count::Vector{Int}
end

function powerlaw_cluster_graph(n::Int, m::Int, p::Float64; rng=Random.default_rng())
    g = SimpleGraph(n)

    seed = m + 1

    # initial clique
    for i in 1:seed
        for j in i+1:seed
            add_edge!(g, i, j)
        end
    end

    # preferential attachment list
    pref = Int[]
    for v in 1:seed
        append!(pref, fill(v, degree(g, v)))
    end

    for v in seed+1:n
        connected = Int[]

        # first edge
        u = rand(rng, pref)
        add_edge!(g, v, u)
        push!(connected, u)

        while length(connected) < m
            if rand(rng) < p
                w = rand(rng, neighbors(g, u))
            else
                w = rand(rng, pref)
            end

            if w != v && !(w in connected) && !has_edge(g, v, w)
                add_edge!(g, v, w)
                push!(connected, w)
            end
        end

        # update preferential attachment list
        for u in connected
            push!(pref, v)
            push!(pref, u)
        end
    end

    return g
end

# identify influencer agents: the top-degree hub nodes (mirrors the Mesa model)
function find_influencers(G; top_fraction::Float64=0.05)
    n = nv(G)
    n_influencers = max(1, floor(Int, top_fraction * n))
    sorted_nodes = sort(collect(1:n); by = v -> degree(G, v), rev = true)
    return sorted_nodes[1:n_influencers]
end

function SocialNetwork(cfg::SimConfig, rng::AbstractRNG)
    
    G = powerlaw_cluster_graph(cfg.num_agents, cfg.num_connections, cfg.p_triadic_closure; rng=rng)

    # sample mean uniformly from [prior_min, prior_max], then convert to (alpha, beta)
    function rand_prior_dist()
        m = cfg.prior_min + rand(rng) * (cfg.prior_max - cfg.prior_min)
        return BetaDist(m * cfg.prior_strength, (1.0 - m) * cfg.prior_strength)
    end
    function rand_prior_alpha()
        m = cfg.prior_min + rand(rng) * (cfg.prior_max - cfg.prior_min)
        return m * cfg.prior_strength
    end
    function rand_prior_beta()
        m = cfg.prior_min + rand(rng) * (cfg.prior_max - cfg.prior_min)
        return (1.0 - m) * cfg.prior_strength
    end

    payoff           = zeros(Float64, cfg.num_agents)
    true_belief      = [rand_prior_dist() for _ in 1:cfg.num_agents]
    tech_tpr_belief  = [rand_prior_dist() for _ in 1:cfg.num_agents]
    tech_fpr_belief  = [rand_prior_dist() for _ in 1:cfg.num_agents]
    base_neighbor_alpha = [rand_prior_alpha() for _ in 1:cfg.num_agents]
    base_neighbor_beta  = [rand_prior_beta()  for _ in 1:cfg.num_agents]
    
    neighbor_trust = [Dict{Int, BetaDist}() for _ in 1:cfg.num_agents]
    best_payoff = zeros(Float64, cfg.num_agents)
    worst_payoff = zeros(Float64, cfg.num_agents)
    true_shares = zeros(Int, cfg.num_agents)
    fake_shares = zeros(Int, cfg.num_agents)
    net = SocialNetwork(
        cfg,
        rng,
        G,
        payoff,
        true_belief,
        tech_tpr_belief,
        tech_fpr_belief,
        neighbor_trust,
        base_neighbor_alpha,
        base_neighbor_beta,
        best_payoff,
        worst_payoff,
        true_shares,
        fake_shares,
        zeros(Int, cfg.num_agents),
        zeros(Int, cfg.num_agents)
    )
    for e in edges(G)
        u, v = src(e), dst(e)
        _init_trust_one_way!(net, u, v)
        _init_trust_one_way!(net, v, u)
    end
    return net
end

function _init_trust_one_way!(net::SocialNetwork, observer::Int, target::Int)
    net.neighbor_trust[observer][target] = BetaDist(net.base_neighbor_alpha[observer], net.base_neighbor_beta[observer])
end

function update_trust!(net::SocialNetwork, observer::Int, sender::Int, item_is_fake::Bool)
    if haskey(net.neighbor_trust[observer], sender)
        stats = net.neighbor_trust[observer][sender]
        item_is_fake ? (stats.beta += 1.0) : (stats.alpha += 1.0)
    end
end

function update_global_belief!(net::SocialNetwork, agent_id::Int, item_is_fake::Bool)
    stats = net.true_belief[agent_id]
    item_is_fake ? (stats.beta += 1.0) : (stats.alpha += 1.0)
end

function update_tech_beliefs!(net, agent_id, item_is_fake, tool_says_fake)
    if item_is_fake 
        tpr = net.tech_tpr_belief[agent_id]
        if tool_says_fake
            tpr.alpha += 1.0
        else
            tpr.beta += 1.0
        end
    else
        fpr = net.tech_fpr_belief[agent_id]
        if tool_says_fake
            fpr.alpha += 1.0
        else
            fpr.beta += 1.0
        end
    end
end

# simulation
mutable struct Simulation
    cfg::SimConfig
    rng::AbstractRNG
    net::SocialNetwork
    cascade_count::Int
    influencers::Vector{Int}
end

function Simulation(cfg::SimConfig)
    rng = Xoshiro(cfg.random_seed)
    net = SocialNetwork(cfg, rng)
    return Simulation(cfg, rng, net, 0, find_influencers(net.G))
end

function run_single_cascade!(sim::Simulation, after_burn_in::Bool)
    sim.cascade_count += 1
    item_is_fake = rand(sim.rng) < sim.cfg.global_fake_prob
    seed_node = rand(sim.rng, sim.influencers)
    seed_degree = degree(sim.net.G, seed_node)
    
    # BFS queue: (current_node, sender_node, depth)
    queue = Queue{Tuple{Int, Int, Int}}()
    enqueue!(queue, (seed_node, 0, 0))
    visited = Set{Int}([seed_node])
    parent_map = Dict{Int, Int}()
    
    actions_log = Dict{Int, Tuple{Symbol, Int}}()
    verification_log = Vector{Tuple{Int, Bool}}()
    
    n_shares, n_verifications = 0, 0
    total_payoff = 0.0
    max_depth = 0

    actual_tpr, actual_fpr = sim.cfg.verification_tpr, sim.cfg.verification_fpr

    while !isempty(queue)
        current, sender, depth = dequeue!(queue)
        max_depth = max(max_depth, depth)
        
        # compute trust: if current is the seed node, it uses its prior trust beliefs instead of actual neighbor trust
        if sender == 0
            trust = AgentLogic.compute_beta_mean(sim.net.base_neighbor_alpha[current], sim.net.base_neighbor_beta[current])
        else
            trust = AgentLogic.compute_beta_mean(sim.net.neighbor_trust[current][sender])
        end

        global_true_est = AgentLogic.compute_beta_mean(sim.net.true_belief[current])
        p_true = AgentLogic.calculate_posterior_truth_prob(trust, global_true_est)
        est_tpr = AgentLogic.compute_beta_mean(sim.net.tech_tpr_belief[current])
        est_fpr = AgentLogic.compute_beta_mean(sim.net.tech_fpr_belief[current])
        reach = degree(sim.net.G, current)

        # update potential max/min payoff after burn-in period
        if after_burn_in
            if !item_is_fake
                sim.net.max_possible_payoff[current] += (sim.cfg.rep_gain * reach)
            else
                sim.net.min_possible_payoff[current] -= (sim.cfg.rep_loss * reach)
            end
        end
        
        # choose action
        action = AgentLogic.decide(p_true, reach, est_tpr, est_fpr, sim.cfg)
        
        if action != :DISCARD
            is_verifying = (action == :VERIFY)
            tool_says_fake = false
            
            if is_verifying
                n_verifications += 1
                total_payoff -= sim.cfg.verification_cost
                tool_says_fake = item_is_fake ? (rand(sim.rng) < actual_tpr) : (rand(sim.rng) < actual_fpr)
                push!(verification_log, (current, tool_says_fake))
            end

            if !tool_says_fake
                n_shares += 1
                outcome_type = item_is_fake ? :shared_fake : :shared_true
                actions_log[current] = (outcome_type, reach)
                total_payoff += item_is_fake ? -(sim.cfg.rep_loss * reach) : (sim.cfg.rep_gain * reach)

                for nbr in neighbors(sim.net.G, current)
                    if !(nbr in visited)
                        push!(visited, nbr)
                        enqueue!(queue, (nbr, current, depth + 1))
                        parent_map[nbr] = current
                    end
                end
            end
        end
    end

    _apply_updates!(sim, actions_log, verification_log, visited, parent_map, item_is_fake, after_burn_in)

    return (is_fake = item_is_fake, size = length(visited), shares = n_shares, 
            verifications = n_verifications, utility = total_payoff, seed_node = seed_node, 
            seed_degree = seed_degree, max_depth = max_depth)
end

function _apply_updates!(sim::Simulation, actions_log, verification_log, visited, parent_map, item_is_fake, after_burn_in::Bool)
    # update beliefs about fake prevalence
    for agent in visited
        update_global_belief!(sim.net, agent, item_is_fake)
        sim.net.exposure_count[agent] += 1
    end

    # update beliefs about neighbor trustworthiness
    for (receiver, sender) in parent_map
        update_trust!(sim.net, receiver, sender, item_is_fake)
    end

    # update beliefs about verification technology
    for (agent, tool_says_fake) in verification_log
        update_tech_beliefs!(sim.net, agent, item_is_fake, tool_says_fake)
        sim.net.verify_count[agent] += 1
    end

    # update payoff only after burn-in period
    if after_burn_in
        for (agent, (action, marginal_reach)) in actions_log
            if action == :shared_true
                sim.net.payoff[agent] += (sim.cfg.rep_gain * marginal_reach)
                sim.net.shares_true[agent] += 1
            elseif action == :shared_fake
                sim.net.payoff[agent] -= (sim.cfg.rep_loss * marginal_reach)
                sim.net.shares_fake[agent] += 1
            end
        end
    end
end

# sweep and export logic
function latin_hypercube_sample(n_samples, bounds; rng=Random.default_rng())
    n_vars = length(bounds)
    samples = zeros(Float64, n_samples, n_vars)
    for i in 1:n_vars
        lb, ub = bounds[i]
        pts = (collect(0:(n_samples-1)) .+ rand(rng, n_samples)) ./ n_samples
        pts = lb .+ pts .* (ub - lb)
        shuffle!(rng, pts)
        samples[:, i] = pts
    end
    return samples
end

const CascadeRecord = @NamedTuple{
    global_sim_id::Int,
    cascade_id::Int,
    is_fake::Bool,
    seed_node::Int,
    seed_degree::Int,
    max_depth::Int,
    cascade_size::Int,
    shares::Int,
    verifications::Int,
    utility::Float64
}

function run_lhs_sweep()
    timestamp = Dates.format(now(), "yyyy_mm_dd_HHMM")
    out_dir = joinpath("data", "sweep_$timestamp")
    mkpath(out_dir)

    bounds = [
        (0.01, 1.00), # 1: verification cost
        (1.00, 3.00), # 2: rep_loss
        (0.60, 0.99), # 3: verification_tpr
        (0.01, 0.40), # 4: verification_fpr
        (0.05, 0.95), # 5: global_fake_prob
    ]

    NUM_LHS_POINTS = 1000
    NUM_REPS = 50
    CASCADES_PER_SIM = 2000
    BURN_IN = 1000
    N_AGENTS = 300
    PAYOFF_GAIN = 1.0
    NUM_CONNECTIONS = 3
    P_TRIADIC_CLOSURE = 0.5
    RNG_NUMBER = 42

    TOTAL_SIMS = NUM_LHS_POINTS * NUM_REPS

    # Save Experiment Metadata
    open(joinpath(out_dir, "experiment_info.txt"), "w") do io
        println(io, "Deepfake Percolation ABM Sweep")
        println(io, "==============================")
        println(io, "Date: ", now())
        println(io, "Julia Threads: ", Threads.nthreads())
        println(io, "Total LHS Points: ", NUM_LHS_POINTS)
        println(io, "Reps per point: ", NUM_REPS)
        println(io, "Agents (N): ", N_AGENTS)
        println(io, "Cascades per Sim: ", CASCADES_PER_SIM)
        println(io, "\nParameter Bounds:")
        println(io, bounds)
    end

    master_rng = Xoshiro(RNG_NUMBER)
    lhs_points = latin_hypercube_sample(NUM_LHS_POINTS, bounds; rng=master_rng)

    # arrays to hold thread outputs
    sim_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)
    cascades_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)
    node_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)
    edge_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)

    println("Saving experiment to: $out_dir/")
    println("Generating $NUM_LHS_POINTS LHS combinations...")
    println("Running $NUM_REPS stochastic reps per combination -> $TOTAL_SIMS total simulations.")
    println("Using $(Threads.nthreads()) threads...")

    prog = Progress(TOTAL_SIMS, 1, "Sweeping... ")
    lk = ReentrantLock()

    Threads.@threads for idx in 1:TOTAL_SIMS
        lhs_id = div(idx - 1, NUM_REPS) + 1
        rep_id = mod(idx - 1, NUM_REPS) + 1

        cost, loss = lhs_points[lhs_id, 1], lhs_points[lhs_id, 2]
        tpr, fpr   = lhs_points[lhs_id, 3], lhs_points[lhs_id, 4]
        prev       = lhs_points[lhs_id, 5]
        
        thread_seed = Int(mod(hash((lhs_id, rep_id, RNG_NUMBER)), typemax(Int)))     

        cfg = SimConfig(
            num_agents=N_AGENTS,
            num_connections=NUM_CONNECTIONS,
            p_triadic_closure=P_TRIADIC_CLOSURE,
            global_fake_prob=prev,
            verification_cost=cost,
            rep_gain=PAYOFF_GAIN,
            rep_loss=loss,
            verification_tpr=tpr,
            verification_fpr=fpr,
            random_seed=thread_seed
        )
        
        sim = Simulation(cfg)
        
        cascade_records = CascadeRecord[]
        point_utilities = Float64[]
        point_fake_sizes = Float64[]
        point_true_sizes = Float64[]
        point_verify_rates = Float64[]

        # burn-in period
        for c_id in 1:BURN_IN
            run_single_cascade!(sim, false)
        end

        # post burn-in, collect metrics
        for c_id in (BURN_IN + 1):CASCADES_PER_SIM
            stats = run_single_cascade!(sim, true)
            
            # store cascade metrics after burn-in period
            push!(cascade_records, (
                global_sim_id = idx,
                cascade_id = c_id,
                is_fake = stats.is_fake,
                seed_node = stats.seed_node,
                seed_degree = stats.seed_degree,
                max_depth = stats.max_depth,
                cascade_size = stats.size,
                shares = stats.shares,
                verifications = stats.verifications,
                utility = stats.utility
            ))

            push!(point_utilities, stats.utility)
            vr = stats.size > 0 ? (stats.verifications / stats.size) : 0.0
            push!(point_verify_rates, vr)
            stats.is_fake ? push!(point_fake_sizes, stats.size) : push!(point_true_sizes, stats.size)
        end

        cascades_dfs[idx] = DataFrame(cascade_records)
        avg_fake_cascade = isempty(point_fake_sizes) ? 0.0 : mean(point_fake_sizes)/N_AGENTS
        avg_true_cascade = isempty(point_true_sizes) ? 0.0 : mean(point_true_sizes)/N_AGENTS

        sim_dfs[idx] = DataFrame(
            global_sim_id = idx,
            lhs_id = lhs_id,
            rep_id = rep_id,
            v_cost = cost,
            loss = loss,
            tpr = tpr,
            fpr = fpr, 
            p_fake = prev,
            avg_payoff = mean(point_utilities),
            avg_fake_cascade = avg_fake_cascade,
            avg_true_cascade = avg_true_cascade,
            veracity_differential = avg_true_cascade - avg_fake_cascade,
            avg_verify_rate = mean(point_verify_rates)
        )

        node_dfs[idx] = DataFrame(
            global_sim_id = fill(idx, N_AGENTS), node_id = 1:N_AGENTS,
            payoff = sim.net.payoff,
            best_payoff = sim.net.max_possible_payoff,
            worst_payoff = sim.net.min_possible_payoff,
            base_alpha = sim.net.base_neighbor_alpha,
            base_beta  = sim.net.base_neighbor_beta,
            true_alpha = [b.alpha for b in sim.net.true_belief],
            true_beta  = [b.beta for b in sim.net.true_belief],
            tpr_alpha  = [b.alpha for b in sim.net.tech_tpr_belief],
            tpr_beta   = [b.beta for b in sim.net.tech_tpr_belief],
            fpr_alpha  = [b.alpha for b in sim.net.tech_fpr_belief],
            fpr_beta   = [b.beta for b in sim.net.tech_fpr_belief],
            true_shares = sim.net.shares_true,
            fake_shares = sim.net.shares_fake
        )

        srcs, dsts, alphas, betas = Int[], Int[], Float64[], Float64[]
        for u in 1:N_AGENTS
            for (v, trust) in sim.net.neighbor_trust[u]
                push!(srcs, u)
                push!(dsts, v)
                push!(alphas, trust.alpha)
                push!(betas, trust.beta)
            end
        end
        edge_dfs[idx] = DataFrame(
            global_sim_id = fill(idx, length(srcs)),
            src = srcs,
            dst = dsts,
            trust_alpha = alphas,
            trust_beta = betas
        )
        
        lock(lk) do
            next!(prog)
        end
    end

    println("\nConcatenating DataFrames...")
    all_sims = vcat(sim_dfs...)
    all_cascades = vcat(cascades_dfs...)
    all_nodes = vcat(node_dfs...)
    all_edges = vcat(edge_dfs...)

    println("Saving to disk as Arrow files...")
    Arrow.write(joinpath(out_dir, "simulations.arrow"), all_sims)
    Arrow.write(joinpath(out_dir, "cascades.arrow"), all_cascades)
    Arrow.write(joinpath(out_dir, "nodes.arrow"), all_nodes)
    Arrow.write(joinpath(out_dir, "edges.arrow"), all_edges)

    println("\nSuccessfully saved into $(out_dir)/")
end


if !@isdefined(_INCLUDED_AS_MODULE)
    run_lhs_sweep()
end