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
using DelimitedFiles

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
    rationality::Float64 = 1.0   # softmax (logit) precision λ; λ→∞ ⇒ argmax, λ→0 ⇒ random
    loss_aversion::Float64 = 1.0  # prospect-theory loss-aversion coeff λ_LA (linear, α=β=1); swept via
                                  # the Saltelli design (col 7). Scales the reputational LOSS only in
                                  # decide(); 1.0 ⇒ loss-neutral (recovers prior model). Reference = DISCARD = 0
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
    function decide(p_true::Float64, reach::Int, est_tpr::Float64, est_fpr::Float64, cfg::SimConfig, rng)
        k = Float64(reach)
        # prospect-theory loss aversion: the reputational LOSS from sharing fake news is
        # weighted ×ℓ (linear value function, α=β=1; reference = DISCARD = 0). The
        # verification cost is an objective effort cost and is NOT scaled. ℓ=1 ⇒ loss-neutral.
        # This biases the PERCEIVED decision utility only — realized payoffs/welfare
        # elsewhere still use the true rep_gain/rep_loss.
        ℓ = cfg.loss_aversion
        u_share_if_true = cfg.rep_gain * k
        u_share_if_fake = -ℓ * cfg.rep_loss * k
        u_share = (p_true * u_share_if_true) + ((1.0 - p_true) * u_share_if_fake)
        eu_given_real = (1.0 - est_fpr) * (cfg.rep_gain * k)
        eu_given_fake = (1.0 - est_tpr) * (-ℓ * cfg.rep_loss * k)
        u_verify = -cfg.verification_cost + (p_true * eu_given_real) + ((1.0 - p_true) * eu_given_fake)
        # bounded rationality: softmax (logit quantal response) over the three action utilities
        # (DISCARD utility = 0). λ→∞ recovers the old argmax; λ→0 → uniform random choice.
        λ = cfg.rationality
        u_max = max(u_share, u_verify, 0.0)
        w_share   = exp(λ * (u_share  - u_max))
        w_verify  = exp(λ * (u_verify - u_max))
        w_discard = exp(λ * (0.0      - u_max))
        t = rand(rng) * (w_share + w_verify + w_discard)
        return t < w_share ? :SHARE : (t < w_share + w_verify ? :VERIFY : :DISCARD)
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
        action = AgentLogic.decide(p_true, reach, est_tpr, est_fpr, sim.cfg, sim.rng)
        
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
            seed_degree = seed_degree, max_depth = max_depth,
            structural_virality = after_burn_in ? _structural_virality(visited, parent_map, seed_node) : 0.0)
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
# The design now comes from a SALib Saltelli sample (sobol/make_design.py); the old
# internal latin_hypercube_sample was removed since Sobol indices require a Saltelli
# design, not LHS.

# sen_welfare per simulation, computed in Julia from in-memory payoffs so the SA
# needs no node-level export (matches Python utils.py: agent_veracity + Sen welfare).
function _gini(x::AbstractVector{Float64})
    n = length(x)
    n == 0 && return 0.0
    xs = sort(x)
    total = sum(xs)
    total <= 0.0 && return 0.0
    return (2.0 * sum((1:n) .* xs)) / (n * total) - (n + 1.0) / n
end

function _agent_veracity(payoff::Float64, best::Float64, worst::Float64)
    if payoff < 0.0 && worst != 0.0
        return -(payoff / worst)
    elseif payoff > 0.0 && best != 0.0
        return payoff / best
    else
        return 0.0
    end
end

_sen_welfare(x::AbstractVector{Float64}) = isempty(x) ? 0.0 : mean(x) * (1.0 - _gini(x .- minimum(x)))

# Structural virality (Goel et al. 2016): mean pairwise shortest-path distance in the diffusion
# tree, g(T) = (1/(n(n-1))) * sum_{i,j} d(i,j). Computed in O(n) on the rooted tree via the edge-
# contribution identity sum_{i<j} d(i,j) = sum_edges s*(n-s), s = subtree size below the edge.
# Pure broadcast (star) -> g->2; deep multi-step chains -> larger g. Undefined for n<2 -> 0.
function _structural_virality(visited, parent_map::Dict{Int,Int}, seed::Int)
    n = length(visited)
    n < 2 && return 0.0
    children = Dict{Int, Vector{Int}}()
    for (child, parent) in parent_map
        push!(get!(children, parent, Int[]), child)
    end
    subsize = Dict{Int, Int}()
    order = Int[]
    stack = Int[seed]
    while !isempty(stack)
        v = pop!(stack)
        push!(order, v)
        for c in get(children, v, Int[])
            push!(stack, c)
        end
    end
    for v in Iterators.reverse(order)
        s = 1
        for c in get(children, v, Int[])
            s += subsize[c]
        end
        subsize[v] = s
    end
    W = 0
    for v in keys(parent_map)
        s = subsize[v]
        W += s * (n - s)
    end
    return 2.0 * W / (n * (n - 1))
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
    utility::Float64,
    structural_virality::Float64
}

function run_sobol_sweep(; design_path::String="sobol/design.csv", num_reps::Int=30, write_full::Bool=false, out_tag::String="")
    timestamp = Dates.format(now(), "yyyy_mm_dd_HHMM")
    out_dir = joinpath("data", isempty(out_tag) ? "sweep_$timestamp" : "sweep_$(timestamp)_$out_tag")
    mkpath(out_dir)

    # Saltelli design generated by SALib (sobol/make_design.py). Columns, in order:
    # 1 v_cost  2 rep_loss  3 tpr  4 fpr  5 p_fake  6 log10(λ rationality)  7 loss_aversion (λ_LA)
    design = readdlm(design_path, ',', Float64; skipstart=1)
    NUM_POINTS = size(design, 1)
    NUM_REPS = num_reps
    CASCADES_PER_SIM = 2000
    BURN_IN = 1000
    N_AGENTS = 300
    PAYOFF_GAIN = 1.0
    NUM_CONNECTIONS = 3
    P_TRIADIC_CLOSURE = 0.5
    RNG_NUMBER = 42

    TOTAL_SIMS = NUM_POINTS * NUM_REPS

    # Save Experiment Metadata
    open(joinpath(out_dir, "experiment_info.txt"), "w") do io
        println(io, "Deepfake Percolation ABM — Saltelli/Sobol sweep")
        println(io, "================================================")
        println(io, "Date: ", now())
        println(io, "Julia Threads: ", Threads.nthreads())
        println(io, "Design source: ", design_path, " (SALib Saltelli)")
        println(io, "Design points N(2k+2): ", NUM_POINTS)
        println(io, "Reps per point: ", NUM_REPS)
        println(io, "Agents (N): ", N_AGENTS)
        println(io, "Cascades per Sim: ", CASCADES_PER_SIM)
        println(io, "Factors: v_cost, rep_loss, tpr, fpr, p_fake, log10(rationality), loss_aversion")
        println(io, "Loss aversion λ_LA: swept factor (col 7), linear range [1.0, 3.0]; scales rep_loss only (verification cost not scaled); reference = DISCARD = 0; decision-bias only")
        println(io, "Seeding: top-5% highest-degree hubs (influencer)")
        println(io, "Structural virality: per-cascade Wiener index recorded (Goel et al. 2016)")
    end

    # arrays to hold thread outputs. cascades/nodes/edges are only materialised when
    # write_full=true (the SA reads only sim-level data), keeping the sweep lean.
    sim_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)
    cascades_dfs = write_full ? Vector{DataFrame}(undef, TOTAL_SIMS) : DataFrame[]
    node_dfs     = write_full ? Vector{DataFrame}(undef, TOTAL_SIMS) : DataFrame[]
    edge_dfs     = write_full ? Vector{DataFrame}(undef, TOTAL_SIMS) : DataFrame[]

    println("Saving experiment to: $out_dir/")
    println("Loaded $NUM_POINTS Saltelli design points from $design_path.")
    println("Running $NUM_REPS stochastic reps per point -> $TOTAL_SIMS total simulations.")
    println("Using $(Threads.nthreads()) threads...")

    prog = Progress(TOTAL_SIMS, 1, "Sweeping... ")
    lk = ReentrantLock()

    Threads.@threads for idx in 1:TOTAL_SIMS
        design_id = div(idx - 1, NUM_REPS) + 1
        rep_id = mod(idx - 1, NUM_REPS) + 1

        cost, loss = design[design_id, 1], design[design_id, 2]
        tpr, fpr   = design[design_id, 3], design[design_id, 4]
        prev       = design[design_id, 5]
        rat        = 10.0 ^ design[design_id, 6]   # back-transform log10(λ) → λ
        loss_av    = design[design_id, 7]          # loss-aversion λ_LA (linear; 1.0 = loss-neutral)

        thread_seed = Int(mod(hash((design_id, rep_id, RNG_NUMBER)), typemax(Int)))     

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
            rationality=rat,
            loss_aversion=loss_av,
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
            
            # store cascade metrics after burn-in period (only when write_full)
            write_full && push!(cascade_records, (
                global_sim_id = idx,
                cascade_id = c_id,
                is_fake = stats.is_fake,
                seed_node = stats.seed_node,
                seed_degree = stats.seed_degree,
                max_depth = stats.max_depth,
                cascade_size = stats.size,
                shares = stats.shares,
                verifications = stats.verifications,
                utility = stats.utility,
                structural_virality = stats.structural_virality
            ))

            push!(point_utilities, stats.utility)
            vr = stats.size > 0 ? (stats.verifications / stats.size) : 0.0
            push!(point_verify_rates, vr)
            stats.is_fake ? push!(point_fake_sizes, stats.size) : push!(point_true_sizes, stats.size)
        end

        avg_fake_cascade = isempty(point_fake_sizes) ? 0.0 : mean(point_fake_sizes)/N_AGENTS
        avg_true_cascade = isempty(point_true_sizes) ? 0.0 : mean(point_true_sizes)/N_AGENTS

        # sen_welfare per sim from agents' normalized payoffs (matches Python utils.sen_welfare)
        agent_veracity = [_agent_veracity(sim.net.payoff[i], sim.net.max_possible_payoff[i],
                                          sim.net.min_possible_payoff[i]) for i in 1:N_AGENTS]
        sw = _sen_welfare(agent_veracity)

        sim_dfs[idx] = DataFrame(
            global_sim_id = idx,
            design_id = design_id,
            rep_id = rep_id,
            v_cost = cost,
            loss = loss,
            tpr = tpr,
            fpr = fpr,
            p_fake = prev,
            rationality = rat,
            loss_aversion = loss_av,
            avg_payoff = mean(point_utilities),
            avg_fake_cascade = avg_fake_cascade,
            avg_true_cascade = avg_true_cascade,
            veracity_differential = avg_true_cascade - avg_fake_cascade,
            avg_verify_rate = mean(point_verify_rates),
            sen_welfare = sw
        )

        # full per-node / per-edge / per-cascade tables only on request (the SA uses
        # only the sim-level table above); skipping them keeps the large sweep lean.
        if write_full
            cascades_dfs[idx] = DataFrame(cascade_records)
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
        end
        
        lock(lk) do
            next!(prog)
        end
    end

    println("\nConcatenating + saving...")
    all_sims = vcat(sim_dfs...)
    Arrow.write(joinpath(out_dir, "simulations.arrow"), all_sims)

    # compact CSV of the sim-level table so the Sobol analyzer reads it with plain
    # pandas (no pyarrow); see sobol/analyze.py
    open(joinpath(out_dir, "simulations.csv"), "w") do io
        println(io, join(names(all_sims), ","))
        for r in eachrow(all_sims)
            println(io, join((string(x) for x in r), ","))
        end
    end

    if write_full
        Arrow.write(joinpath(out_dir, "cascades.arrow"), vcat(cascades_dfs...))
        Arrow.write(joinpath(out_dir, "nodes.arrow"), vcat(node_dfs...))
        Arrow.write(joinpath(out_dir, "edges.arrow"), vcat(edge_dfs...))
    end

    println("\nSuccessfully saved into $(out_dir)/")
end


if !@isdefined(_INCLUDED_AS_MODULE)
    write_full = get(ENV, "WRITE_FULL", "false") == "true"
    out_tag = get(ENV, "OUT_TAG", "")
    # stderr is unbuffered, so this is visible in the slurm log even if the job is later killed
    println(stderr, "WRITE_FULL env=", repr(get(ENV, "WRITE_FULL", "<unset>")), " -> write_full=", write_full, " OUT_TAG=", repr(out_tag)); flush(stderr)
    run_sobol_sweep(; write_full=write_full, out_tag=out_tag)
end