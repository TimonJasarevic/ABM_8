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
    rationality::Float64 = 1.0   # softmax (logit) precision λ; λ→∞ ⇒ argmax, λ→0 ⇒ random
    loss_aversion::Float64 = 1.0  
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
        subjective_loss = cfg.loss_aversion * cfg.rep_loss

        u_share_if_true = cfg.rep_gain * k
        u_share_if_fake = -subjective_loss * k
        u_share = (p_true * u_share_if_true) + ((1.0 - p_true) * u_share_if_fake)

        eu_given_real = (1.0 - est_fpr) * (cfg.rep_gain * k)
        eu_given_fake = (1.0 - est_tpr) * (-subjective_loss * k)
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

# ------------------------------------------------------------
# Network animation
# ------------------------------------------------------------
# Required packages:
#   import Pkg; Pkg.add(["Plots", "GraphRecipes"])
# This is separate from the LHS sweep. It runs one simulation and saves
# a GIF where node labels show payoff, which is the Julia equivalent of
# the Python reputation variable.

using Plots
using GraphRecipes

function payoff_node_labels(payoff::Vector{Float64}; label_every::Int=1)
    labels = Vector{String}(undef, length(payoff))

    for i in eachindex(payoff)
        if label_every <= 1 || i % label_every == 0
            labels[i] = string(round(Int, payoff[i]))
        else
            labels[i] = ""
        end
    end

    return labels
end

function payoff_node_sizes(sim::Simulation;
                           base_size::Float64=0.10,
                           influencer_size::Float64=0.18)
    influencer_set = Set(sim.influencers)

    return [
        node in influencer_set ? influencer_size : base_size
        for node in 1:nv(sim.net.G)
    ]
end

function payoff_node_colors(sim::Simulation)
    influencer_set = Set(sim.influencers)

    return [
        node in influencer_set ? :orange : :lightblue
        for node in 1:nv(sim.net.G)
    ]
end


function fixed_spring_layout(G::SimpleGraph{Int};
                             seed::Int=42,
                             iterations::Int=300,
                             scale::Float64=1.0)
    rng = Xoshiro(seed)
    n = nv(G)

    xs = rand(rng, n) .- 0.5
    ys = rand(rng, n) .- 0.5

    k = sqrt(4.0 / n)
    temperature = 0.35
    cooling = temperature / iterations

    for _ in 1:iterations
        dx = zeros(Float64, n)
        dy = zeros(Float64, n)

        # Repulsive force between all node pairs.
        for v in 1:(n - 1)
            for u in (v + 1):n
                delta_x = xs[v] - xs[u]
                delta_y = ys[v] - ys[u]
                dist = sqrt(delta_x^2 + delta_y^2) + 1e-9

                force = k^2 / dist
                fx = (delta_x / dist) * force
                fy = (delta_y / dist) * force

                dx[v] += fx
                dy[v] += fy
                dx[u] -= fx
                dy[u] -= fy
            end
        end

        # Attractive force along graph edges.
        for e in edges(G)
            v = src(e)
            u = dst(e)

            delta_x = xs[v] - xs[u]
            delta_y = ys[v] - ys[u]
            dist = sqrt(delta_x^2 + delta_y^2) + 1e-9

            force = dist^2 / k
            fx = (delta_x / dist) * force
            fy = (delta_y / dist) * force

            dx[v] -= fx
            dy[v] -= fy
            dx[u] += fx
            dy[u] += fy
        end

        # Move nodes with temperature cap.
        for v in 1:n
            disp = sqrt(dx[v]^2 + dy[v]^2) + 1e-9
            step = min(disp, temperature)
            xs[v] += (dx[v] / disp) * step
            ys[v] += (dy[v] / disp) * step
        end

        temperature = max(0.0, temperature - cooling)
    end

    # Center and rescale to a stable plotting box.
    xs .-= mean(xs)
    ys .-= mean(ys)

    max_abs = max(maximum(abs.(xs)), maximum(abs.(ys)))

    if max_abs > 0
        xs = scale .* xs ./ max_abs
        ys = scale .* ys ./ max_abs
    end

    return xs, ys
end

function draw_network_frame(sim::Simulation, cascade_id::Int, stats, xs, ys;
                            label_every::Int=1)
    A = adjacency_matrix(sim.net.G)

    labels = payoff_node_labels(
        sim.net.payoff;
        label_every=label_every
    )

    # GraphRecipes can error when vector-valued markersize is combined
    # with node labels. Keep the influencer distinction in node color
    # and use one scalar marker size for all nodes.
    node_size = 0.30
    node_colors = payoff_node_colors(sim)

    message_type = stats.is_fake ? "fake" : "true"
    mean_payoff = round(Int, mean(sim.net.payoff))

    graphplot(
        A;
        names=labels,
        x=xs,
        y=ys,
        xlims=(-1.2, 1.2),
        ylims=(-1.2, 1.2),
        markercolor=node_colors,
        markersize=node_size,
        linecolor=:gray,
        linewidth=0.25,
        curves=false,
        nodeshape=:circle,
        fontsize=5,
        title=(
            "Cascade $(cascade_id), $(message_type) message | " *
            "shares=$(stats.shares), verifications=$(stats.verifications), " *
            "mean payoff=$(mean_payoff)"
        ),
        legend=false,
        axis_buffer=0.08,
        framestyle=:none,
        ticks=false,
        size=(900, 700)
    )
end

function run_animation_demo(; 
                            num_agents::Int=120,
                            cascades::Int=250,
                            burn_in::Int=100,
                            draw_every::Int=5,
                            output_file::String="deepfake_network_animation.gif",
                            label_every::Int=8,
                            seed::Int=42)
    cfg = SimConfig(
        random_seed=seed,
        num_agents=num_agents,
        num_connections=3,
        p_triadic_closure=0.5,
        global_fake_prob=0.5,
        verification_cost=1.0,
        rep_gain=1.0,
        rep_loss=2.0,
        verification_tpr=0.90,
        verification_fpr=0.10,
        rationality=1.0,
        loss_aversion=1.0
    )

    sim = Simulation(cfg)

    # Fixed node coordinates. These are computed once and reused in every frame,
    # so nodes do not move around during the animation. This uses a spring layout
    # instead of a circle, so hubs and clusters are easier to see.
    xs, ys = fixed_spring_layout(
        sim.net.G;
        seed=seed,
        iterations=300,
        scale=1.0
    )

    # Burn-in updates beliefs, but does not update payoff because the original
    # model only applies payoff when after_burn_in=true.
    for _ in 1:burn_in
        run_single_cascade!(sim, false)
    end

    frames = 1:draw_every:cascades

    anim = @animate for cascade_id in 1:cascades
        stats = run_single_cascade!(sim, true)

        if cascade_id in frames
            draw_network_frame(
                sim,
                cascade_id,
                stats,
                xs,
                ys;
                label_every=label_every
            )
        end
    end every draw_every

    gif(anim, output_file; fps=6)
    println("Saved animation to: $(output_file)")

    plot_initial_degree_vs_final_payoff(
        sim;
        output_file="initial_degree_vs_final_payoff.png",
        show_plot=true
    )

    return sim
end


function plot_initial_degree_vs_final_payoff(sim::Simulation;
                                             output_file::String="initial_degree_vs_final_payoff.png",
                                             show_plot::Bool=true)
    initial_degrees = [degree(sim.net.G, node) for node in 1:nv(sim.net.G)]
    normalized_final_payoff = sim.net.payoff ./ initial_degrees
    influencer_set = Set(sim.influencers)

    node_colors = [node in influencer_set ? :orange : :lightblue for node in 1:nv(sim.net.G)]

    corr_value = (
        std(Float64.(initial_degrees)) > 0 && std(normalized_final_payoff) > 0
    ) ? cor(Float64.(initial_degrees), normalized_final_payoff) : NaN

    p = scatter(
        initial_degrees,
        normalized_final_payoff;
        markercolor=node_colors,
        markerstrokewidth=0,
        markersize=5,
        alpha=0.75,
        xlabel="Initial node degree",
        ylabel="Final payoff",
        title="Initial node degree vs final payoff, corr=$(round(corr_value; digits=3))",
        legend=false,
        size=(800, 550)
    )

    savefig(p, output_file)

    if show_plot
        display(p)
    end

    println("Saved initial-degree vs final-payoff plot to: $(output_file)")

    return DataFrame(
        node_id = 1:nv(sim.net.G),
        initial_degree = initial_degrees,
        normalized_final_payoff = normalized_final_payoff,
        is_influencer = [node in influencer_set for node in 1:nv(sim.net.G)]
    )
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
        (0.10, 1.90), # 6: rationality (λ; softmax precision, centred on 1.0)
        (1.00, 3.00), # 7: loss_aversion
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
        rat        = lhs_points[lhs_id, 6]
        loss_av    = lhs_points[lhs_id, 7]

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
            rationality = rat,
            loss_aversion = loss_av,
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


if abspath(PROGRAM_FILE) == @__FILE__
    # Use this for the visual demo.
    # The LHS sweep is still available by calling run_lhs_sweep() manually.
    run_animation_demo(
        num_agents=120,
        cascades=500,
        burn_in=0,
        draw_every=20,
        output_file="deepfake_network_animation.gif",
        label_every=1,
        seed=30
    )
end