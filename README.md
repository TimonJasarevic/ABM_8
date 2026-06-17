# Misinformation ABM starter project

This is a runnable **Astral uv + Python + Mesa + NetworkX** starter for the proposed
misinformation project.

## Why Mesa and NetworkX?

- **Mesa** supplies agent/model structure and is a standard Python ABM framework.
- **NetworkX** generates and analyses the social network.
- The main graph is generated with `networkx.barabasi_albert_graph`, which produces a
  preferential-attachment network with heterogeneous degree.

Mesa is useful here, but the network itself should be generated with NetworkX. The code keeps
one Mesa agent per NetworkX node in `model.agent_by_node`.

## Game-theoretic element

The model is a **repeated verification-sharing game**, not a strict Iterated Prisoner's
Dilemma. Each exposed agent strategically chooses:

- `verify`, paying a private verification cost;
- `share`, avoiding that cost but risking penalties if the message is false.

For agent `i`, the four material payoffs are:

```text
pi_i(V,T) = a*n_i - c_v
pi_i(V,F) = d*n_i - c_v
pi_i(D,T) = a*n_i
pi_i(D,F) = a*n_i - p_det_i * (l_f*n_i + l_r*r_i)
```

Agents evaluate these outcomes using subjective truth beliefs, risk aversion, loss aversion,
and a logit decision rule. They then imitate locally successful neighbours using a Fermi rule.
That is sufficient to formalize strategic interaction using game theory.

## Install and run

From the project directory:

```bash
uv sync --extra dev
uv run pytest
```

Run one configuration:

```bash
uv run misinfo-abm run \
  --seed-mode hub \
  --hub-strategy cooperative \
  --payoff-mode normalized \
  --episodes 40
```

Run the minimum factorial experiment:

```bash
uv run misinfo-abm factorial --replications 20
```

This compares:

1. hub versus random seeding;
2. cooperative versus non-verifying hubs;
3. normalized versus accumulated payoff learning.

Generate basic plots:

```bash
uv run misinfo-abm plot
```

Run Morris screening:

```bash
uv run misinfo-abm morris --trajectories 20 --replications 5
```

## Suggested workflow for the nine-day project

1. First run with `n_agents=40`, `n_episodes=5`, and two replications while debugging.
2. Validate diffusion and recovery manually on fixed random seeds.
3. Freeze the model rules before starting the final experiments.
4. Use at least 20 to 30 stochastic replications per factorial condition.
5. Run Morris screening after the main model and metrics are stable.

## Important modelling decisions

- `random` seeding deliberately excludes hubs, so it creates a clean comparison with hub
  seeding.
- Hub strategies are fixed in the main treatment. Other agents adapt.
- Degree-normalized payoff is the recommended main specification.
- Accumulated payoff is retained as a robustness comparison.
- False beliefs can disappear through local correction and forgetting.
- All parameter values are assumptions. They must be justified and sensitivity-tested rather
  than presented as empirical facts.

## Project structure

```text
src/misinfo_abm/
  agent.py          agent decisions, utilities and payoffs
  config.py         all model parameters
  model.py          network, diffusion, recovery and learning
  experiments.py    factorial design
  sensitivity.py    Morris screening
  plotting.py       basic plots
  cli.py            command-line interface
tests/
  test_model.py
```
