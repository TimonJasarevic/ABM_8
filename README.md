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


## Install and run

From the project directory:

```bash
uv sync --extra dev
uv run pytest
```

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
