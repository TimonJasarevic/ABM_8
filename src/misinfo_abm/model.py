"""Mesa model for repeated misinformation diffusion."""

from __future__ import annotations

import math
from dataclasses import asdict

import networkx as nx
import numpy as np
import pandas as pd
from mesa import Model

from misinfo_abm.agent import SocialMediaAgent
from misinfo_abm.config import ModelConfig


class MisinformationModel(Model):
    """Repeated verification-sharing game on a social network.

    NetworkX supplies the graph. Mesa supplies model and agent lifecycle infrastructure.
    One agent is placed at each network node and stored in ``agent_by_node``.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.config.validate()
        super().__init__(seed=self.config.seed)

        self.graph = self._build_graph()
        self.hub_nodes, self.peripheral_nodes = self._classify_nodes()
        self.agent_by_node: dict[int, SocialMediaAgent] = {}
        self._create_agents()

        self.current_episode = 0
        self.episode_step = 0
        self.current_message_is_false = True
        self.current_seed_node = -1
        self.current_peak_false_prevalence = 0.0
        self.current_peak_step = 0
        self.current_ever_exposed: set[int] = set()
        self.current_ever_false_sharers: set[int] = set()
        self.tick_records: list[dict[str, float | int | str | bool]] = []
        self.episode_records: list[dict[str, float | int | str | bool]] = []
        self.running = True
        self._start_new_episode()

    def _build_graph(self) -> nx.Graph:
        cfg = self.config
        if cfg.network_type == "scale_free":
            return nx.barabasi_albert_graph(cfg.n_agents, cfg.m_links, seed=cfg.seed)

        # Match the expected average degree of the BA graph, approximately 2*m_links.
        probability = min(1.0, (2.0 * cfg.m_links) / (cfg.n_agents - 1))
        graph = nx.erdos_renyi_graph(cfg.n_agents, probability, seed=cfg.seed)
        if not nx.is_connected(graph):
            # Join components using one edge between consecutive components.
            components = [list(component) for component in nx.connected_components(graph)]
            for left, right in zip(components, components[1:], strict=False):
                graph.add_edge(left[0], right[0])
        return graph

    def _classify_nodes(self) -> tuple[set[int], set[int]]:
        degrees = sorted(self.graph.degree, key=lambda pair: (pair[1], pair[0]))
        n_hubs = max(1, math.ceil(self.config.hub_fraction * self.config.n_agents))
        n_peripheral = max(1, math.ceil(self.config.peripheral_fraction * self.config.n_agents))
        hub_nodes = {node for node, _ in degrees[-n_hubs:]}
        peripheral_nodes = {node for node, _ in degrees[:n_peripheral]}
        return hub_nodes, peripheral_nodes

    def _initial_theta(self, node_id: int) -> tuple[float, bool]:
        cfg = self.config
        if node_id in self.hub_nodes and cfg.hub_strategy == "cooperative":
            return cfg.cooperative_hub_theta, True
        if node_id in self.hub_nodes and cfg.hub_strategy == "non_verifying":
            return cfg.non_verifying_hub_theta, True
        theta = float(
            np.clip(
                self.rng.normal(cfg.initial_verification_mean, cfg.initial_verification_sd),
                0.01,
                0.99,
            )
        )
        return theta, False

    def _create_agents(self) -> None:
        for node_id in self.graph.nodes:
            theta, fixed = self._initial_theta(node_id)
            agent = SocialMediaAgent(
                self,
                node_id=node_id,
                verification_propensity=theta,
                is_hub=node_id in self.hub_nodes,
                fixed_strategy=fixed,
            )
            self.agent_by_node[node_id] = agent

    def neighbours_of(self, node_id: int) -> list[SocialMediaAgent]:
        return [self.agent_by_node[n] for n in self.graph.neighbors(node_id)]

    def _choose_seed_node(self) -> int:
        cfg = self.config
        if cfg.seed_mode == "hub":
            candidates = sorted(self.hub_nodes)
        elif cfg.seed_mode == "peripheral":
            candidates = sorted(self.peripheral_nodes)
        else:
            # Random means random non-hub, making the contrast with hub seeding explicit.
            candidates = sorted(set(self.graph.nodes) - self.hub_nodes)
        return int(self.random.choice(candidates))

    def _start_new_episode(self) -> None:
        for agent in self.agent_by_node.values():
            agent.reset_for_episode()

        self.current_episode += 1
        self.episode_step = 0
        self.current_message_is_false = (
            self.random.random() < self.config.false_message_probability
        )
        self.current_seed_node = self._choose_seed_node()
        self.agent_by_node[self.current_seed_node].state = "exposed"
        self.current_ever_exposed = {self.current_seed_node}
        self.current_ever_false_sharers = set()
        self.current_peak_false_prevalence = 0.0
        self.current_peak_step = 0

    def _reachable_unaware_neighbours(self, agent: SocialMediaAgent) -> list[SocialMediaAgent]:
        return [a for a in self.neighbours_of(agent.node_id) if a.state == "unaware"]

    def _decision_phase(self) -> tuple[set[int], set[int]]:
        """Make synchronous decisions and return nodes to expose and correct."""
        exposure_targets: set[int] = set()
        correction_sources: set[int] = set()
        exposed_agents = [a for a in self.agent_by_node.values() if a.state == "exposed"]
        self.random.shuffle(exposed_agents)

        for agent in exposed_agents:
            agent.update_belief_from_neighbours()
            reachable = self._reachable_unaware_neighbours(agent)
            action = agent.choose_action(len(reachable))
            agent.last_action = action
            agent.has_acted = True
            agent.realize_payoff(action, len(reachable), self.current_message_is_false)

            if action == "verify":
                agent.ever_verified_current_message = True
                if self.current_message_is_false:
                    agent.state = "corrected"
                    agent.belief_true = 0.0
                    correction_sources.add(agent.node_id)
                else:
                    agent.state = "shared"
                    agent.belief_true = 1.0
                    exposure_targets.update(a.node_id for a in reachable)
            else:
                agent.state = "shared"
                agent.ever_shared_current_message = True
                exposure_targets.update(a.node_id for a in reachable)
                if self.current_message_is_false:
                    self.current_ever_false_sharers.add(agent.node_id)

        return exposure_targets, correction_sources

    def _apply_exposures(self, exposure_targets: set[int]) -> None:
        for node_id in exposure_targets:
            target = self.agent_by_node[node_id]
            if target.state == "unaware":
                target.state = "exposed"
                self.current_ever_exposed.add(node_id)

    def _recovery_phase(self) -> None:
        """Correct or forget false beliefs, preventing an absorbing misinformation state."""
        if not self.current_message_is_false:
            return

        transitions_to_corrected: list[SocialMediaAgent] = []
        transitions_to_unaware: list[SocialMediaAgent] = []
        cfg = self.config

        for agent in self.agent_by_node.values():
            if agent.state != "shared":
                continue
            n_corrected = sum(1 for n in self.neighbours_of(agent.node_id) if n.state == "corrected")
            p_correction = 1.0 - (1.0 - cfg.correction_probability) ** n_corrected
            if self.random.random() < p_correction:
                transitions_to_corrected.append(agent)
            elif self.random.random() < cfg.forgetting_probability:
                transitions_to_unaware.append(agent)

        for agent in transitions_to_corrected:
            agent.state = "corrected"
            agent.belief_true = 0.0
        for agent in transitions_to_unaware:
            agent.state = "unaware"
            agent.belief_true = cfg.prior_truth_belief

    def _record_tick(self) -> None:
        n = self.config.n_agents
        false_prevalence = (
            sum(1 for a in self.agent_by_node.values() if a.state == "shared") / n
            if self.current_message_is_false
            else 0.0
        )
        if false_prevalence > self.current_peak_false_prevalence:
            self.current_peak_false_prevalence = false_prevalence
            self.current_peak_step = self.episode_step

        acted = [a for a in self.agent_by_node.values() if a.has_acted]
        cooperation_rate = (
            sum(a.last_action == "verify" for a in acted) / len(acted) if acted else 0.0
        )
        objective_truth = 0.0 if self.current_message_is_false else 1.0
        belief_error = sum(
            abs(a.belief_true - objective_truth) for a in self.agent_by_node.values()
        ) / n

        self.tick_records.append(
            {
                "episode": self.current_episode,
                "step": self.episode_step,
                "message_is_false": self.current_message_is_false,
                "seed_mode": self.config.seed_mode,
                "seed_node": self.current_seed_node,
                "seed_is_hub": self.current_seed_node in self.hub_nodes,
                "hub_strategy": self.config.hub_strategy,
                "payoff_mode": self.config.payoff_mode,
                "false_prevalence": false_prevalence,
                "cooperation_rate": cooperation_rate,
                "belief_error": belief_error,
                "exposed_fraction": len(self.current_ever_exposed) / n,
                "mean_theta": sum(
                    a.verification_propensity for a in self.agent_by_node.values()
                )
                / n,
            }
        )

    def _episode_is_finished(self) -> bool:
        no_pending_exposure = not any(
            a.state == "exposed" for a in self.agent_by_node.values()
        )
        if not self.current_message_is_false:
            return no_pending_exposure
        no_active_false_belief = not any(
            a.state == "shared" for a in self.agent_by_node.values()
        )
        return no_pending_exposure and no_active_false_belief

    def _learning_payoff(self, agent: SocialMediaAgent) -> float:
        if self.config.payoff_mode == "accumulated":
            return agent.episode_payoff
        return agent.episode_payoff / max(1, agent.degree)

    def _strategy_update(self) -> None:
        """Synchronous Fermi imitation of a randomly selected neighbour."""
        cfg = self.config
        proposed: dict[int, float] = {}
        for agent in self.agent_by_node.values():
            if agent.fixed_strategy:
                continue
            neighbours = self.neighbours_of(agent.node_id)
            if not neighbours:
                continue
            role_model = self.random.choice(neighbours)
            difference = self._learning_payoff(role_model) - self._learning_payoff(agent)
            imitation_probability = 1.0 / (1.0 + math.exp(-cfg.imitation_strength * difference))
            theta = agent.verification_propensity
            if self.random.random() < imitation_probability:
                theta = (
                    (1.0 - cfg.strategy_learning_rate) * theta
                    + cfg.strategy_learning_rate * role_model.verification_propensity
                )
            theta += float(self.rng.normal(0.0, cfg.strategy_mutation_sd))
            proposed[agent.node_id] = float(np.clip(theta, 0.01, 0.99))

        for node_id, theta in proposed.items():
            self.agent_by_node[node_id].verification_propensity = theta

    def _finalize_episode(self) -> None:
        n = self.config.n_agents
        acted = [a for a in self.agent_by_node.values() if a.has_acted]
        cooperation_rate = (
            sum(a.last_action == "verify" for a in acted) / len(acted) if acted else 0.0
        )
        objective_truth = 0.0 if self.current_message_is_false else 1.0
        belief_error = sum(
            abs(a.belief_true - objective_truth) for a in self.agent_by_node.values()
        ) / n
        final_false_prevalence = (
            sum(1 for a in self.agent_by_node.values() if a.state == "shared") / n
            if self.current_message_is_false
            else 0.0
        )
        recovery_time = max(0, self.episode_step - self.current_peak_step)

        self.episode_records.append(
            {
                "episode": self.current_episode,
                "message_is_false": self.current_message_is_false,
                "seed_mode": self.config.seed_mode,
                "seed_node": self.current_seed_node,
                "seed_degree": int(self.graph.degree[self.current_seed_node]),
                "seed_is_hub": self.current_seed_node in self.hub_nodes,
                "hub_strategy": self.config.hub_strategy,
                "payoff_mode": self.config.payoff_mode,
                "network_type": self.config.network_type,
                "cascade_size": len(self.current_ever_false_sharers) / n
                if self.current_message_is_false
                else 0.0,
                "exposure_fraction": len(self.current_ever_exposed) / n,
                "peak_false_prevalence": self.current_peak_false_prevalence,
                "final_false_prevalence": final_false_prevalence,
                "persistence_steps": self.episode_step,
                "recovery_time": recovery_time,
                "cooperation_rate": cooperation_rate,
                "belief_error": belief_error,
                "mean_theta": sum(
                    a.verification_propensity for a in self.agent_by_node.values()
                )
                / n,
                "mean_payoff": sum(a.episode_payoff for a in self.agent_by_node.values()) / n,
            }
        )

        for agent in self.agent_by_node.values():
            agent.update_reputation(self.current_message_is_false)
            agent.aspiration = (
                (1.0 - self.config.aspiration_learning_rate) * agent.aspiration
                + self.config.aspiration_learning_rate * agent.episode_payoff
            )
        self._strategy_update()

    def step(self) -> None:
        if not self.running:
            return

        self.episode_step += 1
        exposure_targets, _ = self._decision_phase()
        self._apply_exposures(exposure_targets)
        self._recovery_phase()
        self._record_tick()

        finished = self._episode_is_finished() or (
            self.episode_step >= self.config.max_steps_per_episode
        )
        if not finished:
            return

        self._finalize_episode()
        if self.current_episode >= self.config.n_episodes:
            self.running = False
        else:
            self._start_new_episode()

    def run_model(self) -> None:
        while self.running:
            self.step()

    def episode_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_records)

    def tick_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.tick_records)

    def configuration_dict(self) -> dict[str, object]:
        return asdict(self.config)
