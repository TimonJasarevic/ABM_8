"""Configuration objects for the misinformation ABM."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

NetworkType = Literal["scale_free", "erdos_renyi"]
SeedMode = Literal["hub", "random", "peripheral"]
HubStrategy = Literal["cooperative", "non_verifying", "adaptive"]
PayoffMode = Literal["normalized", "accumulated"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """All tunable parameters used by a model run.

    The defaults are deliberately normalized rather than empirically calibrated. Treat them
    as assumptions and test them through factorial experiments and sensitivity analysis.
    """

    # Population and network
    n_agents: int = 200
    network_type: NetworkType = "scale_free"
    m_links: int = 3
    hub_fraction: float = 0.05
    peripheral_fraction: float = 0.25

    # Experimental treatments
    seed_mode: SeedMode = "random"
    hub_strategy: HubStrategy = "cooperative"
    payoff_mode: PayoffMode = "normalized"
    cooperative_hub_theta: float = 0.90
    non_verifying_hub_theta: float = 0.10

    # Repeated message process
    n_episodes: int = 40
    max_steps_per_episode: int = 30
    false_message_probability: float = 0.50

    # Initial agent heterogeneity
    initial_verification_mean: float = 0.50
    initial_verification_sd: float = 0.12
    initial_reputation: float = 0.50
    prior_truth_belief: float = 0.50

    # Material payoff parameters
    attention_benefit: float = 0.08
    verification_cost: float = 0.30
    correction_reward: float = 0.10
    false_penalty_per_neighbor: float = 0.22
    reputation_penalty: float = 0.70
    base_detection_probability: float = 0.05
    social_detection_strength: float = 0.80

    # Behavioural decision parameters
    risk_alpha: float = 0.88
    loss_aversion: float = 2.00
    decision_sensitivity: float = 2.50
    habit_weight: float = 0.80
    belief_learning_rate: float = 0.45

    # Recovery
    correction_probability: float = 0.25
    forgetting_probability: float = 0.05

    # Adaptation
    imitation_strength: float = 2.00
    strategy_learning_rate: float = 0.20
    aspiration_learning_rate: float = 0.15
    strategy_mutation_sd: float = 0.015

    # Reproducibility
    seed: int = 42

    def validate(self) -> None:
        """Raise ValueError for invalid parameter combinations."""
        if self.n_agents < 5:
            raise ValueError("n_agents must be at least 5")
        if not 1 <= self.m_links < self.n_agents:
            raise ValueError("m_links must satisfy 1 <= m_links < n_agents")
        for name in (
            "hub_fraction",
            "peripheral_fraction",
            "false_message_probability",
            "cooperative_hub_theta",
            "non_verifying_hub_theta",
            "initial_verification_mean",
            "initial_reputation",
            "prior_truth_belief",
            "base_detection_probability",
            "correction_probability",
            "forgetting_probability",
            "belief_learning_rate",
            "strategy_learning_rate",
            "aspiration_learning_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0.0 < self.risk_alpha <= 1.0:
            raise ValueError("risk_alpha must satisfy 0 < risk_alpha <= 1")
        if self.loss_aversion < 1.0:
            raise ValueError("loss_aversion must be at least 1")
        if self.n_episodes < 1 or self.max_steps_per_episode < 1:
            raise ValueError("episode counts must be positive")

    def with_updates(self, **kwargs: object) -> "ModelConfig":
        """Return a validated copy with selected fields changed."""
        updated = replace(self, **kwargs)
        updated.validate()
        return updated
