"""Agent definition and behavioural equations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from mesa import Agent

if TYPE_CHECKING:
    from misinfo_abm.model import MisinformationModel

Action = Literal["verify", "share"]
MessageState = Literal["unaware", "exposed", "shared", "corrected"]


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _sigmoid(value: float) -> float:
    # Stable enough for the parameter ranges used here.
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class SocialMediaAgent(Agent):
    """A boundedly rational social-media user located at one network node."""

    def __init__(
        self,
        model: "MisinformationModel",
        node_id: int,
        verification_propensity: float,
        is_hub: bool,
        fixed_strategy: bool,
    ) -> None:
        super().__init__(model)
        cfg = model.config
        self.node_id = node_id
        self.is_hub = is_hub
        self.fixed_strategy = fixed_strategy
        self.verification_propensity = _clip01(verification_propensity)
        self.reputation = cfg.initial_reputation
        self.aspiration = 0.0

        self.state: MessageState = "unaware"
        self.belief_true = cfg.prior_truth_belief
        self.has_acted = False
        self.last_action: Action | None = None
        self.episode_payoff = 0.0
        self.total_payoff = 0.0
        self.ever_shared_current_message = False
        self.ever_verified_current_message = False

    @property
    def degree(self) -> int:
        return int(self.model.graph.degree[self.node_id])

    def reset_for_episode(self) -> None:
        self.state = "unaware"
        self.belief_true = self.model.config.prior_truth_belief
        self.has_acted = False
        self.last_action = None
        self.episode_payoff = 0.0
        self.ever_shared_current_message = False
        self.ever_verified_current_message = False

    def prospect_value(self, outcome_relative_to_reference: float) -> float:
        """Prospect-style value function for risk and loss aversion."""
        cfg = self.model.config
        if outcome_relative_to_reference >= 0:
            return outcome_relative_to_reference**cfg.risk_alpha
        return -cfg.loss_aversion * ((-outcome_relative_to_reference) ** cfg.risk_alpha)

    def detection_probability(self) -> float:
        """Probability that false sharing is detected by the local neighbourhood."""
        neighbours = self.model.neighbours_of(self.node_id)
        if not neighbours:
            return self.model.config.base_detection_probability
        local_monitoring = sum(a.verification_propensity for a in neighbours) / len(neighbours)
        cfg = self.model.config
        return _clip01(
            cfg.base_detection_probability + cfg.social_detection_strength * local_monitoring
        )

    def state_contingent_payoffs(self, reachable_neighbours: int) -> dict[str, float]:
        """Return the four payoffs used in the verification-sharing game.

        V,T: verify a true message
        V,F: verify a false message
        D,T: share a true message without verification
        D,F: share a false message without verification
        """
        cfg = self.model.config
        attention = cfg.attention_benefit * reachable_neighbours
        return {
            "V,T": attention - cfg.verification_cost,
            "V,F": cfg.correction_reward * reachable_neighbours - cfg.verification_cost,
            "D,T": attention,
            "D,F": attention
            - self.detection_probability()
            * (
                cfg.false_penalty_per_neighbor * reachable_neighbours
                + cfg.reputation_penalty * self.reputation
            ),
        }

    def expected_utilities(self, reachable_neighbours: int) -> tuple[float, float]:
        """Return subjective utilities (verify, share without verification)."""
        p = self.state_contingent_payoffs(reachable_neighbours)
        b = _clip01(self.belief_true)
        utility_verify = b * self.prospect_value(p["V,T"] - self.aspiration) + (
            1.0 - b
        ) * self.prospect_value(p["V,F"] - self.aspiration)
        utility_share = b * self.prospect_value(p["D,T"] - self.aspiration) + (
            1.0 - b
        ) * self.prospect_value(p["D,F"] - self.aspiration)
        return utility_verify, utility_share

    def probability_verify(self, reachable_neighbours: int) -> float:
        """Boundedly rational logit choice with an adaptive behavioural habit."""
        cfg = self.model.config
        u_verify, u_share = self.expected_utilities(reachable_neighbours)
        theta = min(1.0 - 1e-6, max(1e-6, self.verification_propensity))
        habit_log_odds = math.log(theta / (1.0 - theta))
        decision_index = (
            cfg.decision_sensitivity * (u_verify - u_share)
            + cfg.habit_weight * habit_log_odds
        )
        return _sigmoid(decision_index)

    def choose_action(self, reachable_neighbours: int) -> Action:
        probability = self.probability_verify(reachable_neighbours)
        return "verify" if self.model.random.random() < probability else "share"

    def realize_payoff(self, action: Action, reachable_neighbours: int, is_false: bool) -> float:
        p = self.state_contingent_payoffs(reachable_neighbours)
        key = ("V" if action == "verify" else "D") + (",F" if is_false else ",T")
        payoff = p[key]
        self.episode_payoff += payoff
        self.total_payoff += payoff
        return payoff

    def update_belief_from_neighbours(self) -> None:
        """Update belief using reputation-weighted support and correction signals."""
        neighbours = self.model.neighbours_of(self.node_id)
        support = sum(a.reputation for a in neighbours if a.state == "shared")
        correction = sum(a.reputation for a in neighbours if a.state == "corrected")
        total_signal = support + correction
        if total_signal <= 0:
            return
        local_signal = support / total_signal
        mu = self.model.config.belief_learning_rate
        self.belief_true = _clip01((1.0 - mu) * self.belief_true + mu * local_signal)

    def update_reputation(self, message_is_false: bool) -> None:
        """Reward accurate behaviour and penalize false sharing."""
        rate = 0.08
        accurate = (self.last_action == "verify") or (
            self.last_action == "share" and not message_is_false
        )
        false_share = self.last_action == "share" and message_is_false
        if accurate:
            self.reputation = _clip01(self.reputation + rate * (1.0 - self.reputation))
        elif false_share:
            self.reputation = _clip01(self.reputation - rate * self.reputation)
