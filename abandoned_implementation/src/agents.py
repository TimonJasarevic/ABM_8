import math

import mesa
from definitions import MessageState, Action


class SocialAgent(mesa.Agent):
    # node_id: networkx node id
    # r: reputation (== accumulated payoff with decay)

    def __init__(self, model, node_id, r, is_influencer=False):
        super().__init__(model)

        # Inits
        self.node_id = node_id
        self.r = r  # reputation == accumulated payoff with decay
        self.is_influencer = is_influencer

        # States
        self.message_state = MessageState.Unaware
        self.last_action = None

        # Parameters
        self.perceived_truthfulness = 0.5   # running estimate of P(message is true)
        self.reputation_noise = 1 / 10
        self.reputation_decay = 0.99
        self.risk_aversion = 0.25

        # Sender-specific trust:
        # Each receiver can trust specific neighbors differently.
        self.trust = {}                     # sender_id -> trust level
        self.default_trust = 0.5
        self.trust_learning_rate = self.model.trust_learning_rate
        self.trust_weight = self.model.trust_weight

        # Adaptive fake-news tendency:
        # If fake news gives a sender reputation, the sender becomes more likely
        # to initiate fake news in future rounds.
        self.fake_tendency = 1.0 - self.model.truthfulness
        self.fake_learning_rate = self.model.fake_learning_rate

        # Counters
        self.n_observations = 0             # verified messages seen
        self.n_cooperate = 0
        self.n_defect = 0
        self.n_discard = 0

        # Payoffs
        self.rep_gain = 0.5
        self.rep_loss = 1.0

        self.receiver_true_reward = self.rep_gain
        self.receiver_fake_penalty = self.rep_loss

        self.sender_true_reward = self.rep_gain
        self.sender_fake_reward = 0.2
        self.sender_fake_penalty = self.rep_loss

        self.detect_fake_reward = 0.5

    def receive_message(self, message, sender_id):
        # only the first exposure matters
        if self.message_state != MessageState.Unaware:
            return (0, 0)

        sender_r_change = 0

        # decide to verify based on perceived truthfulness, trust in this sender,
        # and the cost/penalty trade-off
        if self.model.random.random() < self._verify_probability(sender_id):
            # pay the verification cost into reputation and the model aggregate
            self.model.total_verification_cost_paid += self.model.verify_cost
            self._update_perception(1.0 if message else 0.0)  # verification reveals the truth
            self._update_trust(sender_id, 1.0 if message else 0.0)

            if message:
                # Verify true message:
                # receiver pays verification cost, sender gains reputation
                self.update_reputation(-self.model.verify_cost)
                sender_r_change = self.sender_true_reward
                self.message_state = MessageState.TrueBeliever
                self._record_action(Action.Cooperate)
                return (1, sender_r_change)

            else:
                # Verify fake message:
                # receiver is rewarded for detecting fake news, sender is punished
                self.update_reputation(
                    self.detect_fake_reward - self.model.verify_cost
                )
                sender_r_change = -self.sender_fake_penalty
                self.message_state = MessageState.Corrected
                self._record_action(Action.Cooperate)
                return (0, sender_r_change)

        # did not verify
        else:
            # Discard if sharing has negative expected value.
            # This models passive non-engagement: the agent does not pay verification cost,
            # does not learn the true state, does not punish the sender, and does not share.
            expected_share_value = self._expected_share_value(sender_id)

            if expected_share_value <= 0:
                self.message_state = MessageState.Discarded
                self._record_action(Action.Discard)
                return (0, 0)

            if message:
                # Share true message:
                # receiver and sender both gain reputation
                self.update_reputation(self.receiver_true_reward)
                sender_r_change = self.sender_true_reward
                self.message_state = MessageState.TrueBeliever
                self._record_action(Action.Cooperate)
                return (1, sender_r_change)

            else:
                # Share fake message:
                # receiver is punished, sender benefits
                self.update_reputation(-self.receiver_fake_penalty)
                sender_r_change = self.sender_fake_reward
                self.message_state = MessageState.FalseBeliever
                self._record_action(Action.Defect)
                return (1, sender_r_change)

    def update_reputation(self, payoff_change):
        """
        Reputation update with decay.

        reputation_decay = 0.99 means:
        new reputation = 0.99 * old reputation + payoff_change

        This prevents reputation from becoming pure lifetime accumulation.
        """
        self.r = self.reputation_decay * self.r + payoff_change

    def get_trust(self, sender_id):
        """
        Return the receiver's trust in a specific sender.

        If the receiver has no previous verified history with this sender,
        use default_trust.
        """
        return self.trust.get(sender_id, self.default_trust)

    def _subjective_truthfulness(self, sender_id):
        """
        Combine general perceived_truthfulness with trust in this specific sender.

        trust_weight = 0 means only general perceived_truthfulness matters.
        trust_weight = 1 means only sender-specific trust matters.
        """
        sender_trust = self.get_trust(sender_id)

        subjective_truthfulness = (
            (1.0 - self.trust_weight) * self.perceived_truthfulness
            + self.trust_weight * sender_trust
        )

        return min(1.0, max(0.0, subjective_truthfulness))

    def _update_trust(self, sender_id, observation):
        """
        Update trust in this specific sender after verification.

        observation = 1.0 means the sender sent true news.
        observation = 0.0 means the sender sent fake news.
        """
        current_trust = self.get_trust(sender_id)

        updated_trust = current_trust + self.trust_learning_rate * (
            observation - current_trust
        )

        self.trust[sender_id] = min(1.0, max(0.0, updated_trust))

    def adapt_fake_tendency(self, message, sender_r_change):
        """
        Sender-side adaptation.

        If fake news gives positive reputation, the sender becomes more likely
        to initiate fake news later. If fake news is punished, the sender becomes
        less likely to initiate fake news.
        """

        # Only fake messages update fake_tendency.
        if message:
            return

        if sender_r_change > 0:
            self.fake_tendency += self.fake_learning_rate * (
                1.0 - self.fake_tendency
            )

        elif sender_r_change < 0:
            self.fake_tendency += self.fake_learning_rate * (
                0.0 - self.fake_tendency
            )

        self.fake_tendency = min(1.0, max(0.0, self.fake_tendency))

    def _expected_share_value(self, sender_id):
        """
        Risk-averse expected payoff from sharing without verification.

        The agent does not know whether the message is true or fake.
        It uses a combination of perceived_truthfulness and sender-specific trust.

        Risk aversion is modeled as:

            risk_adjusted_value = expected_payoff - risk_aversion * variance

        So sharing becomes less attractive when the outcome is uncertain.
        """

        p_true = self._subjective_truthfulness(sender_id)

        payoff_if_true = self.receiver_true_reward
        payoff_if_fake = -self.receiver_fake_penalty

        expected_payoff = (
            p_true * payoff_if_true
            + (1.0 - p_true) * payoff_if_fake
        )

        payoff_variance = (
            p_true * (payoff_if_true - expected_payoff) ** 2
            + (1.0 - p_true) * (payoff_if_fake - expected_payoff) ** 2
        )

        risk_adjusted_value = (
            expected_payoff
            - self.risk_aversion * payoff_variance
        )

        return risk_adjusted_value

    def _verify_probability(self, sender_id):
        # verify more when this sender is perceived as less truthful and the fake-news penalty
        # is large relative to the verification cost
        subjective_truthfulness = self._subjective_truthfulness(sender_id)
        expected_loss = (1.0 - subjective_truthfulness) * self.receiver_fake_penalty

        return 1.0 / (
            1.0 + math.exp(
                -self.model.rationality * (expected_loss - self.model.verify_cost)
            )
        )

    def _update_perception(self, observation):
        # running mean of observed truth values: infinite memory, decreasing learning rate 1/k
        self.n_observations += 1
        self.perceived_truthfulness += (
            observation - self.perceived_truthfulness
        ) / self.n_observations

    def _record_action(self, action):
        self.last_action = action

        if action == Action.Cooperate:
            self.n_cooperate += 1
        elif action == Action.Defect:
            self.n_defect += 1
        elif action == Action.Discard:
            self.n_discard += 1

    def initiate_message(self):
        # message is true with probability 1 - fake_tendency.
        # fake_tendency adapts when fake news is rewarded or punished.
        message_is_fake = self.model.random.random() < self.fake_tendency
        message = not message_is_fake

        if message:
            self.message_state = MessageState.TrueBeliever
        else:
            self.message_state = MessageState.FalseBeliever

        return message

    def update(self):
        pass

    def observe_reputation(self, agent):
        sigma = max(0.1, abs(agent.r) * self.reputation_noise)
        noise = self.model.random.gauss(0, sigma)
        return agent.r + noise
