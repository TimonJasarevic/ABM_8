import math

import mesa
from definitions import MessageState, Action


class SocialAgent(mesa.Agent):
    # node_id: networkx node id
    # r: reputation (== accumulated payoff: verify cost + fake-news penalty)

    def __init__(self, model, node_id, r, is_influencer=False):
        super().__init__(model)

        # Inits
        self.node_id = node_id
        self.r = r  # reputation == accumulated payoff
        self.is_influencer = is_influencer

        # States
        self.message_state = MessageState.Unaware
        self.last_action = None

        # Parameters
        self.perceived_truthfulness = 0.5   # running estimate of P(message is true)
        self.reputation_noise = 0.1

        # Counters
        self.n_observations = 0             # verified messages seen (for the 1/k learning rate)
        self.n_cooperate = 0
        self.n_defect = 0

        #Payoffs
        self.detect_fake_reward=1.0
        self.receiver_true_reward=0.5
        self.receiver_fake_penalty=0.8
        self.sender_true_reward=0.5
        self.sender_fake_reward=0.3
        self.sender_fake_penalty=1.2

    def receive_message(self, message, sender_id):
        # only the first exposure matters
        if self.message_state != MessageState.Unaware:
            return (0, 0)


        sender_r_change = 0
        # decide to verify based on perceived truthfulness and the cost/penalty trade-off
        if self.model.random.random() < self._verify_probability():
            # pay the verification cost into reputation (== wealth) and the model aggregate
            self.model.total_verification_cost_paid += self.model.verify_cost
            self._update_perception(1.0 if message else 0.0)  # verification reveals the truth

            if message:
                # Verify true message:
                # receiver pays verification cost, sender gains reputation
                self.r -= self.model.verify_cost
                sender_r_change = self.sender_true_reward
                self.message_state = MessageState.TrueBeliever
                self._record_action(Action.Cooperate)
                return (1, sender_r_change)
            else:
                # Verify fake message:
                # receiver is rewarded for detecting fake news, sender is punished
                self.r += self.detect_fake_reward - self.model.verify_cost
                sender_r_change = -self.sender_fake_penalty
                self.message_state = MessageState.Corrected
                self._record_action(Action.Cooperate)       # caught it, did not spread fake
                return (0, sender_r_change)

        # did not verify
        else:
            if message:
                # Share true message:
                # receiver and sender both gain reputation
                self.r += self.receiver_true_reward
                sender_r_change = self.sender_true_reward
                self.message_state = MessageState.TrueBeliever
                self._record_action(Action.Cooperate)       # spread true news, not fake
                return (1, sender_r_change)
            else:
                # Share fake message:
                # receiver is punished or receives no payoff, sender benefits
                self.r -= self.receiver_fake_penalty
                sender_r_change = self.sender_fake_reward
                self.message_state = MessageState.FalseBeliever
                self._record_action(Action.Defect)
                return (1, sender_r_change)

    def _verify_probability(self):
        # verify more when the world is perceived as less truthful and the fake-news penalty
        # is large relative to the verification cost (stochastic best response)
        expected_loss = (1.0 - self.perceived_truthfulness) * self.model.fake_penalty
        return 1.0 / (1.0 + math.exp(-self.model.rationality * (expected_loss - self.model.verify_cost)))

    def _update_perception(self, observation):
        # running mean of observed truth values: infinite memory, decreasing learning rate 1/k
        self.n_observations += 1
        self.perceived_truthfulness += (observation - self.perceived_truthfulness) / self.n_observations

    def _record_action(self, action):
        self.last_action = action
        if action == Action.Cooperate:
            self.n_cooperate += 1
        else:
            self.n_defect += 1

    def initiate_message(self):
        # message is true with probability truthfulness, fake otherwise
        message = self.model.random.random() < self.model.truthfulness
        if message:
            self.message_state = MessageState.TrueBeliever
        else:
            self.message_state = MessageState.FalseBeliever
        return message

    def update(self):
        pass

    def observe_reputation(self, agent):
        noise = self.model.random.gauss(0, self.reputation_noise)
        return agent.r + noise