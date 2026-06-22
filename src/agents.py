import math

import mesa
from definitions import MessageState, Action


class SocialAgent(mesa.Agent):
    # node_id: networkx node id
    # d: degree
    # s: skepticism
    # r: reputation

    def __init__(self, model, node_id, d, s, r, neighbor_ids, is_influencer=False):
        super().__init__(model)

        self.node_id = node_id
        self.d = d
        self.s = s  # skepticism
        self.r = r  # reputation
        self.neighbor_ids = list(neighbor_ids)
        self.is_influencer = is_influencer
        self.message_state = MessageState.Unaware
        self.perceived_truthfulness = 0.5   # running estimate of P(message is true)
        self.n_observations = 0             # verified messages seen (for the 1/k learning rate)
        self.wealth = 0.0                   # accumulated payoff (verify cost / fake penalty)
        self.last_action = None
        self.n_cooperate = 0
        self.n_defect = 0

    


    def receive_message(self, message, agent_id, agent_r):
        # decide to verify based on perceived truthfulness and the cost/penalty trade-off
        if self.message_state == MessageState.Unaware:
            # check message
            if self.model.random.random() < self._verify_probability():
                self.wealth -= self.model.verify_cost          # pay cost c to verify
                self._update_perception(1.0 if message else 0.0)  # verification reveals the truth
                # message turned out true
                if message:
                    self.message_state = MessageState.TrueBeliever
                    self._record_action(Action.Cooperate)
                    return 1
                # message turned out false
                else:
                    self.message_state = MessageState.Corrected
                    self._record_action(Action.Cooperate)       # caught it, did not spread fake
                    return 0
            # dont check message
            else:
                # unaware message is false
                if not message:
                    self.message_state = MessageState.FalseBeliever
                    self.wealth -= self.model.fake_penalty      # spread fake news = defection
                    self._record_action(Action.Defect)
                    return 1
                # unaware message is true
                else:
                    self.message_state = MessageState.TrueBeliever
                    self._record_action(Action.Cooperate)       # spread true news, not fake
                    return 1

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




class NormalUser(SocialAgent):
    def __init__(self, model, node_id, d, s, r, neighbor_ids):
        super().__init__(
            model,
            node_id,
            d,
            s,
            r,
            neighbor_ids,
            is_influencer=False
        )


class Influencer(SocialAgent):
    def __init__(self, model, node_id, d, s, r, neighbor_ids):
        super().__init__(
            model,
            node_id,
            d,
            s,
            r,
            neighbor_ids,
            is_influencer=True
        )