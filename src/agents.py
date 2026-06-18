import mesa
from definitions import MessageState


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

    def decide_cooperation(self, message, agent_id, agent_r):
        # based on agent params decide to verify or pass message
        pass

    def initiate_message(self):
        pass

    def share_message(self):
        pass

    def verify_message(self):
        pass


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