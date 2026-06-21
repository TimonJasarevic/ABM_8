import mesa
from definitions import MessageState


class SocialAgent(mesa.Agent):
    # node_id: networkx node id
    # d: degree
    # s: skepticism
    # r: reputation

    def __init__(self, model, node_id, s, r, is_influencer=False):
        super().__init__(model)

        self.node_id = node_id
        self.s = s  # skepticism
        self.r = r  # reputation
        self.is_influencer = is_influencer
        self.message_state = MessageState.Unaware

    


    def receive_message(self, message, agent_id, agent_r):
        # based on agent params decide to verify or pass message
        if self.message_state == MessageState.Unaware:
            # check message
            if self.model.random.random() < 0.4:
                # message turned out true
                if message:
                    self.message_state = MessageState.TrueBeliever
                    return 1
                # message turned out false
                else:
                    self.message_state = MessageState.Corrected
                    return 0
            # dont check message
            else:
                # unaware message is false
                if not message:
                    self.message_state = MessageState.FalseBeliever
                    return 1
                # unaware message is true
                else:
                    self.message_state = MessageState.TrueBeliever
                    return 1
            
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



class NormalUser(SocialAgent):
    def __init__(self, model, node_id, s, r):
        super().__init__(
            model,
            node_id,
            s,
            r,
            is_influencer=False
        )


class Influencer(SocialAgent):
    def __init__(self, model, node_id, s, r):
        super().__init__(
            model,
            node_id,
            s,
            r,
            is_influencer=True
        )