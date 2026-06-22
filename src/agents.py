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



    def pay_verification_cost(self):
        # cost paid when the agent uses verification technology.
        
        self.add_payoff(-self.model.verification_cost)
        self.model.total_verification_cost_paid += self.model.verification_cost    


                

    def receive_message(self, message, agent_id, agent_r):
        # agent only reacts if it has not seen the message before
        if self.message_state != MessageState.Unaware:
            return 0

        # verification probability using skepticism
        verify_probability = self.s

        # agent decides whether to verify the message
        if self.model.random.random() < verify_probability:

            # pay verification cost
            self.pay_verification_cost()

            # verification reveals whether the message is true or false
            if message:
                self.message_state = MessageState.TrueBeliever
                return 1
            else:
                self.message_state = MessageState.Corrected
                return 0

        # agent does not verify
        else:
            if message:
                self.message_state = MessageState.TrueBeliever
                return 1
            else:
                self.message_state = MessageState.FalseBeliever
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