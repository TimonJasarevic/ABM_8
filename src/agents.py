import mesa
from definitions import AgentType


class Agent():
    # id: agent id
    # d: degree
    # s: skepticism
    # r: reputation
    def __init__(self, id, d, s, r):
        self.id = id
        self.d = d
        self.s = s
        self.r = r

    def decide_cooperation(self, message):
        # based on agent params decide to verify or pass message
        pass


class NormalUser(Agent):
    def __init__(self, id, d, s, r):
        super().__init__(id, d, s, r)

class Influencer(Agent):
    def __init__(self, id, d, s, r):
        super().__init__(id, d, s, r)

    
    