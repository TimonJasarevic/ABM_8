from enum import Enum

class NetworkType(Enum):
    ScaleFree = 1
    Random = 2

class MessageState(Enum):
    Unaware = 1        # has not seen the message
    FalseBeliever = 2  # believes a false message
    TrueBeliever = 3   # believes a true message
    Corrected = 4      # knows the message is false

class AgentType(Enum):
    NormalUser = 1
    Influencer = 2

class Action(Enum):
    Cooperate = 1   # did not spread fake news (verified it, or passed on true news)
    Defect = 2      # spread fake news