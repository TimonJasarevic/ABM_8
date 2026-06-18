from enum import Enum

class NetworkType(Enum):
    ScaleFree = 1
    Random = 2

class MessageState(Enum):
    Unaware = 1
    Exposed = 2
    FalseBeliever = 3
    TrueBeliever = 4
    Corrected = 5