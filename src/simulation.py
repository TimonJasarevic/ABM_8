

class Simulation():
    def __init__(self, social_network, rounds):
        self.social_network = social_network
        self.rounds = rounds


    def run(self):
        for round in range(self.rounds):
            active_agents = self.social_network.simulation_step(max_steps=20)
            self.social_network.update_agents(active_agents)