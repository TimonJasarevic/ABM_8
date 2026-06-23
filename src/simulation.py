import networkx as nx
from plotter import NetworkPlotter

from network import SocialNetwork
from definitions import NetworkType


class Simulation:
    def __init__(self, social_network, rounds):
        self.social_network = social_network
        self.rounds = rounds
        self.plotter = NetworkPlotter(social_network)


    def run(self):
        for round_number in range(self.rounds):
            active_agents = self.social_network.simulation_step(max_steps=20)

            self.social_network.update_agents(active_agents)
            rewired_edges = self.social_network.rewire_network(rewire_prob=0.005)
            if round_number % 10 == 0:
                self.social_network.update_influencers()

            self.plotter.draw_network_graph(
                title=f"Round {round_number + 1}, rewired edges: {rewired_edges}",
                label_type="reputation"
            )
        self.plotter.show_network_graph()



social_network = SocialNetwork(
    network_type=NetworkType.Random,
    n=200,
    p=4/59,
    m=4,
    influencer_th=50,
    truthfulness=0.5,
    seed=42
)

sim = Simulation(
    social_network=social_network,
    rounds=1000
)

sim.run()