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
        history = []

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
            fake_believers = sum(a.message_state.name == "FalseBeliever" for a in self.social_network.social_agents)
            corrected = sum(a.message_state.name == "Corrected" for a in self.social_network.social_agents)
            mean_r = sum(a.r for a in self.social_network.social_agents) / len(self.social_network.social_agents)

            history.append({
                "round": round_number,
                "cascade_size": len(active_agents),
                "fake_believers": fake_believers,
                "corrected": corrected,
                "mean_reputation": mean_r,
                "rewired_edges": rewired_edges,
                "n_influencers": len(self.social_network.influencers),
            })

        self.plotter.show_network_graph()
        self.plotter.analyze_history(history, rolling_window=50, plot=True)


social_network = SocialNetwork(
    network_type=NetworkType.Random,
    n=300,
    p=4/60,
    m=4,
    influencer_th=60,
    truthfulness=0.5,
    seed=42
)

sim = Simulation(
    social_network=social_network,
    rounds=5000
)

sim.run()
