import mesa
import networkx as nx

from mesa.space import NetworkGrid
from definitions import NetworkType
import agents


class SocialNetwork(mesa.Model):
    def __init__(self, network_type, n, p, m, influencer_th=10, seed=None):
        super().__init__(seed=seed)

        self.n = n
        self.p = p
        self.m = m
        self.influencer_th = influencer_th

        self.G = None
        self.grid = None
        self.influencer_nodes = []

        self.social_agents = []

        if network_type == NetworkType.Random:
            self.G = nx.erdos_renyi_graph(n, p)

        elif network_type == NetworkType.ScaleFree:
            G_without_influencer = True
            tries = 0

            while G_without_influencer and tries < 50:
                self.G = nx.barabasi_albert_graph(n, m)
                influencers = self._check_for_influencers()

                if len(influencers) != 0:
                    G_without_influencer = False
                    self.influencer_nodes = influencers

                tries += 1

            if G_without_influencer:
                print("Ease the influencer threshold or initialize more nodes\n")

        self.grid = NetworkGrid(self.G)
        self._init_agents()

    def _check_for_influencers(self):
        top_fraction = 0.05
        number_of_influencers = max(1, int(top_fraction * self.G.number_of_nodes()))

        sorted_degree_nodes = sorted(
            self.G.nodes(),
            key=lambda node: self.G.degree[node],
            reverse=True
        )

        top_degree_nodes = sorted_degree_nodes[:number_of_influencers]

        influencers = [
            node for node in top_degree_nodes
            if self.G.degree[node] >= self.influencer_th
        ]

        return influencers

    def _init_agents(self):
        for node in self.G.nodes():
            neighbor_ids = self.G.neighbors(node)
            agent = None

            if node in self.influencer_nodes:
                agent = agents.Influencer(
                    self,
                    node,
                    self.G.degree[node],
                    0.5,
                    1,
                    neighbor_ids
                )

            else:
                agent = agents.NormalUser(
                    self,
                    node,
                    self.G.degree[node],
                    0.5,
                    1,
                    neighbor_ids
                )

            self.grid.place_agent(agent, node)
            self.social_agents.append(agent)