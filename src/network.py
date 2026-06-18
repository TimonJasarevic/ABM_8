import networkx as nx
from definitions import AgentType, NetworkType


class SocialNetwork():
    def __init__(self, network_type, n, p=None, m=None, influencer_th=10):
        self.n = n
        self.p = p
        self.m = m
        self.influencer_th = influencer_th
        self.G = None
        self.influencers = []

        if network_type == NetworkType.Random:
            self.G = nx.erdos_renyi_graph(n,p)

        elif network_type == NetworkType.ScaleFree:
            G_without_influencer = True
            tries = 0
            while G_without_influencer and tries < 50:
                self.G = nx.barabasi_albert_graph(n,m)
                influencers = self._check_for_influencers()
                if len(influencers) != 0:
                    G_without_influencer = False
                    self.influencers = influencers
                tries += 1
            if G_without_influencer:
                print("Ease the influencer threshold or initialize more nodes\n")
        self._init_agents()

    def _check_for_influencers(self):
        top_fraction = 0.05
        number_of_influencers = max(1, int(top_fraction * self.G.number_of_nodes()))

        sorted_degree_nodes = sorted(self.G.nodes(), key=lambda node: self.G.degree[node], reverse=True)
        top_degree_nodes = sorted_degree_nodes[:number_of_influencers]
        influencers = [node for node in top_degree_nodes if self.G.degree[node] >= self.influencer_th]
        return influencers
    
    def _init_agents(self):
        for node in self.G.nodes():
            if node in self.influencers:
                # assign influencer stats
                pass
            else:
                # assign regular stats
                pass
        