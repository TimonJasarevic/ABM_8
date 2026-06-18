import networkx as nx



class SocialNetwork():
    def __init__(self, network_type, n, p=None, m=None, influencer_th=10):
        self.n = n
        self.p = p
        self.m = m
        self.influencer_th = influencer_th
        self.G = None
        if network_type == 0:
            self.G = nx.erdos_renyi_graph(n,p)
        elif network_type == 1:
            G_without_influencer = True
            tries = 0
            while G_without_influencer and tries < 50:
                self.G = nx.barabasi_albert_graph(n,m)
                influencers = self._check_for_influencers()
                if len(influencers) != 0:
                    G_without_influencer = False
                tries += 1
            if G_without_influencer:
                print("Ease the influencer threshold or initialize more nodes\n")

    def _check_for_influencers(self):
        top_fraction = 0.05
        number_of_influencers = max(1, int(top_fraction * self.G.number_of_nodes()))

        sorted_degree_nodes = sorted(self.G.nodes(), key=lambda node: self.G.degree[node], reverse=True)
        top_degree_nodes = sorted_degree_nodes[:number_of_influencers]
        influencers = [node for node in top_degree_nodes if self.G.degree[node] >= self.influencer_th]
        return influencers