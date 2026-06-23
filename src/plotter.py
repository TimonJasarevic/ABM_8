from network import SocialNetwork
import networkx as nx
import matplotlib.pyplot as plt


class NetworkPlotter():
    def __init__(self, social_network):
        self.social_network = social_network
        self.G = social_network.G
        self.network_fig, self.network_ax = plt.subplots(figsize=(10, 8))

        self.pos = nx.spring_layout(self.G, seed=42)


    def draw_network_graph(self, title="Social network", label_type="reputation"):
        """
        Draw the current graph.

        Node colors stay fixed:
        - orange: influencer
        - lightblue: normal user

        label_type:
        - "reputation": show agent reputation
        - "node_id": show node id
        - None: show no labels
        """

        node_colors = []
        node_sizes = []

        for node in self.G.nodes():
            if node in self.social_network.influencer_nodes:
                node_colors.append("orange")
                node_sizes.append(600)
            else:
                node_colors.append("lightblue")
                node_sizes.append(350)

        self.network_ax.clear()

        nx.draw_networkx_edges(
            self.G,
            self.pos,
            ax=self.network_ax,
            edge_color="gray",
            alpha=0.4
        )

        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            ax=self.network_ax,
            node_color=node_colors,
            node_size=node_sizes
        )

        if label_type == "reputation":
            labels = {
                node: f"{self.social_network.social_agents[node].r:.0f}"
                for node in self.G.nodes()
            }

            nx.draw_networkx_labels(
                self.G,
                self.pos,
                labels=labels,
                ax=self.network_ax,
                font_size=8
            )

        elif label_type == "node_id":
            labels = {
                node: str(node)
                for node in self.G.nodes()
            }

            nx.draw_networkx_labels(
                self.G,
                self.pos,
                labels=labels,
                ax=self.network_ax,
                font_size=8
            )

        self.network_ax.set_title(title)
        self.network_ax.axis("off")

        plt.pause(0.01)

    def show_network_graph(self):
        plt.show()