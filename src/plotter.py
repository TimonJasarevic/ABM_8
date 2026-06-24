from network import SocialNetwork
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class NetworkPlotter():
    def __init__(self, social_network):
        self.social_network = social_network
        self.G = social_network.G
        self.network_fig, self.network_ax = plt.subplots(figsize=(10, 8))

        self.pos = nx.spring_layout(
            self.G,
            seed=42,
            k=1.5,
            iterations=150,
            scale=4
        )

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
                node_sizes.append(350)
            else:
                node_colors.append("lightblue")
                node_sizes.append(180)

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

        plt.pause(0.1)

    def show_network_graph(self):
        plt.show()

    def analyze_history(self, history, rolling_window=50, plot=True, save_csv=None):


        if len(history) == 0:
            print("No history found. Make sure you append metrics to history during run().")
            return None

        df = pd.DataFrame(history)

        if "round" in df.columns:
            df = df.sort_values("round")

        # Add rolling averages for smoother plots
        numeric_cols = df.select_dtypes(include="number").columns

        for col in numeric_cols:
            if col != "round":
                df[f"{col}_rolling"] = df[col].rolling(
                    window=rolling_window,
                    min_periods=1
                ).mean()

        summary = {}

        if "cascade_size" in df.columns:
            summary["mean_cascade_size"] = df["cascade_size"].mean()
            summary["max_cascade_size"] = df["cascade_size"].max()
            summary["final_cascade_size"] = df["cascade_size"].iloc[-1]

        if "fake_believers" in df.columns:
            summary["mean_fake_believers"] = df["fake_believers"].mean()
            summary["max_fake_believers"] = df["fake_believers"].max()
            summary["final_fake_believers"] = df["fake_believers"].iloc[-1]

        if "corrected" in df.columns:
            summary["mean_corrected"] = df["corrected"].mean()
            summary["max_corrected"] = df["corrected"].max()
            summary["final_corrected"] = df["corrected"].iloc[-1]

        if "mean_reputation" in df.columns:
            summary["mean_reputation_over_time"] = df["mean_reputation"].mean()
            summary["final_mean_reputation"] = df["mean_reputation"].iloc[-1]

        if "rewired_edges" in df.columns:
            summary["total_rewired_edges"] = df["rewired_edges"].sum()
            summary["mean_rewired_edges_per_round"] = df["rewired_edges"].mean()

        if "n_influencers" in df.columns:
            summary["mean_n_influencers"] = df["n_influencers"].mean()
            summary["final_n_influencers"] = df["n_influencers"].iloc[-1]

        summary_df = pd.DataFrame.from_dict(
            summary,
            orient="index",
            columns=["value"]
        )

        print("\n=== Simulation summary ===")
        print(summary_df)

        if save_csv is not None:
            df.to_csv(save_csv, index=False)
            print(f"\nSaved history to: {save_csv}")

        if plot:
            x = df["round"] if "round" in df.columns else df.index

            # 1. Cascade size
            if "cascade_size" in df.columns:
                plt.figure(figsize=(8, 4))
                plt.plot(x, df["cascade_size"], alpha=0.35, label="Raw")
                plt.plot(x, df["cascade_size_rolling"], label=f"Rolling mean ({rolling_window})")
                plt.xlabel("Round")
                plt.ylabel("Cascade size")
                plt.title("Cascade size over time")
                plt.legend()
                plt.tight_layout()
                plt.show()

            # 2. Fake believers and corrected agents
            if "fake_believers" in df.columns or "corrected" in df.columns:
                plt.figure(figsize=(8, 4))

                if "fake_believers" in df.columns:
                    plt.plot(x, df["fake_believers_rolling"], label="Fake believers")

                if "corrected" in df.columns:
                    plt.plot(x, df["corrected_rolling"], label="Corrected")

                plt.xlabel("Round")
                plt.ylabel("Number of agents")
                plt.title("Fake believers and corrected agents")
                plt.legend()
                plt.tight_layout()
                plt.show()

            # 3. Mean reputation
            if "mean_reputation" in df.columns:
                plt.figure(figsize=(8, 4))
                plt.plot(x, df["mean_reputation"], alpha=0.35, label="Raw")
                plt.plot(x, df["mean_reputation_rolling"], label=f"Rolling mean ({rolling_window})")
                plt.xlabel("Round")
                plt.ylabel("Mean reputation")
                plt.title("Mean reputation over time")
                plt.legend()
                plt.tight_layout()
                plt.show()

            # 4. Rewired edges
            if "rewired_edges" in df.columns:
                plt.figure(figsize=(8, 4))
                plt.plot(x, df["rewired_edges"], alpha=0.35, label="Raw")
                plt.plot(x, df["rewired_edges_rolling"], label=f"Rolling mean ({rolling_window})")
                plt.xlabel("Round")
                plt.ylabel("Rewired edges")
                plt.title("Network rewiring over time")
                plt.legend()
                plt.tight_layout()
                plt.show()

            # 5. Number of influencers
            if "n_influencers" in df.columns:
                plt.figure(figsize=(8, 4))
                plt.plot(x, df["n_influencers"], alpha=0.35, label="Raw")
                plt.plot(x, df["n_influencers_rolling"], label=f"Rolling mean ({rolling_window})")
                plt.xlabel("Round")
                plt.ylabel("Number of influencers")
                plt.title("Influencer count over time")
                plt.legend()
                plt.tight_layout()
                plt.show()

        return df, summary_df