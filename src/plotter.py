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

        if "discarded" in df.columns:
            summary["mean_discarded"] = df["discarded"].mean()
            summary["max_discarded"] = df["discarded"].max()
            summary["final_discarded"] = df["discarded"].iloc[-1]

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

            # 2. Fake believers, corrected agents, and discarded messages
            if "fake_believers" in df.columns or "corrected" in df.columns or "discarded" in df.columns:
                plt.figure(figsize=(8, 4))

                if "fake_believers" in df.columns:
                    plt.plot(x, df["fake_believers_rolling"], label="Fake believers")

                if "corrected" in df.columns:
                    plt.plot(x, df["corrected_rolling"], label="Corrected")

                if "discarded" in df.columns:
                    plt.plot(x, df["discarded_rolling"], label="Discarded")

                plt.xlabel("Round")
                plt.ylabel("Number of agents")
                plt.title("Fake believers, corrected agents, and discarded messages")
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

    def plot_degree_distribution_vs_barabasi(
        self,
        ba_m=None,
        seed=42,
        normalize=True,
        log_y=False
    ):
        """
        Compare the degree distribution of the current graph with a
        Barabasi-Albert graph.

        The Barabasi-Albert graph is generated with:
        - same number of nodes as the current graph
        - approximately similar average degree

        Parameters
        ----------
        ba_m : int or None
            Number of edges each new node attaches with in the BA graph.
            If None, it is chosen so that BA average degree is close to the
            current graph's average degree.

        seed : int
            Random seed for the BA graph.

        normalize : bool
            If True, plot fractions of nodes.
            If False, plot raw node counts.

        log_y : bool
            If True, use logarithmic y-axis.
        """

        from collections import Counter

        current_degrees = [degree for _, degree in self.G.degree()]
        n = self.G.number_of_nodes()

        if n == 0:
            print("Graph has no nodes.")
            return

        current_avg_degree = sum(current_degrees) / n

        # In a Barabasi-Albert graph, average degree is approximately 2m.
        # So choose m close to current_avg_degree / 2.
        if ba_m is None:
            ba_m = round(current_avg_degree / 2)
            ba_m = max(1, min(ba_m, n - 1))

        ba_graph = nx.barabasi_albert_graph(
            n=n,
            m=ba_m,
            seed=seed
        )

        ba_degrees = [degree for _, degree in ba_graph.degree()]

        current_counts = Counter(current_degrees)
        ba_counts = Counter(ba_degrees)

        max_degree = max(
            max(current_degrees),
            max(ba_degrees)
        )

        degrees = list(range(max_degree + 1))

        current_values = [current_counts.get(degree, 0) for degree in degrees]
        ba_values = [ba_counts.get(degree, 0) for degree in degrees]

        if normalize:
            current_values = [value / n for value in current_values]
            ba_values = [value / n for value in ba_values]
            y_label = "Fraction of nodes"
        else:
            y_label = "Number of nodes"

        bar_width = 0.4

        current_x = [degree - bar_width / 2 for degree in degrees]
        ba_x = [degree + bar_width / 2 for degree in degrees]

        plt.figure(figsize=(10, 5))

        plt.bar(
            current_x,
            current_values,
            width=bar_width,
            alpha=0.7,
            label=(
                f"Current graph "
                f"(n={n}, avg degree={current_avg_degree:.2f})"
            )
        )

        plt.bar(
            ba_x,
            ba_values,
            width=bar_width,
            alpha=0.7,
            label=(
                f"Barabasi-Albert "
                f"(n={n}, m={ba_m}, avg degree={sum(ba_degrees) / n:.2f})"
            )
        )

        plt.xlabel("Degree")
        plt.ylabel(y_label)
        plt.title("Degree distribution: current graph vs Barabasi-Albert graph")
        plt.legend()
        plt.tight_layout()

        if log_y:
            plt.yscale("log")

        plt.show()

    def plot_degree_ccdf_vs_barabasi(self, ba_m=None, seed=42):
        """
        Plot the complementary cumulative degree distribution.

        This is better than a normal histogram when the graph has hubs.
        It shows P(degree >= k), making tail behavior easier to compare.
        """

        current_degrees = [degree for _, degree in self.G.degree()]
        n = self.G.number_of_nodes()

        if n == 0:
            print("Graph has no nodes.")
            return

        current_avg_degree = sum(current_degrees) / n

        if ba_m is None:
            ba_m = round(current_avg_degree / 2)
            ba_m = max(1, min(ba_m, n - 1))

        ba_graph = nx.barabasi_albert_graph(
            n=n,
            m=ba_m,
            seed=seed
        )

        ba_degrees = [degree for _, degree in ba_graph.degree()]

        def ccdf_values(degrees):
            max_degree = max(degrees)
            x_values = []
            y_values = []

            for k in range(1, max_degree + 1):
                fraction = sum(degree >= k for degree in degrees) / len(degrees)

                if fraction > 0:
                    x_values.append(k)
                    y_values.append(fraction)

            return x_values, y_values

        current_x, current_y = ccdf_values(current_degrees)
        ba_x, ba_y = ccdf_values(ba_degrees)

        plt.figure(figsize=(8, 5))

        plt.plot(
            current_x,
            current_y,
            marker="o",
            linestyle="none",
            markersize=4,
            label=(
                f"Current graph "
                f"(n={n}, avg degree={current_avg_degree:.2f}, "
                f"max degree={max(current_degrees)})"
            )
        )

        plt.plot(
            ba_x,
            ba_y,
            marker="o",
            linestyle="none",
            markersize=4,
            label=(
                f"Barabasi-Albert "
                f"(n={n}, m={ba_m}, avg degree={sum(ba_degrees) / n:.2f}, "
                f"max degree={max(ba_degrees)})"
            )
        )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Degree k")
        plt.ylabel("P(degree >= k)")
        plt.title("Degree CCDF: current graph vs Barabasi-Albert graph")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def print_network_degree_summary(self):
        """
        Print basic degree statistics for the current graph.
        """

        degrees = [degree for _, degree in self.G.degree()]
        n = self.G.number_of_nodes()

        if n == 0:
            print("Graph has no nodes.")
            return

        degree_series = pd.Series(degrees)

        print("\n=== Degree summary ===")
        print(f"Number of nodes: {n}")
        print(f"Number of edges: {self.G.number_of_edges()}")
        print(f"Average degree: {degree_series.mean():.2f}")
        print(f"Median degree: {degree_series.median():.2f}")
        print(f"Max degree: {degree_series.max()}")
        print(f"Min degree: {degree_series.min()}")
        print(f"Isolated nodes: {(degree_series == 0).sum()}")
        print(f"Degree std: {degree_series.std():.2f}")

