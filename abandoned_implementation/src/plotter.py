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

        # Conditional cascade-size series.
        # These show whether fake-message cascades are larger than true-message cascades.
        if "message_is_fake" in df.columns and "cascade_size" in df.columns:
            df["fake_message_cascade_size"] = df["cascade_size"].where(
                df["message_is_fake"] == True
            )
            df["true_message_cascade_size"] = df["cascade_size"].where(
                df["message_is_true"] == True
            )

            df["fake_message_cascade_size_rolling"] = (
                df["fake_message_cascade_size"]
                .rolling(window=rolling_window, min_periods=1)
                .mean()
            )
            df["true_message_cascade_size_rolling"] = (
                df["true_message_cascade_size"]
                .rolling(window=rolling_window, min_periods=1)
                .mean()
            )

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

        key_metrics = [
            "fake_share_of_spread",
            "fake_to_true_ratio",
            "correction_rate",
            "discard_rate",
            "verifications_this_round",
            "mean_fake_tendency",
            "mean_trust",
            "max_degree_ratio",
            "max_degree",
            "degree_std"
        ]

        for metric in key_metrics:
            if metric in df.columns:
                summary[f"mean_{metric}"] = df[metric].mean()
                summary[f"final_{metric}"] = df[metric].iloc[-1]

        if (
            "fake_message_cascade_size" in df.columns
            and "true_message_cascade_size" in df.columns
        ):
            mean_fake_cascade = df["fake_message_cascade_size"].mean()
            mean_true_cascade = df["true_message_cascade_size"].mean()

            summary["mean_fake_message_cascade_size"] = mean_fake_cascade
            summary["mean_true_message_cascade_size"] = mean_true_cascade
            summary["fake_to_true_cascade_ratio"] = (
                mean_fake_cascade / mean_true_cascade
                if mean_true_cascade > 0 else float("nan")
            )

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

            # Combined overview figure.
            # This keeps the separate plots below, but also puts the most
            # important time series in one shared x-axis figure.
            overview_panels = []

            misinformation_cols = [
                ("fake_believers", "Fake believers"),
                ("corrected", "Corrected"),
                ("discarded", "Discarded")
            ]

            available_misinformation_cols = [
                (col, label)
                for col, label in misinformation_cols
                if f"{col}_rolling" in df.columns
            ]

            if len(available_misinformation_cols) > 0:
                overview_panels.append("misinformation_states")

            if "mean_reputation_rolling" in df.columns:
                overview_panels.append("mean_reputation")

            if "n_influencers_rolling" in df.columns:
                overview_panels.append("n_influencers")

            if len(overview_panels) > 0:
                fig, axes = plt.subplots(
                    nrows=len(overview_panels),
                    ncols=1,
                    figsize=(10, 3.2 * len(overview_panels)),
                    sharex=True
                )

                if len(overview_panels) == 1:
                    axes = [axes]

                for ax, panel in zip(axes, overview_panels):
                    if panel == "misinformation_states":
                        for col, label in available_misinformation_cols:
                            ax.plot(
                                x,
                                df[f"{col}_rolling"],
                                label=label
                            )

                        ax.set_ylabel("Number of agents")
                        ax.set_title(
                            "Fake believers, corrected agents, and discarded messages"
                        )
                        ax.legend()

                    elif panel == "mean_reputation":
                        ax.plot(
                            x,
                            df["mean_reputation"],
                            alpha=0.35,
                            label="Raw"
                        )
                        ax.plot(
                            x,
                            df["mean_reputation_rolling"],
                            label=f"Rolling mean ({rolling_window})"
                        )
                        ax.set_ylabel("Mean reputation")
                        ax.set_title("Mean reputation over time")
                        ax.legend()

                    elif panel == "n_influencers":
                        ax.plot(
                            x,
                            df["n_influencers"],
                            alpha=0.35,
                            label="Raw"
                        )
                        ax.plot(
                            x,
                            df["n_influencers_rolling"],
                            label=f"Rolling mean ({rolling_window})"
                        )
                        ax.set_ylabel("Number of influencers")
                        ax.set_title("Influencer count over time")
                        ax.legend()

                axes[-1].set_xlabel("Round")
                fig.suptitle("Simulation overview over time", y=1.02)
                fig.tight_layout()
                plt.show()

            # 1. Cascade size
            if "cascade_size" in df.columns:
                plt.figure(figsize=(8, 4))
                plt.plot(x, df["cascade_size_rolling"], label=f"Rolling mean ({rolling_window})")
                plt.xlabel("Round")
                plt.ylabel("Cascade size")
                plt.title("Cascade size over time")
                plt.legend()
                plt.tight_layout()
                plt.show()


            # 2d. Cascade size for fake versus true messages
            if (
                "fake_message_cascade_size_rolling" in df.columns
                and "true_message_cascade_size_rolling" in df.columns
            ):
                plt.figure(figsize=(8, 4))
                plt.plot(
                    x,
                    df["fake_message_cascade_size_rolling"],
                    label="Fake-message cascades"
                )
                plt.plot(
                    x,
                    df["true_message_cascade_size_rolling"],
                    label="True-message cascades"
                )
                plt.xlabel("Round")
                plt.ylabel("Cascade size")
                plt.title("Cascade size by message type")
                plt.legend()
                plt.tight_layout()
                plt.show()

            # 2e. Trust and fake-news adaptation
            behavioral_columns = [
                ("mean_fake_tendency", "Mean fake tendency"),
                ("mean_trust", "Mean sender-specific trust")
            ]

            available_behavioral_columns = [
                (col, label)
                for col, label in behavioral_columns
                if f"{col}_rolling" in df.columns
            ]

            if len(available_behavioral_columns) > 0:
                plt.figure(figsize=(8, 4))

                for col, label in available_behavioral_columns:
                    plt.plot(
                        x,
                        df[f"{col}_rolling"],
                        label=label
                    )

                plt.xlabel("Round")
                plt.ylabel("Mean value")
                plt.title("Trust formation and fake-news adaptation")
                plt.ylim(0, 1)
                plt.legend()
                plt.tight_layout()
                plt.show()

            # 2f. Influencer concentration
            if "max_degree_ratio_rolling" in df.columns:
                plt.figure(figsize=(8, 4))
                plt.plot(
                    x,
                    df["max_degree_ratio"],
                    alpha=0.25,
                    label="Raw"
                )
                plt.plot(
                    x,
                    df["max_degree_ratio_rolling"],
                    label=f"Rolling mean ({rolling_window})"
                )
                plt.xlabel("Round")
                plt.ylabel("Max degree / possible max degree")
                plt.title("Influencer concentration over time")
                plt.ylim(0, 1)
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


        return df, summary_df


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


    def plot_initial_degree_vs_final_degree(self):
        """
        Plot initial degree against final degree.

        Final influencers are highlighted in orange.
        """

        if not hasattr(self.social_network, "initial_degrees"):
            print(
                "No initial degree information found. "
                "Make sure SocialNetwork stores self.initial_degrees after network creation."
            )
            return

        initial_degrees = self.social_network.initial_degrees
        final_degrees = dict(self.G.degree())
        final_influencer_nodes = set(self.social_network.influencer_nodes)

        x = []
        y = []
        colors = []

        for node in self.G.nodes():
            x.append(initial_degrees[node])
            y.append(final_degrees[node])

            if node in final_influencer_nodes:
                colors.append("orange")
            else:
                colors.append("lightblue")

        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, c=colors, alpha=0.8)

        plt.xlabel("Initial degree")
        plt.ylabel("Final degree")
        plt.title("Initial degree vs final degree")

        plt.tight_layout()
        plt.show()

