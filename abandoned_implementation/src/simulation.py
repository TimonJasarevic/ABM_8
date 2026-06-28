import pandas as pd

from plotter import NetworkPlotter
from network import SocialNetwork
from definitions import NetworkType


class Simulation:
    def __init__(
        self,
        social_network,
        rounds,
        rewire_prob=0.005,
        max_steps=20,
        draw_every=10,
        burn_in=1000,
        plot_network=True
    ):
        self.burn_in = burn_in
        self.social_network = social_network
        self.rounds = rounds + burn_in
        self.rewire_prob = rewire_prob
        self.max_steps = max_steps
        self.draw_every = draw_every
        self.plot_network = plot_network
        self.plotter = NetworkPlotter(social_network)

    def _mean_trust(self):
        all_trust_values = []

        for agent in self.social_network.social_agents:
            all_trust_values.extend(agent.trust.values())

        if len(all_trust_values) == 0:
            return 0.5

        return sum(all_trust_values) / len(all_trust_values)

    def _degree_metrics(self):
        degrees = [degree for _, degree in self.social_network.G.degree()]
        n = len(degrees)

        if n == 0:
            return {
                'max_degree': 0,
                'max_degree_ratio': 0,
                'degree_std': 0,
                'mean_degree': 0
            }

        degree_series = pd.Series(degrees)

        return {
            'max_degree': int(degree_series.max()),
            'max_degree_ratio': (
                float(degree_series.max()) / (n - 1)
                if n > 1 else 0
            ),
            'degree_std': float(degree_series.std()),
            'mean_degree': float(degree_series.mean())
        }

    def _collect_metrics(
        self,
        round_number,
        active_agents,
        rewired_edges,
        verification_cost_before
    ):
        agents = self.social_network.social_agents
        n_agents = len(agents)
        cascade_size = len(active_agents)

        message_is_true = bool(self.social_network.last_message)
        message_is_fake = not message_is_true

        true_believers = sum(
            a.message_state.name == 'TrueBeliever'
            for a in agents
        )

        fake_believers = sum(
            a.message_state.name == 'FalseBeliever'
            for a in agents
        )

        corrected = sum(
            a.message_state.name == 'Corrected'
            for a in agents
        )

        discarded = sum(
            a.message_state.name == 'Discarded'
            for a in agents
        )

        unaware = sum(
            a.message_state.name == 'Unaware'
            for a in agents
        )

        mean_reputation = sum(a.r for a in agents) / n_agents

        spread_believers = true_believers + fake_believers

        fake_share_of_spread = (
            fake_believers / spread_believers
            if spread_believers > 0 else 0.0
        )

        correction_rate = (
            corrected / (corrected + fake_believers)
            if corrected + fake_believers > 0 else 0.0
        )

        discard_rate = (
            discarded / cascade_size
            if cascade_size > 0 else 0.0
        )

        verification_cost_after = self.social_network.total_verification_cost_paid
        verification_cost_this_round = (
            verification_cost_after - verification_cost_before
        )

        verifications_this_round = (
            verification_cost_this_round / self.social_network.verify_cost
            if self.social_network.verify_cost > 0 else 0.0
        )

        mean_fake_tendency = sum(
            a.fake_tendency for a in agents
        ) / n_agents

        mean_trust = self._mean_trust()

        degree_metrics = self._degree_metrics()

        return {
            'round': round_number,
            'message_is_true': message_is_true,
            'message_is_fake': message_is_fake,

            'cascade_size': cascade_size,
            'true_believers': true_believers,
            'fake_believers': fake_believers,
            'corrected': corrected,
            'discarded': discarded,
            'unaware': unaware,

            'fake_share_of_spread': fake_share_of_spread,
            'correction_rate': correction_rate,
            'discard_rate': discard_rate,

            'verification_cost_this_round': verification_cost_this_round,
            'verifications_this_round': verifications_this_round,
            'mean_fake_tendency': mean_fake_tendency,
            'mean_trust': mean_trust,

            'mean_reputation': mean_reputation,
            'rewired_edges': rewired_edges,
            'n_influencers': len(self.social_network.influencers),
            **degree_metrics
        }

    def run(
        self,
        rolling_window=50,
        plot_results=True,
        save_csv=None,
        show_degree_plots=True
    ):
        history = []

        for round_number in range(self.rounds):
            verification_cost_before = (
                self.social_network.total_verification_cost_paid
            )

            active_agents = self.social_network.simulation_step(
                max_steps=self.max_steps
            )

            self.social_network.update_agents()

            rewired_edges = self.social_network.rewire_network(
                rewire_prob=self.rewire_prob
            )
            self.social_network.update_influencers()


            if round_number > self.burn_in and self.draw_every is not None and round_number % self.draw_every == 0:

                if self.plot_network:
                    self.plotter.draw_network_graph(
                        title=(
                            f'Round {round_number + 1}, '
                            f'rewired edges: {rewired_edges}'
                        ),
                        label_type='reputation'
                    )
            if round_number > self.burn_in:
                history.append(
                    self._collect_metrics(
                        round_number=round_number,
                        active_agents=active_agents,
                        rewired_edges=rewired_edges,
                        verification_cost_before=verification_cost_before
                    )
                )

        if self.plot_network:
            self.plotter.show_network_graph()

        analysis = self.plotter.analyze_history(
            history,
            rolling_window=rolling_window,
            plot=plot_results,
            save_csv=save_csv
        )

        if show_degree_plots and plot_results:
            self.plotter.plot_degree_ccdf_vs_barabasi()
            self.plotter.plot_initial_degree_vs_final_degree()

        return analysis


if __name__ == '__main__':
    social_network = SocialNetwork(
        network_type=NetworkType.Regular,
        n=400,
        p=4 / 400,
        m=4,
        influencer_th=60,
        truthfulness=0.5,
        seed=42
    )

    sim = Simulation(
        social_network=social_network,
        rounds=4000,
        burn_in=1200,
        rewire_prob=0.005,
        draw_every=25,
        plot_network=True
    )

    sim.run(
        rolling_window=50,
        plot_results=True,
        save_csv=None,
        show_degree_plots=True
    )
