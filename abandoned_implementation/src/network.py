import mesa
import networkx as nx
import math
from mesa.space import NetworkGrid
from definitions import NetworkType, AgentType, MessageState
import agents

class SocialNetwork(mesa.Model):
    def __init__(self, network_type, n, p, m, influencer_th=40, truthfulness=0.5,
                 verify_cost=0.25, rationality=2.5, homophily_weight=2.0, seed=None,
                 trust_learning_rate=0.2, trust_weight=0.6,
                 fake_learning_rate=0.05, risk_aversion=0.25):
        super().__init__(seed=seed)

        self.n = n
        self.p = p
        self.m = m
        self.influencer_th = influencer_th
        self.truthfulness = truthfulness   # initial global P(message is true)
        self.verify_cost = verify_cost     # cost c to verify a message
        self.rationality = rationality     # sharpness of the verify best-response
        self.max_candidates_considered = 2  # limited attention in rewiring
        self.homophily_weight = homophily_weight  # strength of belief-similarity preference
        self.total_verification_cost_paid = 0.0   # model-level aggregate of verification effort
        self.network_type = network_type

        # Added mechanism:
        # receivers trust specific senders, and senders adapt fake-news tendency.
        self.trust_learning_rate = trust_learning_rate
        self.trust_weight = trust_weight
        self.fake_learning_rate = fake_learning_rate
        self.risk_aversion = risk_aversion

        # These are updated every cascade and used for output metrics.
        self.last_message = None
        self.last_initiator_id = None

        self.G = None
        self.grid = None
        self.influencer_nodes = []

        self.social_agents = []
        self.influencers = []
        self.normal_users = []
        if not self._generate_network(network_type):
            raise ValueError("Ease the influencer threshold or initialize more nodes\n")

        # Store initial network structure before any rewiring.
        # This allows checking whether initial high-degree nodes become influencers later.
        self.initial_degrees = dict(self.G.degree())
        self.initial_degree_ranking = sorted(
            self.initial_degrees,
            key=self.initial_degrees.get,
            reverse=True
        )

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

        influencer_nodes = [
            node for node in top_degree_nodes
            if self.G.degree[node] >= self.influencer_th
        ]

        return influencer_nodes

    def update_influencers(self):
        old_influencer_nodes = set(self.influencer_nodes)
        updated_influencer_nodes = set(self._check_for_influencers())

        removed_influencer_nodes = old_influencer_nodes - updated_influencer_nodes
        new_influencer_nodes = updated_influencer_nodes - old_influencer_nodes

        # Remove influencer status
        for node_id in removed_influencer_nodes:
            self.social_agents[node_id].is_influencer = False

        # Add influencer status
        for node_id in new_influencer_nodes:
            self.social_agents[node_id].is_influencer = True

        # Store all current influencer node IDs
        self.influencer_nodes = list(updated_influencer_nodes)

        # Store all current influencer agent objects
        self.influencers = [
            self.social_agents[node_id]
            for node_id in self.influencer_nodes
        ]
        self.normal_users = list(set(self.social_agents).difference(self.influencers))



    def _generate_network(self, network_type):
        if network_type == NetworkType.Random:
            self.G = nx.erdos_renyi_graph(self.n, self.p)
            return 1

        elif network_type == NetworkType.ScaleFree:
            tries = 0

            while tries < 50:
                self.G = nx.barabasi_albert_graph(self.n, self.m)
                influencers = self._check_for_influencers()

                if len(influencers) != 0:
                    self.influencer_nodes = influencers
                    return 1

                tries += 1
            return 0
        elif network_type == NetworkType.Regular:
            degree = self.m

            if degree >= self.n:
                raise ValueError("Regular graph degree must be smaller than n")

            if (degree * self.n) % 2 != 0:
                raise ValueError("For a regular graph, degree * n must be even")

            self.G = nx.random_regular_graph(
                d=degree,
                n=self.n
            )
            return 1

        else:
            raise ValueError("Unknown network type")

    def _init_agents(self):
        for node in self.G.nodes():
            agent = None

            if node in self.influencer_nodes:
                agent = agents.SocialAgent(
                    self,
                    node_id=node,
                    r=0,
                    is_influencer=True
                )
                self.influencers.append(agent)

            else:
                agent = agents.SocialAgent(
                    self,
                    node_id=node,
                    r=0,
                    is_influencer=False
                )

            self.grid.place_agent(agent, node)
            self.social_agents.append(agent)
        self.normal_users = list(set(self.social_agents).difference(self.influencers))


    def simulation_step(self, max_steps=20):
        self._reset_message_states()

        # choose who starts the message
        initiator_agent = self._choose_initiator(None)
        # initiator creates the message
        message = initiator_agent.initiate_message()

        # Store cascade-level information for output metrics.
        self.last_message = message
        self.last_initiator_id = initiator_agent.node_id

        active_agents = {initiator_agent}

        # active_sharers contains agent objects
        active_sharers = [initiator_agent]

        step = 0

        while active_sharers and step < max_steps:
            exposed_agents = []
            seen_agents = set()

            for sender_agent in active_sharers:
                for neighbor_id in self.G.neighbors(sender_agent.node_id):
                    neighbor_agent = self.social_agents[neighbor_id]

                    if (
                        neighbor_agent.message_state == MessageState.Unaware
                        and neighbor_agent not in seen_agents
                    ):
                        exposed_agents.append((sender_agent, neighbor_agent))
                        seen_agents.add(neighbor_agent)

            new_sharers = []
            active_agents.update(seen_agents)


            # let exposed agents decide whether they share further
            for (sender_agent, receiver_agent) in exposed_agents:

                (agent_response, sender_r_change) = receiver_agent.receive_message(
                    message,
                    sender_agent.node_id
                )
                sender_agent.r += sender_r_change

                # Sender notices whether fake news paid off or was punished.
                sender_agent.adapt_fake_tendency(message, sender_r_change)

                # assuming 1 means "shares further"
                if agent_response == 1:
                    new_sharers.append(receiver_agent)

            active_sharers = new_sharers
            step += 1

        return active_agents

    #TODO: fix for no initiators
    def _choose_initiator(self, seed_type):
        if seed_type == AgentType.Influencer and len(self.influencers) > 0:
            return self.random.choice(self.influencers)
        elif seed_type == AgentType.NormalUser:
            return self.random.choice(self.normal_users)
        else:
            return self.random.choice(self.social_agents)

    def _reset_message_states(self):
        for agent in self.social_agents:
            agent.message_state = MessageState.Unaware

    def update_agents(self):
        for agent in self.social_agents:
            agent.update()

    def _connection_score(self, observer_agent, target_agent):
        """
        Score used for rewiring.

        Agents prefer targets with high observed reputation and similar beliefs
        about the truthfulness of the information environment.
        """
        observed_reputation = observer_agent.observe_reputation(target_agent)
        belief_distance = abs(
            observer_agent.perceived_truthfulness
            - target_agent.perceived_truthfulness
        )

        return observed_reputation - self.homophily_weight * belief_distance

    def rewire_network(self, rewire_prob):
        rewired_edges = 0

        for agent in self.normal_users:
            if self.random.random() >= rewire_prob:
                continue

            agent_id = agent.node_id
            current_neighbor_ids = list(self.G.neighbors(agent_id))

            if len(current_neighbor_ids) == 0:
                continue

            current_neighbor_set = set(current_neighbor_ids)

            worst_neighbor_id = None
            worst_neighbor_score = math.inf

            possible_candidate_ids = set()

            for neighbor_id in current_neighbor_ids:
                neighbor_agent = self.social_agents[neighbor_id]

                # Find the least attractive current neighbor.
                # This now uses both reputation and belief similarity.
                neighbor_score = self._connection_score(agent, neighbor_agent)
                if neighbor_score < worst_neighbor_score:
                    worst_neighbor_id = neighbor_id
                    worst_neighbor_score = neighbor_score

                # Collect friends-of-friends as possible new neighbors.
                for candidate_id in self.G.neighbors(neighbor_id):
                    if candidate_id == agent_id:
                        continue

                    if candidate_id in current_neighbor_set:
                        continue

                    possible_candidate_ids.add(candidate_id)

            if len(possible_candidate_ids) == 0:
                continue

            # Limited attention: the agent does not evaluate all possible candidates.
            possible_candidate_ids = list(possible_candidate_ids)
            sample_size = min(
                self.max_candidates_considered,
                len(possible_candidate_ids)
            )
            noticed_candidate_ids = self.random.sample(
                possible_candidate_ids,
                sample_size
            )

            best_candidate_id = None
            best_candidate_score = -math.inf

            for candidate_id in noticed_candidate_ids:
                candidate_agent = self.social_agents[candidate_id]
                candidate_score = self._connection_score(agent, candidate_agent)

                if candidate_score > best_candidate_score:
                    best_candidate_id = candidate_id
                    best_candidate_score = candidate_score

            # No useful candidate found.
            if best_candidate_id is None:
                continue

            # Only rewire if the noticed candidate is actually better.
            if best_candidate_score <= worst_neighbor_score:
                continue

            # Replace worst neighbor with best noticed candidate.
            self.G.remove_edge(agent_id, worst_neighbor_id)
            self.G.add_edge(agent_id, best_candidate_id)

            rewired_edges += 1

        return rewired_edges
