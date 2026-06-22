import mesa
import networkx as nx
import math
from mesa.space import NetworkGrid
from definitions import NetworkType, AgentType, MessageState
import agents
import matplotlib.pyplot as plt


class SocialNetwork(mesa.Model):
    def __init__(self, network_type, n, p, m, influencer_th=10, truthfulness=0.5,
                 verify_cost=0.2, fake_penalty=1.0, rationality=5.0, seed=None):
        super().__init__(seed=seed)

        self.n = n
        self.p = p
        self.m = m
        self.influencer_th = influencer_th
        self.truthfulness = truthfulness   # global P(message is true); fraction of true vs fake messages
        self.verify_cost = verify_cost     # cost c to verify a message
        self.fake_penalty = fake_penalty   # reputation (== wealth) loss for spreading fake news
        self.rationality = rationality     # sharpness of the verify best-response
        self.total_verification_cost_paid = 0.0   # model-level aggregate of verification effort
        self.network_type = network_type

        self.G = None
        self.grid = None
        self.influencer_nodes = []

        self.social_agents = []
        self.influencers = []
        self.normal_users = []
        if not self._generate_network(network_type):
            raise ValueError("Ease the influencer threshold or initialize more nodes\n")
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
        else:
            raise ValueError("Unknown network type")

    def _init_agents(self):
        for node in self.G.nodes():
            agent = None

            if node in self.influencer_nodes:
                agent = agents.Influencer(
                    self,
                    node,
                    0,
                )
                self.influencers.append(agent)

            else:
                agent = agents.NormalUser(
                    self,
                    node,
                    0,
                )

            self.grid.place_agent(agent, node)
            self.social_agents.append(agent)
        self.normal_users = list(set(self.social_agents).difference(self.influencers))


    def simulation_step(self, max_steps=20):
        self._reset_message_states()

        # choose who starts the message
        initiator_agent = self._choose_initiator(AgentType.Influencer)

        # initiator creates the message
        message = initiator_agent.initiate_message()

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

                agent_response = receiver_agent.receive_message(
                    message,
                    sender_agent.node_id,
                    sender_agent.r
                )

                # assuming 1 means "shares further"
                if agent_response == 1:
                    new_sharers.append(receiver_agent)

            active_sharers = new_sharers
            step += 1

        return active_agents

    #TODO: fix for no initiators
    def _choose_initiator(self, seed_type):
        if seed_type == AgentType.Influencer and self.network_type == NetworkType.ScaleFree:
            return self.random.choice(self.influencers)
        elif seed_type == AgentType.NormalUser:
            return self.random.choice(self.normal_users)
        else:
            raise ValueError("Unknown seed type")

    def _reset_message_states(self):
        for agent in self.social_agents:
            agent.message_state = MessageState.Unaware

    def update_agents(self, agent_set):
        for agent in agent_set:
            agent.update()

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
            worst_neighbor_r = math.inf

            best_candidate_id = None
            best_candidate_r = -math.inf

            for neighbor_id in current_neighbor_ids:
                neighbor_agent = self.social_agents[neighbor_id]

                # Find worst current neighbor
                if neighbor_agent.r < worst_neighbor_r:
                    worst_neighbor_id = neighbor_id
                    worst_neighbor_r = neighbor_agent.r

                # Look at friends-of-friends as possible new neighbors
                for candidate_id in self.G.neighbors(neighbor_id):
                    if candidate_id == agent_id:
                        continue

                    if candidate_id in current_neighbor_set:
                        continue

                    candidate_agent = self.social_agents[candidate_id]

                    if candidate_agent.r > best_candidate_r:
                        best_candidate_id = candidate_id
                        best_candidate_r = candidate_agent.r

            # No possible friend-of-friend found
            if best_candidate_id is None:
                continue

            # Only rewire if the candidate is actually better
            if best_candidate_r <= worst_neighbor_r:
                continue

            # Replace worst neighbor with best local candidate
            self.G.remove_edge(agent_id, worst_neighbor_id)
            self.G.add_edge(agent_id, best_candidate_id)

            rewired_edges += 1

        return rewired_edges
