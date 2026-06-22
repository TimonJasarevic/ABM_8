import mesa
import networkx as nx

from mesa.space import NetworkGrid
from definitions import NetworkType, AgentType, MessageState
import agents


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
        self.fake_penalty = fake_penalty   # wealth/reputation loss for spreading fake news
        self.rationality = rationality     # sharpness of the verify best-response

        self.G = None
        self.grid = None
        self.influencer_nodes = []

        self.social_agents = []
        self.influencers = []

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
                self.influencers.append(agent)

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


    def simulation_step(self, max_steps=20):
        self._reset_message_states()

        # choose who starts the message
        initiator_agent = self._choose_initiator(AgentType.Influencer)

        # initiator creates the message
        message = initiator_agent.initiate_message()

        # active_sharers contains agent objects
        active_sharers = [initiator_agent]

        step = 0

        while active_sharers and step < max_steps:
            exposed_nodes = set()

            # collect all neighbors that receive the message this step
            for sender_agent in active_sharers:
                for neighbor_node in sender_agent.neighbor_ids:
                    neighbor_agent = self.social_agents[neighbor_node]

                    # only expose agents that have not received this message yet
                    if neighbor_agent.message_state == MessageState.Unaware:
                        exposed_nodes.add(neighbor_node)

            new_sharers = []

            # let exposed agents decide whether they share further
            for node in exposed_nodes:
                receiver_agent = self.social_agents[node]

                #TODO: fix so sender is not initiator but previous node
                agent_response = receiver_agent.receive_message(
                    message,
                    initiator_agent.node_id,
                    initiator_agent.r
                )

                # assuming 1 means "shares further"
                if agent_response == 1:
                    new_sharers.append(receiver_agent)

            active_sharers = new_sharers
            step += 1

    #TODO: fix for no initiators
    def _choose_initiator(self, seed_type):
        if seed_type == AgentType.Influencer:
            return self.random.choice(self.influencers)
        elif seed_type == AgentType.NormalUser:       
            return self.random.choice(
                list(set(self.social_agents).difference(self.influencers))
            )
            
    def _reset_message_states(self):
        for agent in self.social_agents:
            agent.message_state = MessageState.Unaware