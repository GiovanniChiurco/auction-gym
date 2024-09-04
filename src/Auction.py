from AuctionAllocation import AllocationMechanism
from Bidder import Bidder

import numpy as np

from BidderAllocation import OracleAllocator
from Models import sigmoid

from sklearn.metrics.pairwise import cosine_similarity


class Auction:
    ''' Base class for auctions '''
    def __init__(self, rng, allocation, agents, agent2items, agents2item_values, max_slots, num_participants_per_round):
        self.rng = rng
        self.allocation = allocation
        self.agents = agents
        self.max_slots = max_slots
        self.revenue = .0

        self.agent2items = agent2items
        self.agents2item_values = agents2item_values

        self.num_participants_per_round = num_participants_per_round

    def simulate_opportunity(self, all_users_agent_similarity, publisher_name, num_iteration, true_context):
        # Sample the number of slots uniformly between [1, max_slots]
        num_slots = self.rng.integers(1, self.max_slots + 1)

        # At this point, the auctioneer solicits bids from
        # the list of bidders that might want to compete.
        bids = []
        CTRs = []
        # Select the agents that are eligible to participate with respect to their budget
        eligible_agents = self.check_agents_budget()
        # If there are no agents with budget, we don't need to simulate the auction
        if len(eligible_agents) != 0:
            # Ensure that we don't have more participants than agents
            num_participants = min(len(eligible_agents), self.num_participants_per_round)
            participating_agents_idx = self.rng.choice(len(eligible_agents), num_participants, replace=False)
            participating_agents = [eligible_agents[idx] for idx in participating_agents_idx]
            for agent in participating_agents:
                # Get the bid and the allocated item
                if isinstance(agent.allocator, OracleAllocator):
                    bid, item = agent.bid(all_users_agent_similarity, publisher_name, num_iteration, true_context)
                else:
                    bid, item = agent.bid(all_users_agent_similarity, publisher_name, num_iteration, true_context)
                bids.append(bid)
                # Compute the true CTRs for items in this agent's catalogue
                # true_CTR = sigmoid(true_context @ self.agent2items[agent.name].T)

                # Add this line when we use text embeddings and comment the one above
                # true_CTR = cosine_similarity([true_context], [self.agent2items[agent.name]])[0]
                # Already true CTR
                true_CTR = all_users_agent_similarity[publisher_name][agent.name][:, num_iteration]

                agent.logs[-1].set_true_CTR(np.max(true_CTR * self.agents2item_values[agent.name]), true_CTR[item])
                CTRs.append(true_CTR[item])
            bids = np.array(bids)
            CTRs = np.array(CTRs)

            # Now we have bids, we need to somehow allocate slots
            # "second_prices" tell us how much lower the winner could have gone without changing the outcome
            winners, prices, second_prices = self.allocation.allocate(bids, num_slots)

            # Bidders only obtain value when they get their outcome
            # Either P(view), P(click | view, ad), P(conversion | click, view, ad)
            # For now, look at P(click | ad) * P(view)
            outcomes = self.rng.binomial(1, CTRs[winners])
        else:
            participating_agents = self.agents

            winners = [-1]
            prices = [0.0]
            second_prices = [0.0]
            outcomes = [0]

        # Let bidders know what they're being charged for
        for slot_id, (winner, price, second_price, outcome) in enumerate(zip(winners, prices, second_prices, outcomes)):
            for agent_id, agent in enumerate(participating_agents):
                if agent_id == winner:
                    agent.charge(price, second_price, bool(outcome))
                else:
                    agent.set_price(price)
            self.revenue += price

    def clear_revenue(self):
        self.revenue = 0.0

    def check_agents_budget(self):
        eligible_agents = []
        for agent in self.agents:
            if agent.budget - agent.spending > 1:
                eligible_agents.append(agent)
        return eligible_agents
