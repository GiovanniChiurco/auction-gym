import numpy as np
import pandas as pd

from BidderAllocation import PyTorchLogisticRegressionAllocator, OracleAllocator
from Impression import ImpressionOpportunity
from Models import sigmoid


class Agent:
    ''' An agent representing an advertiser '''

    def __init__(self, rng, name, num_items, item_values, allocator, bidder, budget, memory=0):
        self.rng = rng
        self.name = name
        self.num_items = num_items

        # Value distribution
        self.item_values = item_values

        self.net_utility = .0
        self.gross_utility = .0
        # Keep track of spending
        self.spending = .0
        # Budget
        self.budget = budget

        self.logs = []

        self.allocator = allocator
        self.bidder = bidder

        self.memory = memory

    def select_item(self, all_users_agent_similarity, publisher_name, num_iteration):
        # Estimate CTR for all items
        #estim_CTRs = self.allocator.estimate_CTR(context)
        estim_CTRs = all_users_agent_similarity[publisher_name][self.name][:, num_iteration]
        # Compute value if clicked
        estim_values = estim_CTRs * self.item_values
        # Pick the best item (according to TS)
        best_item = np.argmax(estim_values)

        # If we do Thompson Sampling, don't propagate the noisy bid amount but bid using the MAP estimate
        # if type(self.allocator) == PyTorchLogisticRegressionAllocator and self.allocator.thompson_sampling:
        #     estim_CTRs_MAP = self.allocator.estimate_CTR(context, sample=False)
        #     return best_item, estim_CTRs_MAP[best_item]

        return best_item, estim_CTRs[best_item]

    def bid(self, all_users_agent_similarity, publisher_name, num_iteration, context):
        # First, pick what item we want to choose
        best_item, estimated_CTR = self.select_item(all_users_agent_similarity, publisher_name, num_iteration)

        # Sample value for this item
        value = self.item_values[best_item]

        # Get the bid
        bid = self.bidder.bid(value, context, estimated_CTR)
        # Check if our bid is > budget
        if self.spending + bid > self.budget:
            bid = 0.0

        # Log what we know so far
        self.logs.append(ImpressionOpportunity(context=context,
                                               item=best_item,
                                               estimated_CTR=estimated_CTR,
                                               value=value,
                                               bid=bid,
                                               # These will be filled out later
                                               best_expected_value=0.0,
                                               true_CTR=0.0,
                                               price=0.0,
                                               second_price=0.0,
                                               outcome=0,
                                               won=False,
                                               publisher_name=publisher_name))

        return bid, best_item

    def charge(self, price, second_price, outcome):
        self.logs[-1].set_price_outcome(price, second_price, outcome, won=True)
        last_value = self.logs[-1].value * outcome
        self.net_utility += (last_value - price)
        self.gross_utility += last_value
        # Increment spent
        self.spending += price

    def set_price(self, price):
        self.logs[-1].set_price(price)

    def update(self, iteration, plot=False, figsize=(8,5), fontsize=14):
        # Gather relevant logs
        contexts = np.array(list(opp.context for opp in self.logs))
        items = np.array(list(opp.item for opp in self.logs))
        values = np.array(list(opp.value for opp in self.logs))
        bids = np.array(list(opp.bid for opp in self.logs))
        prices = np.array(list(opp.price for opp in self.logs))
        outcomes = np.array(list(opp.outcome for opp in self.logs))
        estimated_CTRs = np.array(list(opp.estimated_CTR for opp in self.logs))

        # Update response model with data from winning bids
        won_mask = np.array(list(opp.won for opp in self.logs))
        self.allocator.update(contexts[won_mask], items[won_mask], outcomes[won_mask], iteration, plot, figsize, fontsize, self.name)

        # Update bidding model with all data
        self.bidder.update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, self.name)

    def get_allocation_regret(self):
        ''' How much value am I missing out on due to suboptimal allocation? '''
        return np.sum(list(opp.best_expected_value - opp.true_CTR * opp.value for opp in self.logs))

    def get_estimation_regret(self):
        ''' How much am I overpaying due to over-estimation of the value? '''
        return np.sum(list(opp.estimated_CTR * opp.value - opp.true_CTR * opp.value for opp in self.logs))

    def get_overbid_regret(self):
        ''' How much am I overpaying because I could shade more? '''
        return np.sum(list((opp.price - opp.second_price) * opp.won for opp in self.logs))

    def get_underbid_regret(self):
        ''' How much have I lost because I could have shaded less? '''
        # The difference between the winning price and our bid -- for opportunities we lost, and where we could have won without overpaying
        # Important to mention that this assumes a first-price auction! i.e. the price is the winning bid
        return np.sum(list((opp.price - opp.bid) * (not opp.won) * (opp.price < (opp.true_CTR * opp.value)) for opp in self.logs))

    def get_CTR_RMSE(self):
        return np.sqrt(np.mean(list((opp.true_CTR - opp.estimated_CTR)**2 for opp in self.logs)))

    def get_CTR_bias(self):
        return np.mean(list((opp.estimated_CTR / opp.true_CTR) for opp in filter(lambda opp: opp.won, self.logs)))

    def get_agent_spent(self) -> float:
        spent = 0
        for opp in self.logs:
            if opp.won:
                spent += opp.bid
        return spent

    def get_agent_opp_won(self) -> int:
        opp_won = 0
        for opp in self.logs:
            if opp.won:
                opp_won += 1
        return opp_won

    def get_mean_won_bid(self) -> float:
        spent_won_bids = self.get_agent_spent()
        num_won_bids = self.get_agent_opp_won()
        if num_won_bids == 0:
            return 0
        return spent_won_bids / num_won_bids

    def get_most_bid_publisher(self) -> pd.DataFrame:
        pub_names = [opp.publisher_name for opp in self.logs if opp.won]
        pub_df = pd.DataFrame({'Name': pub_names})
        pub_df = pub_df.groupby(by='Name') \
            .size() \
            .reset_index(name='Occurrences') \
            .sort_values(by='Occurrences', ascending=False)
        return pub_df

    def get_agent_bid_df(self) -> pd.DataFrame:
        items = [opp.item for opp in self.logs]
        values = [opp.value for opp in self.logs]
        bids = [opp.bid for opp in self.logs]
        prices = [opp.price for opp in self.logs]
        true_CTR = [opp.true_CTR for opp in self.logs]
        estimated_CTR = [opp.estimated_CTR for opp in self.logs]
        won = [opp.won for opp in self.logs]
        publisher_name = [opp.publisher_name for opp in self.logs]

        df = pd.DataFrame({
            'Item': items,
            'Value': values,
            'Bid': bids,
            'Price': prices,
            'True CTR': true_CTR,
            'Estimated CTR': estimated_CTR,
            'Won': won,
            'Publisher Name': publisher_name
        })
        return df

    def agent_stats_won_publisher(self) -> pd.DataFrame:
        agent_df = self.get_agent_bid_df()
        agent_df = agent_df[agent_df['Won'] == True]
        if agent_df.empty:
            print('No opportunities won')
            return agent_df
        agent_df = agent_df.groupby(by='Publisher Name') \
            .agg({'Value': 'mean', 'Bid': 'mean', 'Price': 'mean', 'Won': 'count'}) \
            .reset_index()
        agent_df = agent_df.sort_values(by='Won', ascending=False)
        return agent_df

    def get_agent_won_ratio(self) -> float:
        num_won = 0
        for opp in self.logs:
            if opp.won:
                num_won += 1
        return num_won / len(self.logs)

    def clear_utility(self):
        self.net_utility = .0
        self.gross_utility = .0
        self.spending = .0

    def clear_logs(self):
        if not self.memory:
            self.logs = []
        else:
            self.logs = self.logs[-self.memory:]
        self.bidder.clear_logs(memory=self.memory)

