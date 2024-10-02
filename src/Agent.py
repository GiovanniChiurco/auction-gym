import numpy as np

from BidderAllocation import PyTorchLogisticRegressionAllocator, OracleAllocator
from Impression import ImpressionOpportunity
from Models import sigmoid


class Agent:
    ''' An agent representing an advertiser '''

    def __init__(self, rng, name, adv_name, num_items, item_values, allocator, bidder, memory=0):
        self.rng = rng
        self.name = name
        # Advertisement name to pick the embeddings
        self.adv_name = adv_name
        self.num_items = num_items

        # Value distribution
        self.item_values = item_values

        self.net_utility = .0
        self.gross_utility = .0

        self.logs = []

        self.allocator = allocator
        self.bidder = bidder

        self.memory = memory

    def select_item(self, context, precomputed_cos_sim=None):
        # Estimate CTR for all items
        # estim_CTRs = self.allocator.estimate_CTR(context)
        estim_CTRs = precomputed_cos_sim
        # Compute value if clicked
        estim_values = estim_CTRs * self.item_values
        # Pick the best item (according to TS)
        best_item = np.argmax(estim_values)

        # If we do Thompson Sampling, don't propagate the noisy bid amount but bid using the MAP estimate
        if type(self.allocator) == PyTorchLogisticRegressionAllocator and self.allocator.thompson_sampling:
            estim_CTRs_MAP = self.allocator.estimate_CTR(context, sample=False)
            return best_item, estim_CTRs_MAP[best_item]

        return best_item, estim_CTRs[best_item]

    def bid(self, context: np.ndarray, publisher_name: str, precomputed_cos_sim: float):
        # First, pick what item we want to choose
        best_item, estimated_CTR = self.select_item(context, precomputed_cos_sim)

        # Sample value for this item
        value = self.item_values[best_item]

        # Get the bid
        bid = self.bidder.bid(value, context, estimated_CTR)

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
                                               publisher=publisher_name))

        return bid, best_item

    def charge(self, price, second_price, outcome):
        self.logs[-1].set_price_outcome(price, second_price, outcome, won=True)
        last_value = self.logs[-1].value * outcome
        self.net_utility += (last_value - price)
        self.gross_utility += last_value

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

    def iteration_stats_per_publisher(self):
        publishers = list(set(opp.publisher for opp in self.logs))
        publishers_data = []
        for publisher in publishers:
            publisher_logs = [opp for opp in self.logs if opp.publisher == publisher]

            publisher_won_logs = [opp for opp in publisher_logs if opp.won]
            publisher_lost_logs = [opp for opp in publisher_logs if not opp.won]

            publisher_won_auctions = len(publisher_won_logs)
            publisher_lost_auctions = len(publisher_lost_logs)
            win_rate = publisher_won_auctions / (publisher_won_auctions + publisher_lost_auctions)
            num_clicks = np.sum([opp.outcome for opp in publisher_won_logs])
            ctr = (num_clicks / publisher_won_auctions) * 100 if publisher_won_auctions > 0 else 0
            true_ctr = np.mean([opp.true_CTR for opp in publisher_won_logs]) if publisher_won_logs else 0
            spent = np.sum([opp.price for opp in publisher_won_logs])
            mean_bid = np.mean([opp.bid for opp in publisher_won_logs]) if publisher_won_logs else 0
            cpc = spent / num_clicks if num_clicks > 0.0 else 0
            cpm = (spent / publisher_won_auctions) * 1000 if publisher_won_auctions > 0 else 0

            publishers_data.append({
                'publisher': publisher,
                'won_auctions': publisher_won_auctions,
                'lost_auctions': publisher_lost_auctions,
                'win_rate': win_rate,
                'num_clicks': num_clicks,
                'ctr': ctr,
                'true_ctr': true_ctr,
                'spent': spent,
                'mean_bid': mean_bid,
                'cpc': cpc,
                'cpm': cpm
            })
        return publishers_data

    # def iteration_stats_per_publisher(self):
    #     publishers = list(set(opp.publisher for opp in self.logs))
    #     publishers_data = []
    #     for publisher in publishers:
    #         publisher_logs = [opp for opp in self.logs if opp.publisher == publisher]
    #
    #         publisher_won_logs = [opp for opp in publisher_logs if opp.won]
    #         publisher_lost_logs = [opp for opp in publisher_logs if not opp.won]
    #
    #         publisher_won_logs_count = len(publisher_won_logs)
    #         publisher_lost_logs_count = len(publisher_lost_logs)
    #         publisher_won_logs_outcome = np.sum([opp.outcome for opp in publisher_won_logs])
    #         publisher_won_logs_ctr = np.mean([opp.true_CTR for opp in publisher_won_logs])
    #         publisher_lost_logs_ctr = np.mean([opp.true_CTR for opp in publisher_lost_logs])
    #         publisher_won_logs_estimated_ctr = np.mean([opp.estimated_CTR for opp in publisher_won_logs])
    #         publisher_lost_logs_estimated_ctr = np.mean([opp.estimated_CTR for opp in publisher_lost_logs])
    #         publisher_won_logs_bid = np.mean([opp.bid for opp in publisher_won_logs])
    #         publisher_lost_logs_bid = np.mean([opp.bid for opp in publisher_lost_logs])
    #         publisher_won_logs_price = np.mean([opp.price for opp in publisher_won_logs])
    #         publisher_lost_logs_price = np.mean([opp.price for opp in publisher_lost_logs])
    #
    #         publishers_data.append({
    #             'publisher': publisher,
    #             'won_auctions': publisher_won_logs_count,
    #             'lost_auctions': publisher_lost_logs_count,
    #             'win_ratio': publisher_won_logs_count / (publisher_won_logs_count + publisher_lost_logs_count),
    #             'num_clicks': int(publisher_won_logs_outcome),
    #             'won_true_CTR': publisher_won_logs_ctr,
    #             'lost_true_CTR': publisher_lost_logs_ctr,
    #             'won_estimated_CTR': publisher_won_logs_estimated_ctr,
    #             'lost_estimated_CTR': publisher_lost_logs_estimated_ctr,
    #             'mean_won_bid': publisher_won_logs_bid,
    #             'mean_lost_bid': publisher_lost_logs_bid,
    #             'mean_won_price': publisher_won_logs_price,
    #             'mean_lost_price': publisher_lost_logs_price,
    #             'spent': self.spending
    #         })
    #     return publishers_data

    def get_publisher_clicks(self, publisher: str):
        return np.sum([opp.outcome for opp in self.logs if opp.publisher == publisher])

    def clear_utility(self):
        self.net_utility = .0
        self.gross_utility = .0

    def clear_logs(self):
        if not self.memory:
            self.logs = []
        else:
            self.logs = self.logs[-self.memory:]
        self.bidder.clear_logs(memory=self.memory)

