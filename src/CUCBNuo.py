from typing import List

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from Publisher import Publisher
from KnapsackSolver_CTR import get_data, solver


class CUCBNuo:
    def __init__(self, publisher_list: List[Publisher], alpha: float):
        self.publisher_list = publisher_list
        self.alpha = alpha
        # Expected clicks
        self.exp_clicks = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Expected impressions
        self.exp_impressions = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Confidence Bound
        self.conf_bound = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Time step
        self.t = 0
        # Number of times each arm has been selected
        self.Na = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Dataframe with LinUCB parameters per iteration
        self.est_ucb = None

    def update_arm(self, publisher_name: str, clicks: float, impressions: int, run: int, iteration: int):
        self.Na[publisher_name] += 1
        self.exp_clicks[publisher_name] += (clicks - self.exp_clicks[publisher_name]) / self.Na[publisher_name]
        self.exp_impressions[publisher_name] += (impressions - self.exp_impressions[publisher_name]) / self.Na[publisher_name]
        # Save the parameters
        if self.est_ucb is None:
            self.est_ucb = pd.DataFrame({
                'Iteration': iteration,
                'Run': run,
                'publisher': publisher_name,
                'est_clicks': self.exp_clicks[publisher_name],
                'est_impressions': self.exp_impressions[publisher_name],
                'conf_bound': self.conf_bound[publisher_name]
            }, index=[0])
        else:
            self.est_ucb = pd.concat([
                self.est_ucb,
                pd.DataFrame({
                    'Iteration': iteration,
                    'Run': run,
                    'publisher': publisher_name,
                    'est_clicks': self.exp_clicks[publisher_name],
                    'est_impressions': self.exp_impressions[publisher_name],
                    'conf_bound': self.conf_bound[publisher_name]
                }, index=[0])
            ])

    def update(self, publisher_name: str):
        adj_term = np.sqrt((2 * np.log(self.t)) / (self.Na[publisher_name]))

        self.conf_bound[publisher_name] = self.alpha * adj_term

    def knapsack_solver(
            self, soglia_clicks: float = None, soglia_spent: float = None, soglia_cpc: float = None,
            soglia_num_publisher: int = None, soglia_ctr: float = None
    ) -> List[Publisher]:
        curr_estimates = self.est_ucb.drop_duplicates(subset=['publisher'], keep='last')
        # Add the UCBs to the dataframe
        curr_estimates.loc[:, 'lcb_clicks'] = curr_estimates['est_clicks'] - curr_estimates['conf_bound']
        curr_estimates.loc[:, 'ucb_impressions'] = curr_estimates['est_impressions'] + curr_estimates['conf_bound']
        # Get the data from the dataframe for the solver
        n, clicks, impressions = get_data(curr_estimates)
        results = solver(
            df=curr_estimates,
            n=n,
            clicks=clicks,
            impressions=impressions,
            soglia_spent=soglia_spent,
            soglia_clicks=soglia_clicks,
            soglia_cpc=soglia_cpc,
            soglia_num_publisher=soglia_num_publisher,
            soglia_ctr=soglia_ctr
        )
        if results.empty:
            results = curr_estimates
        publisher_names = results['publisher'].unique()

        return [
            publisher
            for publisher in self.publisher_list
            if publisher.name in publisher_names
        ]

    def round_iteration(
            self, soglia_clicks: float = None, soglia_spent: float = None, soglia_cpc: float = None,
            soglia_num_publisher: int = None, soglia_ctr: float = None
    ) -> List[Publisher]:
        self.t += 1
        # Update actual means
        for publisher in self.publisher_list:
            self.update(publisher.name)
        # Solve the knapsack problem
        selected_publishers = self.knapsack_solver(
            soglia_clicks=soglia_clicks,
            soglia_spent=soglia_spent,
            soglia_cpc=soglia_cpc,
            soglia_num_publisher=soglia_num_publisher,
            soglia_ctr=soglia_ctr
        )
        return selected_publishers

    def set_time_t(self, t: int):
        self.t = t
