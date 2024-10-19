from collections import deque
from typing import List

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from Publisher import Publisher
from KnapsackSolver_CTR import get_data, solver


class SWCUCB:
    def __init__(self, publisher_list: List[Publisher], window_size: int, alpha: float):
        self.publisher_list = publisher_list
        # Exploration parameter
        self.alpha = alpha
        # Window size
        self.window_size = window_size
        # Current window clicks
        self.window_clicks = {
            publisher.name: deque(maxlen=window_size) for publisher in publisher_list
        }
        # Current window impressions
        self.window_impressions = {
            publisher.name: deque(maxlen=window_size) for publisher in publisher_list
        }
        # Current window Na
        self.window_Na = {
            publisher.name: deque(maxlen=window_size) for publisher in publisher_list
        }
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
        # Dataframe with LinUCB parameters per iteration
        self.est_ucb = None

    def update_arm(self, publisher_name: str, clicks: float, impressions: int, run: int, iteration: int):
        self.window_Na[publisher_name].append(1)

        self.window_clicks[publisher_name].append(clicks)
        self.window_impressions[publisher_name].append(impressions)

        curr_Na = np.sum(self.window_Na[publisher_name])

        curr_clicks = np.sum(self.window_clicks[publisher_name])
        curr_impressions = np.sum(self.window_impressions[publisher_name])

        self.exp_clicks[publisher_name] = curr_clicks / curr_Na
        self.exp_impressions[publisher_name] = curr_impressions / curr_Na
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
        curr_Na = np.sum(self.window_Na[publisher_name])
        if curr_Na == 0:
            adj_term = 0 
        else:
            adj_term = np.sqrt((2 * np.log(min(self.window_size, self.t))) / curr_Na)

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
        # Add 0s to the window for the publishers not selected
        for publisher in self.publisher_list:
            if publisher not in selected_publishers:
                self.window_clicks[publisher.name].append(0)
                self.window_impressions[publisher.name].append(0)
                self.window_Na[publisher.name].append(0)
        return selected_publishers

    def set_time_t(self, t: int):
        self.t = t
