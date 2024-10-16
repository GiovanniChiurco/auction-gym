from typing import List

import numpy as np
import pandas as pd
from Publisher import Publisher
from KnapsackSolver_CTR import get_data, solver


class CUCB:
    def __init__(self, publisher_list: List[Publisher]):
        self.publisher_list = publisher_list
        # Empirical means of the arms (= observed CTRs)
        self.empirical_means = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Empirical means + adjusted term
        self.actual_means = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Time step
        self.t = 0
        # Number of times each arm has been selected
        self.Na = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Saved dataframe with current iteration stats
        self.iteration_stats = None
        # Dataframe with LinUCB parameters per iteration
        self.linucb_params = pd.DataFrame(columns=['Iteration', 'publisher', 'exp_rew', 'ucb'])

    def save_stats(self, iteration_stats: pd.DataFrame):
        # Save the stats of the current iteration keeping the last one
        concat_df = pd.concat([self.iteration_stats, iteration_stats], ignore_index=True)
        self.iteration_stats = concat_df.drop_duplicates(subset=['publisher'], keep='last')

    def update_arm(self, publisher_name: str, reward: float, iteration: int):
        self.Na[publisher_name] += 1
        self.empirical_means[publisher_name] += (reward - self.empirical_means[publisher_name]) / self.Na[publisher_name]
        # Save the parameters
        if not self.linucb_params.empty:
            self.linucb_params = pd.concat([
                self.linucb_params,
                pd.DataFrame({
                    'Iteration': iteration,
                    'publisher': publisher_name,
                    'exp_rew': self.empirical_means[publisher_name],
                    'ucb': self.actual_means[publisher_name]
                }, index=[0])
            ], ignore_index=True)
        else:
            self.linucb_params = pd.DataFrame({
                'Iteration': iteration,
                'publisher': publisher_name,
                'exp_rew': self.empirical_means[publisher_name],
                'ucb': self.actual_means[publisher_name]
            }, index=[0])

    def update(self, publisher_name: str):
        adj_term = np.sqrt((2 * np.log(self.t)) / (self.Na[publisher_name]))
        # actual_mean can be more than 1
        self.actual_means[publisher_name] = self.empirical_means[publisher_name] + adj_term

    def knapsack_solver(
            self, soglia_clicks: float = None, soglia_spent: float = None, soglia_cpc: float = None,
            soglia_num_publisher: int = None, soglia_ctr: float = None
    ) -> List[Publisher]:
        # Add the UCBs to the dataframe
        self.iteration_stats['rew_ucb'] = self.iteration_stats['publisher'].apply(lambda x: self.actual_means[x])
        self.iteration_stats['actual_mean'] = self.iteration_stats['publisher'].apply(lambda x: self.actual_means[x])
        self.iteration_stats['empirical_mean'] = self.iteration_stats['publisher'].apply(lambda x: self.empirical_means[x])
        # Get the data from the dataframe for the solver
        n, clicks, rew_ucb, impressions, spent, cpc = get_data(self.iteration_stats)
        results = solver(df=self.iteration_stats,
                         n=n,
                         rew_ucb=rew_ucb,
                         clicks=clicks,
                         impressions=impressions,
                         spent=spent,
                         cpc=cpc,
                         soglia_spent=soglia_spent,
                         soglia_clicks=soglia_clicks,
                         soglia_cpc=soglia_cpc,
                         soglia_num_publisher=soglia_num_publisher,
                         soglia_ctr=soglia_ctr)
        if results.empty:
            results = self.iteration_stats
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
