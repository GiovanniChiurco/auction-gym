from typing import List

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from Publisher import Publisher
from KnapsackSolver import get_data, solver


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

    def update_arm(self, publisher_name: str, clicks: float, impressions: int):
        self.Na[publisher_name] += 1
        self.exp_clicks[publisher_name] += (clicks - self.exp_clicks[publisher_name]) / self.Na[publisher_name]
        self.exp_impressions[publisher_name] += (impressions - self.exp_impressions[publisher_name]) / self.Na[publisher_name]

    def update(self, publisher_name: str, run: int, iteration: int):
        adj_term = np.sqrt((2 * np.log(self.t)) / (self.Na[publisher_name]))

        self.conf_bound[publisher_name] = self.alpha * adj_term
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

    def extract_estimates(self, run: int, iteration: int):
        click_estimates = pd.DataFrame(self.exp_clicks.items(), columns=['publisher', 'est_clicks'])
        impr_estimates = pd.DataFrame(self.exp_impressions.items(), columns=['publisher', 'est_impressions'])
        confidence_bounds = pd.DataFrame(self.conf_bound.items(), columns=['publisher', 'conf_bound'])
        estimates = pd.merge(click_estimates, impr_estimates, on='publisher')
        estimates = pd.merge(estimates, confidence_bounds, on='publisher')
        estimates['Run'] = run
        estimates['Iteration'] = iteration
        return estimates

    def knapsack_solver(
            self, run: int, iteration: int, soglia_clicks: float = None, soglia_spent: float = None, soglia_cpc: float = None,
            soglia_num_publisher: int = None, soglia_ctr: float = None
    ) -> List[Publisher]:
        # curr_estimates = self.est_ucb.drop_duplicates(subset=['publisher'], keep='last')
        curr_estimates = self.extract_estimates(run=run, iteration=iteration)
        # Add the UCBs to the dataframe
        curr_estimates.loc[:, 'ucb_clicks'] = curr_estimates['est_clicks'] + curr_estimates['conf_bound']
        curr_estimates.loc[:, 'lcb_clicks'] = curr_estimates['est_clicks'] - curr_estimates['conf_bound']
        curr_estimates.loc[:, 'ucb_impressions'] = curr_estimates['est_impressions'] + curr_estimates['conf_bound']
        curr_estimates.loc[:, 'lcb_impressions'] = curr_estimates['est_impressions'] - curr_estimates['conf_bound']
        # Get the data from the dataframe for the solver
        n, ucb_clicks, lcb_clicks, ucb_impressions, lcb_impressions = get_data(curr_estimates)
        results = solver(
            df=curr_estimates,
            n=n,
            ucb_clicks=ucb_clicks,
            lcb_clicks=lcb_clicks,
            ucb_impressions=ucb_impressions,
            lcb_impressions=lcb_impressions,
            soglia_spent=soglia_spent,
            soglia_clicks=soglia_clicks,
            soglia_cpc=soglia_cpc,
            soglia_num_publisher=soglia_num_publisher,
            soglia_ctr=soglia_ctr
        )
        if results.empty:
            # No solution found
            return []
            # results = curr_estimates

        publisher_names = results['publisher'].unique()

        return [
            publisher
            for publisher in self.publisher_list
            if publisher.name in publisher_names
        ]

    def round_iteration(
            self, curr_publisher_list: List[Publisher], run: int, iteration: int, soglia_clicks: float = None, soglia_spent: float = None, soglia_cpc: float = None,
            soglia_num_publisher: int = None, soglia_ctr: float = None
    ) -> List[Publisher]:
        self.t += 1
        # Update actual means
        for publisher in self.publisher_list:
            self.update(publisher.name, run, iteration)
        # Solve the knapsack problem
        selected_publishers = self.knapsack_solver(
            run=run,
            iteration=iteration,
            soglia_clicks=soglia_clicks,
            soglia_spent=soglia_spent,
            soglia_cpc=soglia_cpc,
            soglia_num_publisher=soglia_num_publisher,
            soglia_ctr=soglia_ctr
        )
        if not selected_publishers:
            # No solution found -> return the previous super-arm
            return curr_publisher_list
        return selected_publishers

    def set_time_t(self, t: int):
        self.t = t
