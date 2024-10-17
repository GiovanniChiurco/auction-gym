from typing import List
import numpy as np
import pandas as pd

from Publisher import Publisher
from Publisher_Reward import PublisherReward
from KnapsackSolver import get_data, solver


class CombinatorialLinUCBNuo:
    """ LinUCB algorithm """
    def __init__(self, alpha: float, d: int, publisher_list: List[Publisher]):
        self.alpha = alpha
        # Embedding size
        self.d = d
        self.publisher_list = publisher_list
        self.n_arms = len(publisher_list)
        self.A = {
            publisher.name: np.eye(d) for publisher in publisher_list
        }
        # Confidence Bound
        self.conf_bound = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Parametri per stimare i click
        self.b_click = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        self.theta_click = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        self.est_click = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Parametri per stimare le impression
        self.b_impr = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        self.theta_impr = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        self.est_impr = {
            publisher.name: 0 for publisher in publisher_list
        }
        self.linucb_params = None

    def add_new_arm(self, publisher: Publisher):
        self.n_arms += 1
        self.publisher_list.append(publisher.name)
        self.A[publisher.name] = np.eye(self.d)
        # Initialize the new arm parameters
        self.b_click[publisher.name] = np.zeros(self.d)
        self.b_impr[publisher.name] = np.zeros(self.d)

    def update_arm(self, publisher: Publisher, run: int, iteration: int):
        self.conf_bound[publisher.name] = self.alpha * np.sqrt(
            publisher.embedding.dot(np.linalg.inv(self.A[publisher.name]).dot(publisher.embedding)))
        A_inv = np.linalg.inv(self.A[publisher.name])
        # Update click parameters
        self.theta_click[publisher.name] = A_inv.dot(self.b_click[publisher.name])
        self.est_click[publisher.name] = self.theta_click[publisher.name].dot(publisher.embedding)
        # Update impression parameters
        self.theta_impr[publisher.name] = A_inv.dot(self.b_impr[publisher.name])
        self.est_impr[publisher.name] = self.theta_impr[publisher.name].dot(publisher.embedding)
        # Save the parameters
        if self.linucb_params is None:
            self.linucb_params = pd.DataFrame({
                'Iteration': iteration,
                'Run': run,
                'publisher': publisher.name,
                'est_clicks': self.est_click[publisher.name],
                'est_impressions': self.est_impr[publisher.name],
                'conf_bound': self.conf_bound[publisher.name]
            }, index=[0])
        else:
            self.linucb_params = pd.concat([
                self.linucb_params,
                pd.DataFrame({
                    'Iteration': iteration,
                    'publisher': publisher.name,
                    'est_clicks': self.est_click[publisher.name],
                    'est_impressions': self.est_impr[publisher.name],
                    'conf_bound': self.conf_bound[publisher.name]
                }, index=[0])
            ], ignore_index=True)

    def round_iteration(
            self, publisher_list: List[Publisher], run: int, iteration: int, soglia_clicks: float = None,
            soglia_spent: float = None, soglia_cpc: float = None, soglia_num_publisher: int = None, soglia_ctr: float = None) -> List[Publisher]:
        # Check if there are new arms (= new publishers in the list)
        for publisher in publisher_list:
            if not self.check_publisher_exist(publisher):
                self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(publisher=publisher, run=run, iteration=iteration)
        # Select the super-arm
        # il parametro publisher_list non viene passato al solver perché i dati necessari sono già presenti nel dataframe iteration_stats
        super_arm = self.knapsack_solver(
            soglia_spent=soglia_spent,
            soglia_clicks=soglia_clicks,
            soglia_cpc=soglia_cpc,
            soglia_num_publisher=soglia_num_publisher,
            soglia_ctr=soglia_ctr
        )
        # Return the super-arm
        return super_arm

    def update(self, publisher_name: str, publisher_embedding: np.array, clicks: float | int, impressions: float | int, iteration: int):
        # Method to update the parameters of the selected arm
        self.A[publisher_name] += np.outer(publisher_embedding, publisher_embedding)
        # Update click parameters
        self.b_click[publisher_name] += clicks * publisher_embedding
        # Update impression parameters
        self.b_impr[publisher_name] += impressions * publisher_embedding

    def initial_round(
            self, publisher_list: List[Publisher], iteration: int,
    ):
        # Check if there are new arms (= new publishers in the list)
        for publisher in publisher_list:
            if not self.check_publisher_exist(publisher):
                self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(publisher=publisher, iteration=iteration)

    def check_publisher_exist(self, publisher: Publisher):
        for pub in self.publisher_list:
            if pub.name == publisher.name:
                return True
        return False

    def knapsack_solver(
            self, soglia_clicks: float = None, soglia_spent: float = None, soglia_cpc: float = None,
            soglia_num_publisher: int = None, soglia_ctr: float = None
    ) -> List[Publisher]:
        # Add the UCBs to the dataframe
        self.linucb_params['lcb_clicks'] = self.linucb_params['est_clicks'] - self.linucb_params['conf_bound']
        self.linucb_params['ucb_impressions'] = self.linucb_params['est_impressions'] + self.linucb_params['conf_bound']
        # Get the data from the dataframe for the solver
        n, clicks, impressions = get_data(self.linucb_params)
        results = solver(
            df=self.linucb_params,
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
            results = self.linucb_params
        publisher_names = results['publisher'].unique()
        return [
            publisher
            for publisher in self.publisher_list
            if publisher.name in publisher_names
        ]
