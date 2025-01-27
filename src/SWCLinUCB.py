from typing import List
import numpy as np
import pandas as pd
import scipy.linalg
pd.options.mode.chained_assignment = None
from Publisher import Publisher
from KnapsackSolver import get_data, solver


class SWCombinatorialLinUCBOpt:
    def __init__(self, alpha: float, d: int, publisher_list: List[Publisher], window_size: int):
        self.alpha = alpha
        self.window_size = window_size
        self.t = 0
        # Embedding size
        self.d = d
        self.publisher_list = publisher_list
        self.n_arms = len(publisher_list)
        self.A = np.eye(d)
        # Confidence Bound
        self.conf_bound = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Parametri per stimare i click
        self.b_click = np.zeros(d)
        self.theta_click = np.zeros(d)
        self.est_click = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Parametri per stimare le impression
        self.b_impr = np.zeros(d)
        self.theta_impr = np.zeros(d)
        self.est_impr = {
            publisher.name: 0 for publisher in publisher_list
        }
        self.curr_superarm = {}
        self.curr_superarm_stats = None
        self.linucb_params = None

    def save_params(self):
        return self.theta_click, self.theta_impr

    def add_new_arm(self, publisher: Publisher):
        self.n_arms += 1
        self.publisher_list.append(publisher)
        self.conf_bound[publisher.name] = 0
        self.est_click[publisher.name] = 0
        self.est_impr[publisher.name] = 0

    def update_arm(self, L, lower, publisher: Publisher, run: int, iteration: int):
        # Matrix inversion with Cholesky decomposition
        embedding = publisher.embedding
        x = scipy.linalg.cho_solve((L, lower), embedding)
        self.conf_bound[publisher.name] = self.alpha * np.sqrt(embedding.dot(x))
        # Aggiorna le stime di click e impression
        self.est_click[publisher.name] = max(0, np.dot(self.theta_click, embedding))
        self.est_impr[publisher.name] = max(1, np.dot(self.theta_impr, embedding))
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
                    'Run': run,
                    'publisher': publisher.name,
                    'est_clicks': self.est_click[publisher.name],
                    'est_impressions': self.est_impr[publisher.name],
                    'conf_bound': self.conf_bound[publisher.name]
                }, index=[0])
            ], ignore_index=True)

    def extract_estimates(self, run: int, iteration: int):
        click_estimates = pd.DataFrame(self.est_click.items(), columns=['publisher', 'est_clicks'])
        impr_estimates = pd.DataFrame(self.est_impr.items(), columns=['publisher', 'est_impressions'])
        confidence_bounds = pd.DataFrame(self.conf_bound.items(), columns=['publisher', 'conf_bound'])
        estimates = pd.merge(click_estimates, impr_estimates, on='publisher')
        estimates = pd.merge(estimates, confidence_bounds, on='publisher')
        estimates['Run'] = run
        estimates['Iteration'] = iteration
        return estimates
    
    def compute_theta(self):
        L, lower = scipy.linalg.cho_factor(self.A, lower=True)

        self.theta_click = scipy.linalg.cho_solve((L, lower), self.b_click)
        self.theta_impr = scipy.linalg.cho_solve((L, lower), self.b_impr)

        return L, lower

    def add_miss_rows(self, publisher_list: List[Publisher], run: int, iteration: int):
        for publisher in publisher_list:
            self.linucb_params = pd.concat([
                self.linucb_params,
                pd.DataFrame({
                    'Iteration': iteration,
                    'Run': run,
                    'publisher': publisher.name,
                    'est_clicks': self.est_click[publisher.name],
                    'est_impressions': self.est_impr[publisher.name],
                    'conf_bound': self.conf_bound[publisher.name]
                }, index=[0])
            ], ignore_index=True)

    def round_iteration(
            self, curr_publisher_list: List[Publisher], run: int, iteration: int, soglia_clicks: float = None,
            soglia_spent: float = None, soglia_cpc: float = None, soglia_num_publisher: int = None, soglia_ctr: float = None) -> List[Publisher]:
        self.t += 1
        # Update A and thetas
        L, lower = self.compute_theta()
        for publisher in self.publisher_list:
            # if not self.check_publisher_exist(publisher):
            #     self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(L, lower, publisher=publisher, run=run, iteration=iteration)
        # Ripeto i dati già presenti per statistiche successive
        # not_updated_publishers = [publisher for publisher in self.publisher_list if publisher not in curr_publisher_list]
        # self.add_miss_rows(not_updated_publishers, run, iteration)
        # Select the super-arm
        # il parametro publisher_list non viene passato al solver perché i dati necessari sono già presenti nel dataframe iteration_stats
        super_arm = self.knapsack_solver(
            run=run,
            iteration=iteration,
            soglia_spent=soglia_spent,
            soglia_clicks=soglia_clicks,
            soglia_cpc=soglia_cpc,
            soglia_num_publisher=soglia_num_publisher,
            soglia_ctr=soglia_ctr
        )
        
        if not super_arm:
            self.curr_superarm[self.t] = curr_publisher_list
            # No solution found -> return the previous super-arm
            return curr_publisher_list
        self.curr_superarm[self.t] = super_arm
        # Return the super-arm
        return super_arm

    def update(self, agent_stats_pub: List[dict], init_publisher_embeddings: dict):
        for publisher_data in agent_stats_pub:
            embedding = init_publisher_embeddings[publisher_data['publisher']]
            self.A += np.outer(embedding, embedding)
            self.b_click += publisher_data['clicks'] * embedding
            self.b_impr += publisher_data['impressions'] * embedding

            curr_pub_stats = pd.DataFrame({
                't': self.t,
                'publisher': publisher_data['publisher'],
                'clicks': publisher_data['clicks'],
                'impressions': publisher_data['impressions']
            }, index=[0])

            if self.curr_superarm_stats is None:
                self.curr_superarm_stats = curr_pub_stats
            else:
                self.curr_superarm_stats = pd.concat([self.curr_superarm_stats, curr_pub_stats], ignore_index=True)
        if self.t > self.window_size - 1:
            old_time = self.t - self.window_size
            for old_pub in self.curr_superarm[old_time]:
                self.A -= np.outer(old_pub.embedding, old_pub.embedding)
                old_pub_stats = self.curr_superarm_stats[(self.curr_superarm_stats['publisher'] == old_pub.name) & (self.curr_superarm_stats['t'] == old_time)]
                self.b_click -= old_pub_stats['clicks'].values[0] * old_pub.embedding
                self.b_impr -= old_pub_stats['impressions'].values[0] * old_pub.embedding


    def initial_round(
            self, run: int, iteration: int, curr_publisher_list: List[Publisher]
    ):
        L, lower = self.compute_theta()
        # Check if there are new arms (= new publishers in the list)
        for publisher in self.publisher_list:
            if not self.check_publisher_exist(publisher):
                self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(L, lower, publisher=publisher, run=run, iteration=iteration)
        # Save the super-arm for the current timestamp
        self.curr_superarm[iteration] = curr_publisher_list

    def check_publisher_exist(self, publisher: Publisher):
        for pub in self.publisher_list:
            if pub.name == publisher.name:
                return True
        return False

    def knapsack_solver(
            self, run: int, iteration: int, soglia_clicks: float = None, soglia_spent: float = None, soglia_cpc: float = None,
            soglia_num_publisher: int = None, soglia_ctr: float = None
    ) -> List[Publisher]:
        curr_estimates = self.extract_estimates(run=run, iteration=iteration)
        # Add the UCBs to the dataframe
        curr_estimates.loc[:, 'ucb_clicks'] = curr_estimates['est_clicks'] + curr_estimates['conf_bound']
        curr_estimates.loc[:, 'lcb_clicks'] = np.maximum(0, curr_estimates['est_clicks'] - curr_estimates['conf_bound'])
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
        
        publisher_names = results['publisher'].unique()
        return [
            publisher
            for publisher in self.publisher_list
            if publisher.name in publisher_names
        ]
