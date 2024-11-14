from collections import deque
from typing import List
import numpy as np
import scipy.linalg
import pandas as pd
pd.options.mode.chained_assignment = None
from Publisher import Publisher
from Publisher_Reward import PublisherReward
from KnapsackSolver import get_data, solver


class SWCLinUCB_arce_v:
    """ 
    Sliding Window Combinatorial LinUCB algorithm
    https://ieeexplore.ieee.org/document/6694077
    qunado t>ws, A rimane invariata
    """
    def __init__(self, alpha: float, d: int, publisher_list: List[Publisher], window_size: int):
        self.alpha = alpha
        # Embedding size
        self.d = d
        self.publisher_list = publisher_list
        self.n_arms = len(publisher_list)
        self.window_size = window_size
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
        # Current time step
        self.t = 0
        # History of the selected arms (only names)
        self.h_superarms = {
            0: []
        }
        # History of the clicks and impressions
        self.h_clicks_impressions = None
        self.linucb_params = None

    def add_new_arm(self, publisher: Publisher):
        self.n_arms += 1
        self.publisher_list.append(publisher.name)
        # Initialize the new arm parameters
        self.est_click[publisher.name] = 0
        self.est_impr[publisher.name] = 0
        self.conf_bound[publisher.name] = 0

    def update_arm(self, publisher: Publisher, run: int, iteration: int):
        # Calcola la fattorizzazione di Cholesky di A (la parte triangolare inferiore di A)
        L, lower = scipy.linalg.cho_factor(self.A, lower=True)
        embedding = publisher.embedding  # Salvare embedding per evitare lookup ripetuti
        # Aggiorna il confine di confidenza usando la fattorizzazione di Cholesky
        # Risolvi Lx = embedding (forward substitution) e quindi L^Ty = x (back substitution)
        x = scipy.linalg.cho_solve((L, lower), embedding)
        self.conf_bound[publisher.name] = self.alpha * np.sqrt(embedding.dot(x))
        # Aggiorna i parametri click e stima usando la fattorizzazione di Cholesky
        self.theta_click = scipy.linalg.cho_solve((L, lower), self.b_click)
        self.est_click[publisher.name] = np.dot(self.theta_click, embedding)
        # Aggiorna i parametri impression e stima
        self.theta_impr = scipy.linalg.cho_solve((L, lower), self.b_impr)
        self.est_impr[publisher.name] = np.dot(self.theta_impr, embedding)
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
        # Check if there are new arms (= new publishers in the list)
        for publisher in curr_publisher_list:
            # if not self.check_publisher_exist(publisher):
            #     self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(publisher=publisher, run=run, iteration=iteration)
        # Ripeto i dati già presenti per statistiche successive
        not_updated_publishers = [publisher for publisher in self.publisher_list if publisher not in curr_publisher_list]
        self.add_miss_rows(not_updated_publishers, run, iteration)
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
            self.h_superarms[self.t] = [publisher.name for publisher in curr_publisher_list]
            # No solution found -> return the previous super-arm
            return curr_publisher_list
        self.h_superarms[self.t] = [publisher.name for publisher in super_arm]
        # Return the super-arm
        return super_arm

    # def update(self, agent_stats_pub: List[dict], init_publisher_embeddings: dict):
    #     dot_prod_emb = 0
    #     dot_prod_click = 0
    #     dot_prod_impr = 0
    #     for publisher_data in agent_stats_pub:
    #         publisher_embedding = init_publisher_embeddings[publisher_data['publisher']]
    #         dot_prod_emb += np.outer(publisher_embedding, publisher_embedding)
    #         dot_prod_click += publisher_data['clicks'] * publisher_embedding
    #         dot_prod_impr += publisher_data['impressions'] * publisher_embedding
    #     self.A += dot_prod_emb
    #     self.b_click += dot_prod_click
    #     self.b_impr += dot_prod_impr
    #     if self.t > self.window_size:
    #         # Rimuovi i dati più vecchi
    #         oldest_data = self.h_superarms[self.t - self.window_size]
    #         for publisher_name in oldest_data:
    #             publisher_embedding = init_publisher_embeddings[publisher_name]
    #             self.A -= np.outer(publisher_embedding, publisher_embedding)
                
    #             self.b_click -= agent_stats_pub[publisher_name]['clicks'] * publisher_embedding
    #             self.b_impr -= agent_stats_pub[publisher_name]['impressions'] * publisher_embedding
    
    def update(self, agent_stats_pub: List[dict], init_publisher_embeddings: dict):
        # Salvo i dati correnti per dimenticarli in futuro
        agent_stats_pub_df = pd.DataFrame(agent_stats_pub)
        agent_stats_pub_df['timestamp'] = self.t
        if self.h_clicks_impressions is None:
            self.h_clicks_impressions = agent_stats_pub_df
        else:
            self.h_clicks_impressions = pd.concat([self.h_clicks_impressions, agent_stats_pub_df], ignore_index=True)

        publisher_embeddings = np.array([init_publisher_embeddings[pd['publisher']] for pd in agent_stats_pub])
        clicks = np.array([pd['clicks'] for pd in agent_stats_pub])
        impressions = np.array([pd['impressions'] for pd in agent_stats_pub])

        # Calcola i prodotti scalari
        dot_prod_emb = np.einsum('ij,ik->jk', publisher_embeddings, publisher_embeddings)
        dot_prod_click = np.dot(clicks, publisher_embeddings)
        dot_prod_impr = np.dot(impressions, publisher_embeddings)

        # Aggiorna A, b_click e b_impr
        self.A += dot_prod_emb
        self.b_click += dot_prod_click
        self.b_impr += dot_prod_impr

        if self.t > self.window_size:
            # Rimuovi i dati più vecchi
            oldest_arm = self.h_superarms[self.t - self.window_size]

            oldest_embeddings = np.array([init_publisher_embeddings[pn] for pn in oldest_arm])
            oldest_dot_prod_emb = np.einsum('ij,ik->jk', oldest_embeddings, oldest_embeddings)

            self.A -= oldest_dot_prod_emb

            oldest_clicks = self.h_clicks_impressions[self.h_clicks_impressions['timestamp'] == self.t - self.window_size]['clicks'].values
            oldest_impressions = self.h_clicks_impressions[self.h_clicks_impressions['timestamp'] == self.t - self.window_size]['impressions'].values

            oldest_dot_prod_click = np.dot(oldest_clicks, oldest_embeddings)
            oldest_dot_prod_impr = np.dot(oldest_impressions, oldest_embeddings)

            self.b_click -= oldest_dot_prod_click
            self.b_impr -= oldest_dot_prod_impr
            

    def initial_round(
            self, run: int, iteration: int,
    ):
        # Check if there are new arms (= new publishers in the list)
        for publisher in self.publisher_list:
            # if not self.check_publisher_exist(publisher):
            #     self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(publisher=publisher, run=run, iteration=iteration)

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
        
        publisher_names = results['publisher'].unique()
        return [
            publisher
            for publisher in self.publisher_list
            if publisher.name in publisher_names
        ]
