from typing import List
import numpy as np
import pandas as pd

from Publisher import Publisher
from Publisher_Reward import PublisherReward
from KnapsackSolver import get_data, solver


class CombinatorialLinUCB:
    """ LinUCB algorithm """
    def __init__(self, alpha: float, d: int, publisher_list: List[Publisher]):
        self.alpha = alpha
        # Embedding size
        self.d = d
        self.publisher_list = publisher_list
        self.n_arms = len(publisher_list)
        # self.A = np.array([np.eye(d) for _ in range(self.n_arms)])
        self.A = {
            publisher.name: np.eye(d) for publisher in publisher_list
        }
        # self.b = np.array([np.zeros(d) for _ in range(self.n_arms)])
        self.b = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        # self.theta = np.array([np.zeros(d) for _ in range(self.n_arms)])
        self.theta = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        # self.p = np.zeros(self.n_arms)
        self.p = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Saved dataframe with current iteration stats
        self.iteration_stats = None
        # Dataframe with LinUCB parameters per iteration
        self.linucb_params = pd.DataFrame(columns=['Iteration', 'publisher', 'exp_rew', 'p'])

    def add_new_arm(self, publisher: Publisher):
        self.n_arms += 1
        self.publisher_list.append(publisher.name)
        # Initialize the new arm parameters
        self.A[publisher.name] = np.eye(self.d)
        self.b[publisher.name] = np.zeros(self.d)

    def update_arm(self, publisher: Publisher, iteration: int):
        self.theta[publisher.name] = np.linalg.inv(self.A[publisher.name]).dot(self.b[publisher.name])
        self.p[publisher.name] = (self.theta[publisher.name].dot(publisher.embedding) +
                             self.alpha * np.sqrt(publisher.embedding.dot(np.linalg.inv(self.A[publisher.name]).dot(publisher.embedding))))
        # # Save the parameters
        # self.linucb_params = pd.concat([
        #     self.linucb_params,
        #     pd.DataFrame({
        #         'Iteration': iteration,
        #         'publisher': publisher.name,
        #         'exp_rew': self.theta[publisher.name].dot(publisher.embedding),
        #         'p': self.p[publisher.name]
        #     }, index=[0])
        # ], ignore_index=True)

    def round_iteration(
            self, publisher_list: List[Publisher], iteration: int, soglia_clicks: float = None,
            soglia_spent: float = None, soglia_cpc: float = None, soglia_num_publisher: int = None, soglia_ctr: float = None) -> List[Publisher]:
        # Check if there are new arms (= new publishers in the list)
        for publisher in publisher_list:
            if not self.check_publisher_exist(publisher):
                self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(publisher=publisher, iteration=iteration)
        # Select the super-arm
        # il parametro publisher_list non viene passato al solver perché i dati necessari sono già presenti nel dataframe iteration_stats
        super_arm = self.knapsack_solver(
            soglia_spent=soglia_spent,
            soglia_clicks=soglia_clicks,
            soglia_cpc=soglia_cpc,
            soglia_num_publisher=soglia_num_publisher,
            soglia_ctr=soglia_ctr)
        # Return the super-arm
        return super_arm

    def update(self, publisher_name: str, publisher_embedding: np.array, reward: float | int, iteration: int):
        # Method to update the parameters of the selected arm
        self.A[publisher_name] += np.outer(publisher_embedding, publisher_embedding)
        self.b[publisher_name] += reward * publisher_embedding
        # Save the parameters
        if not self.linucb_params.empty:
            self.linucb_params = pd.concat([
                self.linucb_params,
                pd.DataFrame({
                    'Iteration': iteration,
                    'publisher': publisher_name,
                    'exp_rew': self.theta[publisher_name].dot(publisher_embedding),
                    'p': self.p[publisher_name]
                }, index=[0])
            ], ignore_index=True)
        else:
            self.linucb_params = pd.DataFrame({
                'Iteration': iteration,
                'publisher': publisher_name,
                'exp_rew': self.theta[publisher_name].dot(publisher_embedding),
                'p': self.p[publisher_name]
            }, index=[0])

    # def initial_round(self, publisher_rewards_list: List[PublisherReward], iteration: int):
    #     """
    #     Method to initialize the parameters of the arms without choosing an arm
    #     """
    #     for publisher_reward in publisher_rewards_list:
    #         if not self.check_publisher_exist(publisher_reward.publisher):
    #             self.add_new_arm(publisher_reward.publisher)
    #         self.update(publisher_reward.publisher.name, publisher_reward.publisher.embedding, publisher_reward.reward, iteration)

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

    def get_p(self, publisher_list: List[Publisher]):
        return {
            pub.name: self.p[pub.name]
            for pub in publisher_list
        }

    def save_stats(self, iteration_stats: pd.DataFrame):
        # Save the stats of the current iteration keeping the last one
        concat_df = pd.concat([self.iteration_stats, iteration_stats], ignore_index=True)
        self.iteration_stats = concat_df.drop_duplicates(subset=['publisher'], keep='last')

    def knapsack_solver(
            self, soglia_clicks: float = None, soglia_spent: float = None, soglia_cpc: float = None, soglia_num_publisher: int = None, soglia_ctr: float = None
    ) -> List[Publisher]:
        # Add the UCBs to the dataframe
        self.iteration_stats['rew_ucb'] = self.iteration_stats['publisher'].apply(lambda x: self.p[x])
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
