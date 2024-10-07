from typing import List
import numpy as np
import pandas as pd

from Publisher import Publisher


class KLinUCB:
    """ k-LinUCB algorithm """
    def __init__(self, alpha: float, d: int, publisher_list: List[Publisher], k: int):
        self.alpha = alpha
        # Embedding size
        self.d = d
        self.publisher_list = publisher_list
        self.n_arms = len(publisher_list)
        self.A = {
            publisher.name: np.eye(d) for publisher in publisher_list
        }
        self.b = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        self.theta = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        self.p = {
            publisher.name: 0 for publisher in publisher_list
        }
        # Saved dataframe with current iteration stats
        self.iteration_stats = None
        # Dataframe with LinUCB parameters per iteration
        self.linucb_params = pd.DataFrame(columns=['Iteration', 'publisher', 'exp_rew', 'p'])
        # Number of arms to select at each round
        self.k = k

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

    def round_iteration(
            self, publisher_list: List[Publisher], iteration: int,
    ) -> List[Publisher]:
        # Check if there are new arms (= new publishers in the list)
        for publisher in publisher_list:
            if not self.check_publisher_exist(publisher):
                self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(publisher=publisher, iteration=iteration)
        # Select the super-arm
        sel_arms = self.select_super_arm()
        return sel_arms

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

    def select_super_arm(self) -> List[Publisher]:
        """
        Return the first k arms with the highest p value
        """
        sorted_p = sorted(self.p.items(), key=lambda x: x[1], reverse=True)
        super_arm = []
        for p in sorted_p[:self.k]:
            for pub in self.publisher_list:
                if pub.name == p[0]:
                    super_arm.append(pub)
        return super_arm
