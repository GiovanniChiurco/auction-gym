from typing import List

import numpy as np
import pandas as pd
import scipy

from Publisher import Publisher


class SpentPredictor:
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
        # Parametri per stimare la spesa
        self.b_spent = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        self.theta_spent = {
            publisher.name: np.zeros(d) for publisher in publisher_list
        }
        self.est_spent = {
            publisher.name: 0 for publisher in publisher_list
        }
        self.linucb_params = None

    def add_new_arm(self, publisher: Publisher):
        self.n_arms += 1
        self.publisher_list.append(publisher)
        self.A[publisher.name] = np.eye(self.d)
        # Initialize the new arm parameters
        self.b_spent[publisher.name] = np.zeros(self.d)
        self.theta_spent[publisher.name] = np.zeros(self.d)
        self.est_spent[publisher.name] = 0
        self.conf_bound[publisher.name] = 0

    def update_arm(self, publisher: Publisher, run: int, iteration: int):
        # Matrix inversion with Cholesky decomposition
        # Calcola la fattorizzazione di Cholesky di A (la parte triangolare inferiore di A)
        L, lower = scipy.linalg.cho_factor(self.A[publisher.name], lower=True)
        embedding = np.append(publisher.embedding, 1)
        # Aggiorna il confine di confidenza usando la fattorizzazione di Cholesky
        # Risolvi Lx = embedding (forward substitution) e quindi L^Ty = x (back substitution)
        x = scipy.linalg.cho_solve((L, lower), embedding)
        self.conf_bound[publisher.name] = self.alpha * np.sqrt(embedding.dot(x))
        # Aggiorna i parametri spent e stima usando la fattorizzazione di Cholesky
        self.theta_spent[publisher.name] = scipy.linalg.cho_solve((L, lower), self.b_spent[publisher.name])
        self.est_spent[publisher.name] = np.dot(self.theta_spent[publisher.name], embedding)
        # Save the parameters
        if self.linucb_params is None:
            self.linucb_params = pd.DataFrame({
                'Iteration': iteration,
                'Run': run,
                'publisher': publisher.name,
                'est_spents': self.est_spent[publisher.name],
                'conf_bound': self.conf_bound[publisher.name]
            }, index=[0])
        else:
            self.linucb_params = pd.concat([
                self.linucb_params,
                pd.DataFrame({
                    'Iteration': iteration,
                    'Run': run,
                    'publisher': publisher.name,
                    'est_spents': self.est_spent[publisher.name],
                    'conf_bound': self.conf_bound[publisher.name]
                }, index=[0])
            ], ignore_index=True)

    def round_iteration(
            self, curr_publisher_list: List[Publisher], run: int, iteration: int):
        # Check if there are new arms (= new publishers in the list)
        for publisher in curr_publisher_list:
            if not self.check_publisher_exist(publisher):
                self.add_new_arm(publisher)
            # Update arms parameters
            self.update_arm(publisher=publisher, run=run, iteration=iteration)

    def update(self, publisher_name: str, publisher_embedding: np.array, spent: float | int):
        embedding = np.append(publisher_embedding, 1)
        # Method to update the parameters of the selected arm
        self.A[publisher_name] += np.outer(embedding, embedding)
        # Update spent parameters
        self.b_spent[publisher_name] += spent * embedding

    def check_publisher_exist(self, publisher: Publisher):
        if publisher.name in self.A.keys():
            return True
        else:
            return False