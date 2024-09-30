from typing import List
import numpy as np
from Publisher import Publisher


class LinUCB:
    """ LinUCB algorithm """
    def __init__(self, alpha: float, d: int, publisher_list: List[str]):
        self.alpha = alpha
        # Embedding size
        self.d = d
        self.publisher_list = publisher_list
        self.n_arms = len(publisher_list)
        self.A = np.array([np.eye(d) for _ in range(self.n_arms)])
        self.b = np.array([np.zeros(d) for _ in range(self.n_arms)])
        self.theta = np.array([np.zeros(d) for _ in range(self.n_arms)])
        self.p = np.zeros(self.n_arms)

    def add_new_arm(self, publisher: Publisher):
        self.n_arms += 1
        self.publisher_list.append(publisher.name)
        # Initialize the new arm parameters
        self.A = np.append(self.A, np.eye(self.d))
        self.b = np.append(self.b, np.zeros(self.d))

    def update_arm(self, arm_index: int, context: np.ndarray):
        self.theta[arm_index] = np.linalg.inv(self.A[arm_index]).dot(self.b[arm_index])
        self.p[arm_index] = (self.theta[arm_index].dot(context) +
                             self.alpha * np.sqrt(context.dot(np.linalg.inv(self.A[arm_index]).dot(context))))

    def round_iteration(self, publisher_list: List[Publisher]):
        # Check if there are new arms (= new publishers in the list)
        for publisher in publisher_list:
            if publisher.name not in self.publisher_list:
                self.add_new_arm(publisher)
            # Update arms parameters
            pub_index = self.publisher_list.index(publisher.name)
            # The context is in the publisher object
            self.update_arm(pub_index, publisher.embedding)
        # Select the arm
        selected_arm = np.argmax(self.p)
        # Return the publisher name
        return self.publisher_list[selected_arm]

    def round_iteration_k(self, publisher_list: List[Publisher]):
        # Check if there are new arms (= new publishers in the list)
        for publisher in publisher_list:
            if publisher.name not in self.publisher_list:
                self.add_new_arm(publisher)
            # Update arms parameters
            pub_index = self.publisher_list.index(publisher.name)
            # The context is in the publisher object
            self.update_arm(pub_index, publisher.embedding)
        # Select the arm
        current_publishers = [publisher.name for publisher in publisher_list]
        selected_arm = np.argmax([self.p[i] for i in range(self.n_arms) if self.publisher_list[i] in current_publishers])
        # Return the publisher name
        return current_publishers[selected_arm]


    def update(self, context: np.ndarray, publisher: Publisher, reward: float | int):
        # Method to update the parameters of the selected arm
        arm = self.publisher_list.index(publisher.name)
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
