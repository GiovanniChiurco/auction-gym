import numpy as np


class Publisher:
    def __init__(self, rng, name, mean, variance, num_auctions, embedding_size):
        self.rng = rng
        self.name = name
        # Gaussian parameters to generate user context features vectors
        self.mean = mean
        self.variance = variance
        # Number of auctions to simulate according to past data
        self.num_auctions = num_auctions
        # User context vector size
        self.embedding_size = embedding_size
        # Metrics of revenue
        self.revenue = .0

    def generate_user_context(self):
        user_context = np.concatenate((self.rng.normal(self.mean, self.variance, size=self.embedding_size), [1.0]))
        return user_context
