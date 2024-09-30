import pickle
import numpy as np


class Publisher:
    def __init__(self, name, embedding, num_auctions):
        self.name = name
        # Read from memory publisher embedding
        self.embedding = embedding
        # self.embedding = np.concatenate((self.rng.normal(0, 1, size=5), [1.0]))
        # Number of auctions to simulate according to past data
        self.num_auctions = num_auctions
        # Metrics of revenue
        self.revenue = .0

    def generate_user_context(self):
        # Definisci l'intensità del rumore
        noise_strength = 0.01
        # Genera il rumore gaussiano
        noise = np.random.normal(0, noise_strength, self.embedding.shape)
        # Aggiungi il rumore agli embedding originali
        noisy_embeddings = self.embedding + noise
        return noisy_embeddings