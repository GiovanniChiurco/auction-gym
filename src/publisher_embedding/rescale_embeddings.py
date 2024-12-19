from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np


if __name__ == '__main__':
    pub_emb = pickle.load(open('data/embeddings_to_pick/sites_embeddings_red_70.pkl', 'rb'))

    pub_emb_matrix = np.array([pub_emb[k] for k in pub_emb.keys()])

    scaled_pub_emb_matrix = MinMaxScaler().fit_transform(pub_emb_matrix)
    scaled_pub_embeddings = {k: scaled_pub_emb_matrix[i] for i, k in enumerate(pub_emb.keys())}

    pickle.dump(scaled_pub_embeddings, open('data/embeddings_to_pick/sites_embeddings_red_70_scaled.pkl', 'wb'))
