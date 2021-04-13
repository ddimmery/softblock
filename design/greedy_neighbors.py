import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, laplacian
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from .design import Design
from .soft_block import greedy_cut


class GreedyNeighbors(Design):
    def __init__(self) -> None:
        self.X = None

    def fit(self, X: np.ndarray) -> None:
        # standardize and get distances / similarities
        X_std = (X - X.mean(0)) / X.std(0)
        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=30)
        knn.fit(X_std)
        self.A = knn.kneighbors_graph(mode='distance', n_neighbors=1)

    def assign(self, X: np.ndarray) -> np.ndarray:
        sparse_forest = self.A + self.A.T
        self.L = laplacian(sparse_forest)
        # get the max-cut of the tree via greedy selection
        treatments = greedy_cut(sparse_forest)
        # cuts -> assignments and return
        return treatments.reshape(-1)
