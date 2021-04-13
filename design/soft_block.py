import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, laplacian
from sklearn.neighbors import NearestNeighbors

from .design import Design


def greedy_cut(forest):
    N = forest.shape[0]
    root_id = np.random.choice(N, 1).item()
    a = np.random.choice([0, 1], 1).item()
    nodes = set()
    unvisited = set(range(N))
    colors = np.array([-1] * N)
    stack = [(root_id, a)]
    while stack or len(unvisited) > 0:
        if not stack:
            cur_node = unvisited.pop()
            color = np.random.choice([0,1], 1).item()
        else:
            cur_node, color = stack.pop()
            unvisited.remove(cur_node)
        nodes.add(cur_node)
        colors[cur_node] = color
        for child in forest.getrow(cur_node).indices:
            if child not in nodes:
                stack.append((child, 1 - color))
    return colors


def get_breadth_first_nodes(edges, root_id):
    N = edges.shape[0]
    nodes = set()
    colors = np.array([-1] * N)
    stack = [(root_id, np.random.choice([0,1], 1).item())]
    while stack:
        cur_node, color = stack.pop()
        nodes.add(cur_node)
        colors[cur_node] = color
        for child in edges.getrow(cur_node).indices:
            if child not in nodes:
                stack.append((child, 1 - color))
    return colors


class SoftBlock(Design):
    def __init__(self, num_neighbors=6, s2=2) -> None:
        self.X = None
        self.k = num_neighbors
        self.s2 = s2

    def fit(self, X: np.ndarray) -> None:
        # standardize and get distances / similarities
        X_std = (X - X.mean(0)) / X.std(0)
        if self.k is not None:
            knn = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree', leaf_size=30)
            knn.fit(X_std)
            A = knn.kneighbors_graph(mode='distance', n_neighbors=self.k)
            np.exp(-A.data / self.s2, out=A.data)
        else:
            dists = cdist(X_std, X_std)
            A = np.exp(-dists / self.s2)
        self.A = A

    def assign(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        # take the maximum spanning tree
        tree_sparse = minimum_spanning_tree(-1 * self.A)
        tree_sparse = tree_sparse + tree_sparse.T
        self.L = laplacian(tree_sparse)
        # get the max-cut of the tree via greedy selection
        result = greedy_cut(tree_sparse)
        # cuts -> assignments and return
        return result.reshape(-1)
