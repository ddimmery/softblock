import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import pinv
from scipy import sparse
from design import Design


import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


class MatchedPair(Design):
    def fit(self, X):
        self.X = X
        s = pinv(np.cov(X,rowvar=0))
        n = len(X)
        n2 = n/2
        D = squareform(pdist(X, 'mahalanobis', VI = s))
        self.blocks = [
            [a, b]
            for a, b in nx.matching.max_weight_matching(nx.Graph(-D), True)
        ]
        self.L = np.zeros(D.shape)
        for left, right in self.blocks:
            self.L[left, right] = 1
            self.L[right, left] = 1
        self.L = sparse.csr_matrix(self.L)


    def assign(self, X):
        if X is None:
            X = self.X
        elif X is self.X:
            pass
        else:
            raise ValueError("Can't go out of sample here.")
        N = X.shape[0]
        A = np.array([0] * N)
        for block in self.blocks:
            M = len(block)
            En_trt = M / 2
            n_trt = int(max(1, np.floor(En_trt)))
            n_ctl = int(max(1, np.floor(M - En_trt)))
            n_extra = int(np.floor(M - n_trt - n_ctl))
            a_extra = int(np.random.choice([0, 1], 1)[0])
            n_trt += a_extra * n_extra
            trted = np.random.choice(M, n_trt, replace=False)
            for unit in trted:
                A[block[unit]] = 1
        return A
