#! /usr/bin/python3
from sklearn.neighbors import DistanceMetric
import numpy as np

from .nbpMatch import nbpwrap
from .design import Design


class OptBlock(Design):
    def __init__(self, treatment_prob: float = 0.5):
        self.treatment_prob = treatment_prob
        super(OptBlock, self).__init__()

    def fit(self, X: np.ndarray, distance="mahalanobis") -> None:
        N = X.shape[0]
        if N % 2 == 1:
            idx_ignore = np.random.choice(N, 1).item()
        else:
            idx_ignore = None
        self.X = X


        if distance == "mahalanobis":
            inv_cov = np.linalg.pinv(np.cov(X, rowvar=False))
            dist_maker = DistanceMetric.get_metric("mahalanobis", VI=inv_cov)
        elif distance == "euclidean":
            dist_maker = DistanceMetric.get_metric("euclidean")
        else:
            raise NotImplementedError(
                "Only Mahalanobis and Euclidean distance are implemented."
            )
        distances = dist_maker.pairwise(X)
        if idx_ignore is not None:
            dist_ignore = distances[idx_ignore, :]
            dist_ignore[idx_ignore] = np.inf
            idx_nn = np.argmin(dist_ignore)
            distances = np.delete(np.delete(distances, idx_ignore, 0), idx_ignore, 1)
        n_to_pass = N if idx_ignore is None else N - 1
        self.matches = nbpwrap(wt=distances.T.reshape(-1), n=n_to_pass)
        # nbpwrap indexes from 1.
        self.matches = self.matches - 1
        blocks = {tuple(sorted(x)) for x in enumerate(self.matches)}
        self.blocks = [list(block) for block in blocks]

        self.block_membership = np.array([-1] * N)
        for block_id, block in enumerate(blocks):
            for member_idx, member in enumerate(block):
                if idx_ignore is not None:
                    if member == idx_nn:
                        self.blocks[block_id].append(idx_ignore)
                        self.block_membership[idx_ignore] = block_id
                    if member >= idx_ignore:
                        self.block_membership[member+1] = block_id
                        self.blocks[block_id][member_idx] = member + 1
                else:
                    self.block_membership[member] = block_id

    def assign(self, X: np.ndarray) -> np.ndarray:
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
            En_trt = M * self.treatment_prob
            n_trt = int(max(1, np.floor(En_trt)))
            n_ctl = int(max(1, np.floor(M - En_trt)))
            n_extra = int(np.floor(M - n_trt - n_ctl))
            a_extra = int(np.random.choice([0, 1], 1).item())
            n_trt += a_extra * n_extra
            trted = np.random.choice(M, n_trt, replace=False)
            for unit in trted:
                A[block[unit]] = 1
        return A
