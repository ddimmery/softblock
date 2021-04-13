#! /usr/bin/python3
import numpy as np

from design.design import Design
import design.kallus_utils as util


class ReRandomization(Design):
    def __init__(self, acceptance_probability=0.01):
        super(ReRandomization, self).__init__()
        self.pr_accept = acceptance_probability
        self.num_assignments = 100

    def fit(self, X: np.ndarray) -> None:
        self.X = X
        N = self.X.shape[0]
        nrand = int(self.num_assignments / self.pr_accept)
        inv_cov = np.linalg.pinv(np.cov(X, rowvar=False))
        n2 = int(N / 2)
        A = np.array([-1] * n2 + [1] * (N - n2))
        l = []
        for i in range(nrand):
            np.random.shuffle(A)
            y = np.dot(A, X) / n2
            l.append((np.dot(np.dot(y.T, inv_cov), y), A))
        l.sort(key=lambda z: z[0])
        self.draws = [(np.array(a[1]) + 1 / 2).astype(int) for a in l[:self.num_assignments]]

    def assign(self, X: np.ndarray):
        idx = np.random.choice(range(self.num_assignments), 1).item()
        flip = np.random.binomial(1, 0.5, 1)
        return flip * (1 - self.draws[idx]) + (1 - flip) * self.draws[idx]
