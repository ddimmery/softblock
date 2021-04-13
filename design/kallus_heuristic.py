#! /usr/bin/python3
import numpy as np

from design.design import Design
import design.kallus_utils as util


class Heuristic(Design):
    def __init__(self, kernel='gaus', kernel_kwargs=None):
        super(Heuristic, self).__init__()
        if kernel == 'gaus':
            self.kernel = util.GaussianKernel
        elif kernel == 'exp':
            self.kernel = util.ExpKernel
        else:
            NotImplementedError("Unknown kernel")
        self.num_solutions = 100
        self.kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}

    def fit(self, X: np.ndarray, distance="mahalanobis") -> None:
        self.X = X
        N = self.X.shape[0]
        K = self.kernel(X, normalize=False, **self.kernel_kwargs)
        self.solutions = util.MSODHeuristic(K, self.num_solutions)

    def assign(self, X):
        return (np.array(util.MSODDraw(self.solutions))>0).astype(int)
