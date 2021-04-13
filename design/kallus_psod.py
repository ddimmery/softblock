#! /usr/bin/python3
import numpy as np

from design.design import Design
import design.kallus_utils as util


class PSOD(Design):
    def __init__(self, kernel='gaus', kernel_kwargs=None):
        super(PSOD, self).__init__()
        if kernel == 'lin':
            self.kernel = util.LinearKernel
        elif kernel == 'quad':
            self.kernel = util.PolynomialKernel
        elif kernel == 'gaus':
            self.kernel = util.GaussianKernel
        elif kernel == 'exp':
            self.kernel = util.ExpKernel
        self.kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}

    def fit(self, X: np.ndarray) -> None:
        self.X = X
        N = self.X.shape[0]
        K = self.kernel(X, normalize=False, **self.kernel_kwargs)
        self.solution = util.PSOD(K)

    def assign(self, X: np.ndarray):
        return (np.array(util.PSODDraw(self.solution))>0).astype(int)
