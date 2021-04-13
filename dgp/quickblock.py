#! /usr/bin/python3

from .dgp import DGP

import numpy as np


class QuickBlockDGP(DGP):
    def __init__(self, N=100, K=2, pate=1, scale=0, ite_sd=0):
        self._X = np.random.uniform(size=(N, K)) * 10
        e = np.random.normal(size=N)
        self.y = np.prod(self._X, axis=1) + e
        self.tau = np.random.normal(size=N, loc=pate, scale=ite_sd)
        super(QuickBlockDGP, self).__init__(N=N)

    def Y(self, A: np.ndarray) -> np.ndarray:
        return np.where(A==1, self.y + self.tau, self.y)

    @property
    def X(self) -> np.ndarray:
        return self._X


class QuickBlockFactory(object):
    def __init__(self, N, K=2):
        self.N = N
        self.K = K

    def create_dgp(self):
        return QuickBlockDGP(N=self.N, K=self.K)
