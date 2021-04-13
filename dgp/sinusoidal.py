#! /usr/bin/python3

from .dgp import DGP

import numpy as np


class SinusoidalDGP(DGP):
    def __init__(self, N=100, pate=1, ice_sd=0, noise_mu=0, noise_sd=0.1, num_covariates=4):
        tau = np.random.normal(size=N, loc=pate, scale=ice_sd)
        x_sds = [1] * num_covariates
        self._X = np.random.normal(size=(N, num_covariates)) @ np.diag(x_sds)
        beta = np.random.uniform(size=(num_covariates, 1))
        self.y0 = np.sin((self._X @ beta).reshape(-1)) + np.random.normal(
            size=N, loc=noise_mu, scale=noise_sd
        )
        self.y1 = self.y0 + tau + np.random.normal(
            size=N, loc=noise_mu, scale=noise_sd
        )
        super(SinusoidalDGP, self).__init__(N=N)

    def Y(self, A: np.ndarray) -> np.ndarray:
        return np.where(np.array(A).flatten() == 1, self.y1, self.y0)

    @property
    def X(self) -> np.ndarray:
        return self._X


class SinusoidalFactory(object):
    def __init__(self, N):
        self.N = N

    def create_dgp(self):
        return SinusoidalDGP(N=self.N)
