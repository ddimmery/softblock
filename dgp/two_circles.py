#! /usr/bin/python3

from .dgp import DGP

import numpy as np


class TwoCircles(DGP):
    def __init__(self, N=100, pate=1, ice_sd=0, noise_sd=0.05):
        N1 = int(N / 2)
        N2 = N - N1
        tau = np.random.normal(size=N, loc=pate, scale=ice_sd)
        phi_1 = np.random.uniform(size=N1) * 2 * np.pi
        r_1 = np.random.normal(size=N1, loc=1, scale=0.1)
        x_1 = np.cos(phi_1) * r_1
        y_1 = np.sin(phi_1) * r_1
        phi_2 =np.random.uniform(size=N2) * 2 * np.pi
        r_2 = np.random.normal(size=N2, loc=2, scale=0.1)
        x_2 = np.cos(phi_2) * r_2
        y_2 = np.sin(phi_2) * r_2
        x = np.hstack((x_1, x_2)).reshape(-1, 1)
        y = np.hstack((y_1, y_2)).reshape(-1, 1)
        self._X = np.hstack((x, y))
        e0 = np.random.normal(size=N, scale=noise_sd)
        beta = np.random.uniform(size=(2, 1))
        self.y0 = np.concatenate((beta[0] * phi_1 + beta[1] * r_1, beta[0] * phi_2 + beta[1] * r_2))
        e1 = np.random.normal(size=N, scale=noise_sd)
        self.y1 = self.y0 + tau
        self.y0 += e0
        self.y1 += e1
        super(TwoCircles, self).__init__(N=N)

    def Y(self, A: np.ndarray) -> np.ndarray:
        return np.where(np.array(A).flatten() == 1, self.y1, self.y0)

    @property
    def X(self) -> np.ndarray:
        return self._X


class TwoCirclesFactory(object):
    def __init__(self, N):
        self.N = N

    def create_dgp(self):
        return TwoCircles(N=self.N)
