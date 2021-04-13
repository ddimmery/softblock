#! /usr/bin/python3

from typing import Optional
import numpy as np
import pandas as pd

from dgp import DGP

from scipy import stats
class IHDPDGP(DGP):
    def __init__(
        self, df: pd.DataFrame, w_val: float=0.5,
        tau: float=4.0, sigma_y: float=1.0, setting: str = 'A',
        sd_ite: float=1.0
    ) -> None:
        covs = [
            "bw", "b.head", "preterm", "birth.o", "nnhealth", "momage",
            "sex", "twin", "b.marr", "mom.lths", "mom.hs", "mom.scoll", "cig",
            "first", "booze", "drugs", "work.dur", "prenatal", "ark", "ein",
            "har", "mia", "pen", "tex", "was"
        ]
        ok_rows = np.logical_or(df["momwhite"] != 0, df["treat"] != 1)
        self._A = df.loc[ok_rows, "treat"].values
        self._X = df.loc[ok_rows, covs].values
        for col in range(self._X.shape[1]):
            if len(np.unique(self._X[:, col])) <= 2:
                next
            self._X[:, col] = (
                (self._X[:, col] - self._X[:, col].mean()) /
                self._X[:, col].std()
            )
        self._X = np.hstack((np.ones((self._X.shape[0], 1)), self._X))
        super(IHDPDGP, self).__init__(N=self._X.shape[0])

        if setting == "A":
            W = np.zeros(self._X.shape)
            beta = np.random.choice([0, 1, 2, 3, 4], self._X.shape[1], replace=True, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        elif setting == "B":
            W = np.ones(self._X.shape) * w_val
            beta = np.random.choice([0, 0.1, 0.2, 0.3, 0.4], self._X.shape[1], replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])

        self.mu0 = (self._X + W) @ beta
        self.mu1 = self._X @ beta
        if setting == "B":
            self.mu0 = np.exp(self.mu0)
        adjustment = np.average(self.mu1 - self.mu0) - tau
        self.mu1 -= adjustment
        self.init_sd_ite = np.std(self.mu1 - self.mu0)
        self.mu0 = np.average(self.mu0) + (self.mu0 - np.average(self.mu0)) #/ self.init_sd_ite * sd_ite
        self.mu1 = np.average(self.mu1) + (self.mu1 - np.average(self.mu1)) #/ self.init_sd_ite * sd_ite
        self._Y0 = np.random.normal(size=self.n, loc=self.mu0, scale=sigma_y)
        self._Y1 = np.random.normal(size=self.n, loc=self.mu1, scale=sigma_y)
        self._Y = np.where(self._A == 1, self._Y1, self._Y0)

    @property
    def A(self) -> np.ndarray:
        return self._A

    def Y(self, A: Optional[np.ndarray] = None) -> np.ndarray:
        if A is None:
            return self._Y
        return np.where(A == 1, self._Y1, self._Y0)

    @property
    def X(self) -> np.ndarray:
        return self._X[:, 1:]

    def ITE(self) -> np.ndarray:
        return self.mu1 - self.mu0


class IHDPFactory(object):
    def __init__(self, csv_path: str, setting: str = 'B', sd_ite: float=1.0):
        self.setting = setting
        self.df = pd.read_csv(csv_path)
        self.sd_ite = sd_ite

    def create_dgp(self):
        return IHDPDGP(df=self.df.copy(), setting=self.setting, sd_ite=self.sd_ite)
