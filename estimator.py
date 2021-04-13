#! /usr/bin/python3

from abc import ABCMeta, abstractmethod
from typing import NamedTuple

from design import Design

import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import dia_matrix

import statsmodels.api as sm


class Estimate(NamedTuple):
    estimate: np.ndarray
    std_error: np.ndarray


class Estimator(metaclass=ABCMeta):
    def __init__(self, design: Design) -> None:
        self.design = design

    @abstractmethod
    def ATE(self, X, A, YA) -> Estimate:
        pass

    @abstractmethod
    def ITE(self, X, A, YA) -> Estimate:
        pass


class LaplacianNorm(Estimator):
    def __init__(self, design: Design) -> None:
        super(LaplacianNorm, self).__init__(design)
        self.L = self.design.L

    def ATE(self, X, A, YA):
        n = A.shape[0]
        # estimate regression method
        X = np.hstack((A.reshape((-1, 1)), np.ones((n, 1))))
        xlx = X.T @ self.L @ X
        bread = np.linalg.pinv(xlx)
        coefs = bread @ X.T @ self.L @ YA.reshape(-1, 1)
        r = dia_matrix((np.power(YA - np.ravel(X @ coefs), 2), 0), shape=(n,n))
        meat = X.T @ (self.L @ r @ self.L) @ X
        vcv = bread @ meat @ bread
        return Estimate(estimate=coefs[0].item(), std_error=np.sqrt(vcv[0, 0]))

    def ITE(self, X, A, YA):
        N = A.shape[0]
        Ldiag = self.L.diagonal()
        Dinv = dia_matrix(((2 * A - 1) / Ldiag, 0), shape=(N, N))
        est =  Dinv @ self.L @ YA.reshape(-1, 1)
        return Estimate(estimate=est.flatten(), std_error=[np.inf] * est.shape[0])


class DifferenceInMeans(Estimator):
    def _diff_in_means(self, Y, A):
        return np.average(Y[A == 1]) - np.average(Y[A == 0])

    def _var_for_diff_in_means(self, Y, A):
        return np.var(Y[A == 1]) / np.sum(A) + np.var(Y[A == 0]) / np.sum(1 - A)

    def ATE(
        self, X, A, YA
    ) -> Estimate:
        return Estimate(
            estimate=self._diff_in_means(YA, A),
            std_error=np.sqrt(self._var_for_diff_in_means(YA, A))
        )


class Blocking(DifferenceInMeans):
    def ATE(self, X, A, YA) -> Estimate:
        cates = []
        ns = []
        vars = []
        overall_var = np.var(YA[A==1]) + np.var(YA[A==0])
        for block in self.design.blocks:
            ns.append(len(block))
            cates.append(self._diff_in_means(YA[block], A[block]))
            block_var = self._var_for_diff_in_means(YA[block], A[block])
            vars.append(block_var if block_var > 0 else overall_var)
        ns = np.array(ns) / np.sum(ns)
        return Estimate(
            estimate=np.average(cates, weights=ns),
            std_error=np.sqrt(np.average(vars, weights=np.power(ns, 2))),
        )

    def ITE(self, X, A, YA) -> Estimate:
        ites = np.array([np.inf] * A.shape[0])
        ses = np.array([np.inf] * A.shape[0])
        for block in self.design.blocks:
            ites[block] = self._diff_in_means(YA[block], A[block])
            ses[block] = np.sqrt(self._var_for_diff_in_means(YA[block], A[block]))
        return Estimate(
            estimate=ites,
            std_error=ses
        )


class MatchedPairBlocking(DifferenceInMeans):
    def ATE(self, X, A, YA) -> Estimate:
        cates = []
        for block in self.design.blocks:
            cates.append(self._diff_in_means(YA[block], A[block]))
        return Estimate(
            estimate=np.average(cates),
            std_error=np.sqrt(np.var(cates) / len(self.design.blocks)),
        )

    def ITE(self, X, A, YA) -> Estimate:
        ites = np.array([np.inf] * A.shape[0])
        ses = np.array([np.inf] * A.shape[0])
        overall_var = np.var(YA[A==1]) + np.var(YA[A==0])
        for block in self.design.blocks:
            ites[block] = self._diff_in_means(YA[block], A[block])
            ses[block] = np.sqrt(overall_var)
        return Estimate(
            estimate=ites,
            std_error=ses
        )

class CovariateAdjustedMean(Estimator):
    def ATE(self, X, A, YA) -> Estimate:
        A = A.reshape(-1, 1)
        X = preprocessing.scale(X)
        XA = A * X
        X = sm.add_constant(np.hstack((A, X, XA)))
        model = sm.OLS(YA, X)
        results = model.fit(cov_type='HC0')
        return Estimate(
            estimate=results.params[1],
            std_error=results.bse[1]
        )


class KNNT(Estimator):
    def ITE(self, X, A, YA) -> Estimate:
        k_to_try = int(np.log(A.shape[0]))
        min_units = min(sum(A), sum(1-A))
        k = 5 if min_units >= 5 else min_units
        knn0 = KNeighborsRegressor(weights='distance', n_neighbors=k, algorithm='kd_tree')
        knn1 = KNeighborsRegressor(weights='distance', n_neighbors=k, algorithm='kd_tree')
        knn0.fit(X[A==0, :], YA[A==0])
        knn1.fit(X[A==1, :], YA[A==1])
        y0 = knn0.predict(X)
        y1 = knn1.predict(X)
        return Estimate(
            estimate=y1 - y0,
            std_error=[0] * X.shape[0]
        )


class RFT(Estimator):
    def ITE(self, X, A, YA) -> Estimate:
        n_trees = 20 * int(np.power(A.shape[0], 0.25))
        rf0 = RandomForestRegressor(n_estimators=n_trees, max_depth=8)
        rf1 = RandomForestRegressor(n_estimators=n_trees, max_depth=8)
        if sum(A==0) == 0 or sum(A==0) == X.shape[0]:
            print(type(self.design), sum(A), sum(1-A))
        rf0.fit(X[A==0, :], YA[A==0])
        rf1.fit(X[A==1, :], YA[A==1])
        y0 = rf0.predict(X)
        y1 = rf1.predict(X)
        return Estimate(
            estimate=y1 - y0,
            std_error=[0] * X.shape[0]
        )


class RFS(Estimator):
    def ITE(self, X, A, YA) -> Estimate:
        n_trees = 20 * int(np.power(A.shape[0], 0.25))
        rf = RandomForestRegressor(n_estimators=n_trees, max_depth=8)
        rf.fit(np.hstack(A, X), YA)
        y0 = rf0.predict(np.hstack(([0] * X.shape[0], X)))
        y1 = rf0.predict(np.hstack(([1] * X.shape[0], X)))
        return Estimate(
            estimate=y1 - y0,
            std_error=[np.inf] * est.shape[0]
        )


class DMandKNNT(DifferenceInMeans, KNNT):
    pass


class DMandRFS(DifferenceInMeans, RFS):
    pass


class DMandRFT(DifferenceInMeans, RFT):
    pass


class OLSandRFT(CovariateAdjustedMean, RFT):
    pass


class OLSandKNNT(CovariateAdjustedMean, KNNT):
    pass


class BlockingRF(RFT, Blocking):
    pass

class LaplacianRF(RFT, LaplacianNorm):
    pass
