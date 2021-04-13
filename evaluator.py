#! /usr/bin/python3

from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Union, List

import numpy as np
from scipy.stats import norm

from dgp import DGP


NORMAL_QUANTILE = norm.ppf(0.975)


class Evaluator(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:
        pass


class ATEError(Evaluator):
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:
        return ATE - ATEhat.estimate


class ITEBias(Evaluator):
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:
        return np.average(ITE - ITEhat.estimate)


class ITEMSE(Evaluator):
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:
        return np.average(np.power(ITE - ITEhat.estimate, 2))


class CovariateMSE(Evaluator):
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:
        X1 = np.average(X[A==1, :], 0)
        X0 = np.average(X[A==0, :], 0)
        return np.mean(np.power(X1 - X0, 2)).item()


class ATECovers(Evaluator):
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:
        lwr = ATEhat.estimate - NORMAL_QUANTILE * ATEhat.std_error
        upr = ATEhat.estimate + NORMAL_QUANTILE * ATEhat.std_error
        return (ATE >= lwr) & (ATE <= upr)


class CISize(Evaluator):
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:
        return 2 * NORMAL_QUANTILE * ATEhat.std_error
