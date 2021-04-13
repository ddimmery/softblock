#! /usr/bin/python3

from abc import ABCMeta, abstractmethod

import numpy as np

class DGP(metaclass=ABCMeta):
    def __init__(self, N: int) -> None:
        self.n = N

    @abstractmethod
    def Y(self, A: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    def ATE(self) -> np.ndarray:
        return np.average(self.Y([1] * self.n) - self.Y([0] * self.n))

    def ITE(self) -> np.ndarray:
        return self.Y([1] * self.n) - self.Y([0] * self.n)
