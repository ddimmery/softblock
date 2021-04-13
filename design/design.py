#! /usr/bin/python3

from abc import ABCMeta, abstractmethod

import numpy as np


class Design(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def assign(self, X: np.ndarray) -> np.ndarray:
        pass
