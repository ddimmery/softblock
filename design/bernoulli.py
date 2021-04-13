import numpy as np

from .design import Design


class Bernoulli(Design):
    def __init__(self) -> None:
        self.X = None

    def fit(self, X: np.ndarray) -> None:
        pass

    def assign(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        treatment = np.random.binomial(n=1, p=0.5, size=n)
        if sum(treatment) == 0 or sum(treatment) == n:
            return self.assign(X)
        return treatment
