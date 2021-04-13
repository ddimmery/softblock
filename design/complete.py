import numpy as np

from .design import Design


class Complete(Design):
    def __init__(self) -> None:
        self.X = None

    def fit(self, X: np.ndarray) -> None:
        pass

    def assign(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        n2 = int(n / 2)
        treatment = np.array([0] * n2 + [1] * n2)
        if n != 2 * n2:
            treatment += np.random.choice([0, 1], 1).tolist()
        return np.random.choice(treatment, size=n, replace=False)
