#! /usr/bin/python3

from abc import ABCMeta, abstractmethod
import time
from typing import Type

from design import Design
from estimator import Estimator
from evaluator import Evaluator

import numpy as np
import pandas as pd

class Plan(metaclass=ABCMeta):
    def __init__(self):
        self.evaluators = {}
        self.designs = {}

    def add_design(self, design_name, design_class: Type[Design], estimator_class: Type[Estimator], design_kwargs = None):
        self.designs[design_name] = (design_class, estimator_class, design_kwargs)

    def add_evaluator(self, evaluator_name: str, evaluator: Evaluator):
        self.evaluators[evaluator_name] = evaluator()

    def execute(self, dgp_factory, seed):
        np.random.seed(seed)
        dgp = dgp_factory.create_dgp()
        self.dgp = dgp
        X = dgp.X
        Y0 = dgp.Y([0] * dgp.n)
        Y1 = dgp.Y([1] * dgp.n)
        ITE = dgp.ITE()
        ATE = dgp.ATE()
        results = []
        for design_name, (design_class, estimator_class, design_kwargs) in self.designs.items():
            def make_row(name, value):
                return pd.DataFrame({"design": [design_name], "metric": [name], "value": [value]})
            time_start = time.time()
            if design_kwargs is None:
                design_kwargs = {}
            design = design_class(**design_kwargs)
            design.fit(X)
            A = design.assign(X)
            time_end = time.time()
            time_elapsed = time_end - time_start
            results.append(make_row("time_design", time_elapsed))
            YA = np.where(A==1, Y1, Y0)
            time_start = time.time()
            estimator = estimator_class(design)
            ITEhat = estimator.ITE(X, A, YA)
            ATEhat = estimator.ATE(X, A, YA)
            time_end = time.time()
            time_elapsed = time_end - time_start
            results.append(make_row("time_estimation", time_elapsed))
            for name, evaluator in self.evaluators.items():
                val = evaluator.evaluate(X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat)
                results.append(make_row(name, val))
        return pd.concat(results)
