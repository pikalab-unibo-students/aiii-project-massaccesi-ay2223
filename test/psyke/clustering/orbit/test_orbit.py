import unittest
from psyke import Extractor
import pandas as pd
import numpy as np


class AbstractTestMixedRulesGenerator(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [2, 1, 0],
                                [1, 2, 0],
                                [1.1, 2, 0],
                                [1.5, 2.7, 0],
                                [1, 1, 0],
                                [3, 4, 1],
                                [3, 3, 1],
                                [3, 5, 1],
                                [4, 4, 1],
                                [4, 5, 1],
                                [5, 4, 1],
                                [5, 5, 1],
                                [5.5, 5.5, 1],
                                [5.3, 5.3, 1],
                                [5, 5.3, 1],
                                [5.2, 5, 1]],
                               columns=["X", "Y", "cluster"])

        predictor = Predictor()
        predictor.fit(x=self.df.iloc[:, :-1], y=self.df.iloc[:, -1])
        self.extractor = Extractor.orbit(predictor=predictor,
                                         depth=2,
                                         error_threshold=0,
                                         gauss_components=2,
                                         steps=1000,
                                         min_accuracy_increase=0,
                                         max_disequation_num=4)


class TestMixedRulesGenerator(AbstractTestMixedRulesGenerator):
    def test_extraction(self):
        self.extractor.extract(self.df)


class Predictor:
    def __init__(self):
        self.result = {}

    def fit(self, x, y):
        X = x.to_numpy()
        Y = y
        for x, y in zip(X, Y):
            self.result[tuple(x)] = y

    def predict(self, x):
        return np.array([self.result[tuple(x_)] for x_ in x.to_numpy()])
