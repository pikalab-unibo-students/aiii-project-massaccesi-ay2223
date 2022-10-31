import unittest
from psyke.clustering.orbit.mixed_rules_extractor import *


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
        self.extractor = MixedRulesExtractor(depth=2,
                                             error_threshold=0,
                                             gauss_components=2,
                                             steps=1000,
                                             min_accuracy_increase=0,
                                             max_disequation_num=4)


class TestMixedRulesGenerator(AbstractTestMixedRulesGenerator):
    def test_extraction(self):
        self.extractor.extract(self.df)

