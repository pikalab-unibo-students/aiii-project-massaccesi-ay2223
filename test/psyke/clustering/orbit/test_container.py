import unittest
import pandas as pd

from psyke.extraction.hypercubic.hypercube import FeatureNotFoundException
from psyke.utils import get_int_precision
from sklearn.neighbors import KNeighborsRegressor
from test.psyke import Predictor
from psyke.clustering.orbit.container import Container


class AbstractTestContainer(unittest.TestCase):

    def setUp(self):
        self.dimensions = {'X': (0.2, 0.6), 'Y': (0.7, 0.9)}
        self.mean = 0.5
        self.container_cube = Container(self.dimensions)
        self.disequations = {("X", "Y"):
                             [(-1, -1, -1),
                              (1, 0, 1),
                              (-1, 1, 0.01)]}
        self.container = Container({}, self.disequations)
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


class TestContainer(AbstractTestContainer):

    def test_get_dimension(self):
        self.assertEqual(self.dimensions, self.container_cube.dimensions)

    def test_get_diseqution(self):
        self.assertEqual(self.disequations, self.container.disequations)

    def test_get(self):
        self.assertEqual((0.2, 0.6), self.container_cube['X'])
        self.assertEqual((0.7, 0.9), self.container_cube['Y'])
        with self.assertRaises(FeatureNotFoundException):
            dummy = self.container_cube['Z']

    def test_get_first(self):
        self.assertEqual(0.2, self.container_cube.get_first('X'))
        self.assertEqual(0.7, self.container_cube.get_first('Y'))
        with self.assertRaises(FeatureNotFoundException):
            self.container_cube.get_first('Z')

    def test_get_second(self):
        self.assertEqual(0.6, self.container_cube.get_second('X'))
        self.assertEqual(0.9, self.container_cube.get_second('Y'))
        with self.assertRaises(FeatureNotFoundException):
            self.container_cube.get_second('Z')

    def test_copy(self):
        copy = self.container_cube.copy()
        self.assertEqual(self.container_cube.dimensions, copy.dimensions)
        self.assertEqual(self.container_cube.output, copy.output)

        copy = self.container.copy()
        self.assertEqual(self.container.disequations, copy.disequations)
        self.assertEqual(self.container.output, copy.output)

    def test_count(self):
        self.assertEqual(self.df.shape[0], Container.create_surrounding_cube(self.df).count(self.df))

    def test_update_mean(self):
        model = KNeighborsRegressor()
        model.fit(self.df.iloc[:, :-1], self.df.iloc[:, -1])
        predictor = Predictor(model)
        self.container.update(self.df, predictor)

    def test_update_dimension(self):
        new_lower, new_upper = 0.6, 1.4
        updated = {'X': (new_lower, new_upper),
                   'Y': (0.7, 0.9)}
        new_cube1 = self.container_cube.copy()
        new_cube1.update_dimension('X', new_lower, new_upper)
        self.assertEqual(updated, new_cube1.dimensions)
        new_cube2 = self.container_cube.copy()
        new_cube2.update_dimension('X', (new_lower, new_upper))
        self.assertEqual(updated, new_cube2.dimensions)

    def test_create_surrounding_cube(self):
        surrounding = Container.create_surrounding_cube(self.df)
        for feature in self.df.columns[:-1]:
            self.assertEqual((round(min(self.df[feature]) - Container.EPSILON * 2, get_int_precision()),
                              round(max(self.df[feature]) + Container.EPSILON * 2, get_int_precision())),
                             surrounding.dimensions[feature])
