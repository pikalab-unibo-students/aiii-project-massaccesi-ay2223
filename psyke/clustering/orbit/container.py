from __future__ import annotations
from statistics import mode
from typing import Iterable, Union
import pandas as pd
from numpy import ndarray

from psyke.extraction.hypercubic.hypercube import ClosedClassificationCube
from psyke.utils import get_default_precision, get_int_precision
from psyke.extraction.hypercubic import Node
from psyke.extraction.hypercubic.hypercube import HyperCube
from psyke.utils.logic import create_term
import numpy as np

from tuprolog.core import Var, Struct
from psyke.schema import Between
from tuprolog.core import struct, real
PRECISION: int = get_int_precision()


def get_int_precision() -> int:
    from math import log10
    return -1 * int(log10(get_default_precision()))


Constraint = tuple[float, float, float]
Constraints = dict[tuple[str, str], tuple[float, float, float]]


class FeatureNotFoundException(Exception):
    def __init__(self, feature: str):
        super().__init__('Feature "' + feature + '" not found.')


class OutputNotDefinedException(Exception):
    def __init__(self):
        super().__init__('The output of the container is not defined')


class ConstraintFeaturesNotFoundException(Exception):

    def __init__(self, feature: tuple[str, str]):
        super().__init__(f'Constraint feature {feature} not found.')


class ContainerNode(Node):
    def __init__(self, dataframe: pd.DataFrame, container: Container = None):
        super().__init__(dataframe, container)
        self.dataframe = dataframe
        self.container: Container = container
        self.right: ContainerNode | None = None
        self.left: ContainerNode | None = None

    @property
    def children(self) -> list[ContainerNode]:
        return [self.right, self.left]

    def search(self, point: dict[str, float]) -> Container:
        if self.right is None:
            return self.container
        if self.right.container.__contains__(point):
            return self.right.search(point)
        return self.left.search(point)

    @property
    def leaves(self):
        if self.right is None:
            return 1
        return self.right.leaves + self.left.leaves


class Container(ClosedClassificationCube):
    """
    A N-dimensional cube holding a numeric value.
    """

    EPSILON = get_default_precision()  # Precision used when comparing two hypercubes
    INT_PRECISION = get_int_precision()

    def __init__(self,
                 dimension: dict[str, tuple],
                 disequation: dict[tuple[str, str], list[tuple[float, float, float]]] = {}):
        """
        :param dimension: given a dimension is the interval which constraint the data
        :param disequation: is in the form (X,Y): a,b,c, which identifies the constraint aX + bY <= c,
            where X and Y are the names of the features that are being constrained
        """
        self._disequations = disequation
        self._output = None
        super().__init__(dimension=dimension)

    def update(self, dataset: pd.DataFrame, predictor) -> None:
        # filtered = self._filter_dataframe(dataset.iloc[:, :-1])
        filtered = self._filter_dataframe(dataset.iloc[:, :-1])
        if len(filtered > 0):
            predictions = predictor.predict(filtered)
            self._output = mode(predictions)
            self._diversity = 1 - sum(prediction == self.output for prediction in predictions) / len(filtered)

    def filter_indices(self, dataset: pd.DataFrame) -> ndarray:
        output = np.full(len(dataset.index), True, dtype=bool)
        for column in self.dimensions.keys():
            out = np.logical_and(dataset[column] >= self.dimensions[column][0],
                                 dataset[column] <= self.dimensions[column][1])
            if output is None:
                output = out
            else:
                output = np.logical_and(output, out)
        output = np.logical_and(output, Container.check_sat_constraints(self._disequations, dataset))
        if isinstance(output, pd.Series):
            output = output.to_numpy()
        return output

        # v = np.array([v for _, v in self._dimensions.items()])
        # ds = dataset.to_numpy(copy=True)
        # return np.all((v[:, 0] <= ds) & (ds < v[:, 1]), axis=1)

    def _filter_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[self.filter_indices(dataset)]

    @staticmethod
    def check_sat_constraints(constraints: dict[tuple[str, str], list[tuple[float, float, float]]], dataset) \
            -> np.ndarray:
        output = np.full(len(dataset.index), True, dtype=bool)
        for constr_columns in constraints:
            col1 = constr_columns[0]
            col2 = constr_columns[1]
            for a, b, c in constraints[constr_columns]:
                out = dataset[col1] * a + dataset[col2] * b <= c
                if output is None:
                    output = out
                else:
                    output = output & out
        return output

    @staticmethod
    def create_surrounding_cube(dataset: pd.DataFrame, closed: bool = False,
                                output=None) -> Container:
        hyper_cube = HyperCube.create_surrounding_cube(dataset, closed)
        return Container(hyper_cube.dimensions)

    def copy(self) -> Container:
        return Container(self.dimensions.copy(), self._disequations.copy())

    @property
    def disequations(self) -> dict[tuple[str, str], list[tuple[float, float, float]]]:
        return self._disequations

    def body(self, variables: dict[str, Var], ignore: list[str], unscale=None, normalization=None) -> Iterable[Struct]:
        """
        generate the body of the theory that describes this container
        :param variables:
        :param ignore:
        :param unscale:
        :param normalization:
        :return:
        """
        dimensions = dict(self.dimensions)
        constraints = self.disequations.copy()
        for dimension in ignore:
            del dimensions[dimension]
        for (dim1, dim2) in self.disequations:
            if dim1 in ignore or dim2 in ignore:
                del constraints[(dim1, dim2)]
        dimension_out = [create_term(variables[name], Between(unscale(values[0], name), unscale(values[1], name)))
                         for name, values in dimensions.items()]

        constr_out = []
        for dim1, dim2 in self.disequations:
            for a, b, c in self.disequations[(dim1, dim2)]:
                x = struct("*", real(round(a, PRECISION)), variables[dim1])
                y = struct("*", real(round(b, PRECISION)), variables[dim2])
                constr_out.append(struct("=<", struct("+", x, y), real(round(c, PRECISION))))

        return dimension_out + constr_out

    def __contains__(self, point: dict[str, float]) -> bool:
        """
        Note that a point (dict[str, float]) is inside a container if ALL its dimensions' values satisfy:
            aX + bY <= c
        :param point: an N-dimensional point
        :return: true if the point is inside the container, false otherwise
        """
        for X, Y in self.disequations.keys():
            x = point[X]
            y = point[Y]
            for a, b, c in self._disequations[(X, Y)]:
                if not(a * x + b * y <= c):
                    return False
        for dim in self.dimensions:
            start, end = self.dimensions[dim]
            if not(start <= point[dim] <= end):
                return False
        return True

    @property
    def output(self):
        if self._output is None:
            raise OutputNotDefinedException()
        else:
            return self._output

    def _fit_constraint(self, dimension: dict[tuple[str, str], tuple[float, float, float]]) \
            -> dict[tuple[str, str], tuple[float, float, float]]:
        new_dimension: dict[tuple[str, str], tuple[float, float, float]] = {}
        for key, value in dimension.items():
            new_dimension[key] = (round(value[0], self.INT_PRECISION),
                                  round(value[1], self.INT_PRECISION),
                                  round(value[2], self.INT_PRECISION))
        return new_dimension


GenericContainer = Union[Container]
