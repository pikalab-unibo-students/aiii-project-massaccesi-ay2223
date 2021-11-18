import unittest
from parameterized import parameterized_class
from tuprolog.solve.prolog import prolog_solver
from psyke import logger
from test import get_in_rule
from test.psyke import initialize, data_to_struct


@parameterized_class(initialize('real'))
class TestReal(unittest.TestCase):

    def test_extract(self):
        logger.info(self.expected_theory)
        logger.info(self.extracted_theory)
        self.assertTrue(self.expected_theory.equals(self.extracted_theory, False))

    def test_predict(self):
        predictions = self.extractor.predict(self.test_set.iloc[:, :-1])
        solver = prolog_solver(static_kb=self.extracted_theory.assertZ(get_in_rule()))

        # Sort columns alphabetically to be compliant to rules (terms are alphabetically ordered).
        columns = sorted(self.test_set.columns[:-1]) + [self.test_set.columns[-1]]
        self.test_set = self.test_set.reindex(columns, axis=1)
        substitutions = [solver.solveOnce(data_to_struct(data)) for _, data in self.test_set.iterrows()]
        index = self.test_set.shape[1] - 1
        expected = [str(query.solved_query.get_arg_at(index)) if query.is_yes else -1 for query in substitutions]

        self.assertTrue(predictions == expected)