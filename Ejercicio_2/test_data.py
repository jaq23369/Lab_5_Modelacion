import unittest
from pathlib import Path

import numpy as np

from Ejercicio_2.berlin52_ga import leer_tsplib, matriz_euc_2d


DATA = Path(__file__).with_name("berlin52.tsp")


class Berlin52DataTests(unittest.TestCase):
    def test_parser_reads_all_52_nodes(self):
        coordinates, _ = leer_tsplib(DATA)

        self.assertEqual(coordinates.shape, (52, 2))

    def test_parser_preserves_euc_2d_metadata(self):
        _, metadata = leer_tsplib(DATA)

        self.assertEqual(metadata["EDGE_WEIGHT_TYPE"], "EUC_2D")

    def test_parser_preserves_first_and_last_coordinates(self):
        coordinates, _ = leer_tsplib(DATA)

        np.testing.assert_array_equal(coordinates[0], [565, 575])
        np.testing.assert_array_equal(coordinates[-1], [1740, 245])

    def test_euc_2d_uses_tsplib_nearest_integer_rounding(self):
        coordinates = np.array([[0, 0], [1, 1], [2, 2], [3, 4]], dtype=float)

        distances = matriz_euc_2d(coordinates)

        self.assertEqual(distances[0, 1], 1)
        self.assertEqual(distances[0, 2], 3)
        self.assertEqual(distances[0, 3], 5)


if __name__ == "__main__":
    unittest.main()
