import tempfile
import unittest
from pathlib import Path

import numpy as np

from Ejercicio_2.berlin52_ga import (
    ResultadoGA,
    algoritmo_genetico_tsp,
    distancia_recorrido,
    ejecutar_experimentos,
    es_recorrido_valido,
    leer_tsplib,
    matriz_euc_2d,
)


class TestDatosTSP(unittest.TestCase):
    def test_leer_tsplib_conserva_orden_y_metadatos(self):
        contenido = """NAME: mini
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 10 20
2 13 24
3 -2 7
EOF
"""
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "mini.tsp"
            ruta.write_text(contenido, encoding="utf-8")
            coords, meta = leer_tsplib(ruta)

        np.testing.assert_array_equal(coords, [[10, 20], [13, 24], [-2, 7]])
        self.assertEqual(meta["NAME"], "mini")
        self.assertEqual(meta["DIMENSION"], 3)
        self.assertEqual(meta["EDGE_WEIGHT_TYPE"], "EUC_2D")

    def test_leer_tsplib_rechaza_dimension_inconsistente(self):
        contenido = """NAME: roto
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
EOF
"""
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "roto.tsp"
            ruta.write_text(contenido, encoding="utf-8")
            with self.assertRaises(ValueError):
                leer_tsplib(ruta)

    def test_matriz_euc_2d_redondea_segun_tsplib(self):
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 4.0]])
        matriz = matriz_euc_2d(coords)
        esperado = np.array([[0, 1, 5], [1, 0, 4], [5, 4, 0]])
        np.testing.assert_array_equal(matriz, esperado)

    def test_distancia_incluye_regreso_al_origen(self):
        matriz = np.array([[0, 2, 9], [2, 0, 3], [9, 3, 0]])
        self.assertEqual(distancia_recorrido([0, 1, 2], matriz), 14)

    def test_valida_exactamente_una_visita_por_ciudad(self):
        self.assertTrue(es_recorrido_valido([2, 0, 1], 3))
        self.assertFalse(es_recorrido_valido([0, 1, 1], 3))
        self.assertFalse(es_recorrido_valido([0, 1], 3))
        self.assertFalse(es_recorrido_valido([0, 1, 3], 3))


class TestAlgoritmoGenetico(unittest.TestCase):
    COORDS = np.array(
        [[0, 0], [4, 0], [6, 2], [5, 5], [2, 6], [-1, 4], [-2, 1]],
        dtype=float,
    )

    def test_semilla_produce_resultado_repetible_y_valido(self):
        parametros = dict(N=28, max_iter=15, semilla=17)
        primero = algoritmo_genetico_tsp(self.COORDS, **parametros)
        segundo = algoritmo_genetico_tsp(self.COORDS, **parametros)

        self.assertIsInstance(primero, ResultadoGA)
        self.assertEqual(primero.distancia, segundo.distancia)
        np.testing.assert_array_equal(primero.recorrido, segundo.recorrido)
        self.assertTrue(es_recorrido_valido(primero.recorrido, len(self.COORDS)))
        self.assertEqual(len(primero.historial_mejor), 16)
        self.assertEqual(len(primero.historial_promedio), 16)
        self.assertTrue(np.all(np.diff(primero.historial_mejor) <= 0))
        self.assertEqual(primero.semilla, 17)
        self.assertEqual(primero.parametros["seleccion"], "torneo")

    def test_parametros_invalidos_se_rechazan(self):
        casos = (
            {"N": 1},
            {"max_iter": -1},
            {"seleccion": "ruleta"},
            {"k": 0},
            {"s": 0.8, "c": 0.5, "m": 0.1},
        )
        for caso in casos:
            with self.subTest(caso=caso), self.assertRaises(ValueError):
                algoritmo_genetico_tsp(self.COORDS, **caso)

    def test_proporciones_y_conteos_invalidos_se_rechazan(self):
        casos = (
            {"s": -0.1},
            {"s": 1.9},
            {"c": 2.8},
            {"m": float("nan")},
            {"s": float("inf")},
            {"c": float("-inf")},
            {"m": "dos"},
            {"s": None},
            {"c": True},
            {"s": 21},
        )
        for caso in casos:
            with self.subTest(caso=caso), self.assertRaises((TypeError, ValueError)):
                algoritmo_genetico_tsp(self.COORDS, N=20, max_iter=0, **caso)

    def test_experimentos_devuelve_el_minimo(self):
        mejor, resultados = ejecutar_experimentos(
            self.COORDS, [3, 5, 7], N=20, max_iter=5
        )
        self.assertEqual([r.semilla for r in resultados], [3, 5, 7])
        self.assertEqual(mejor.distancia, min(r.distancia for r in resultados))
        self.assertTrue(any(mejor is resultado for resultado in resultados))

    def test_berlin52_se_acerca_al_optimo_con_configuracion_reproducible(self):
        ruta_datos = Path(__file__).with_name("berlin52.tsp")
        coords, _ = leer_tsplib(ruta_datos)
        resultado = algoritmo_genetico_tsp(
            coords,
            N=120,
            max_iter=300,
            k=4,
            s=0.05,
            c=0.70,
            m=0.20,
            semilla=7,
        )
        self.assertLessEqual(resultado.distancia, 7800)


if __name__ == "__main__":
    unittest.main()
