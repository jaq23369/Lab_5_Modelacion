import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Ejercicio_2.ipynb"
RESULTS = ROOT / "resultados_berlin52.json"
PLOTS = (ROOT / "mejor_ruta_berlin52.png", ROOT / "convergencia_berlin52.png")


class NotebookDeliverableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.notebook = notebook
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        cls.code = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_is_valid_and_has_required_sections(self):
        notebook = self.notebook

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(notebook["nbformat_minor"], 5)
        self.assertGreaterEqual(len(notebook["cells"]), 8)

        for section in (
            "Berlin52",
            "Objetivo",
            "Metodología",
            "Configuración",
            "Resultados",
            "Visualización",
            "Conclusión",
        ):
            self.assertIn(section, self.markdown)

        for required_text in ("7542", "EUC_2D", "berlin52.tsp"):
            self.assertIn(required_text, self.markdown + self.code)

    def test_code_compiles_and_supports_package_and_local_imports(self):
        tree = ast.parse(self.code, NOTEBOOK.name)
        imported_modules = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        self.assertIn("Ejercicio_2.berlin52_ga", imported_modules)
        self.assertIn("berlin52_ga", imported_modules)
        compile(tree, NOTEBOOK.name, "exec")

        for required_code in (
            "resultados_berlin52.json",
            "mejor_ruta_berlin52.png",
            "convergencia_berlin52.png",
        ):
            self.assertIn(required_code, self.code)

    def test_executed_artifacts_report_a_closed_one_based_best_route(self):
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
        route = results["ruta_1_based_cerrada"]
        distances = [run["distancia"] for run in results["corridas"]]

        self.assertEqual(results["semillas"], [5, 7, 42])
        self.assertEqual(
            results["parametros"],
            {
                "N": 120,
                "max_iter": 300,
                "seleccion": "torneo",
                "k": 4,
                "s": 0.05,
                "c": 0.70,
                "m": 0.20,
            },
        )
        self.assertEqual(len(results["semillas"]), len(distances))
        self.assertEqual(results["mejor_distancia"], min(distances))
        self.assertEqual(results["mejor_distancia"], 7542)
        self.assertEqual(results["gap_absoluto"], 0)
        self.assertEqual(route[0], route[-1])
        self.assertEqual(set(route[:-1]), set(range(1, 53)))
        self.assertEqual(len(route), 53)
        self.assertEqual(
            results["gap_absoluto"],
            results["mejor_distancia"] - results["optimo"],
        )

        for plot in PLOTS:
            self.assertGreater(plot.stat().st_size, 1_000)
            self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
