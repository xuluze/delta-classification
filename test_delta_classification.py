"""Regression tests: run ``python -m unittest -v test_delta_classification``."""

import ast
from collections import Counter
from itertools import combinations
import logging
from math import gcd
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from delta_classification import (
    Polyhedron, delta_classification,
    lattice_polytopes_with_given_dimension_and_delta,
)


def normal_form(polytope):
    """Compare up to lattice equivalence, independently of vertex ordering."""
    return tuple(tuple(v) for v in polytope.normal_form())


class DeltaClassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.WARNING)
        cls.result = delta_classification(m=2, Delta=5, mode="delta")

    def test_matches_published_classification(self):
        data = Path(__file__).resolve().parent / "data/dim_2_delta_5.txt"
        expected = [Polyhedron(vertices=v) for v in ast.literal_eval(data.read_text())]
        self.assertEqual(len(self.result), 16)
        actual_keys = Counter(map(normal_form, self.result))
        self.assertEqual(actual_keys, Counter(map(normal_form, expected)))
        self.assertTrue(all(count == 1 for count in actual_keys.values()))

    def test_geometric_invariants(self):
        for polytope in self.result:
            with self.subTest(vertices=polytope.n_vertices()):
                self.assertEqual(polytope.dim(), 2)
                vertices = {tuple(v) for v in polytope.vertices()}
                self.assertEqual(vertices, {tuple(-x for x in v) for v in vertices})
                self.assertTrue(all(x.denominator() == 1 for v in vertices for x in v))
                points = list(polytope.integral_points())
                largest_minor = max(abs(a[0]*b[1] - a[1]*b[0])
                                    for a, b in combinations(points, 2))
                self.assertEqual(largest_minor, 5)

    def test_extremal_data(self):
        maximum = max(len(p.integral_points()) for p in self.result)
        self.assertEqual(maximum, 23)
        extremal = [p for p in self.result if len(p.integral_points()) == maximum]
        data = Path(__file__).resolve().parent / "data/dim_2_delta_5_extremal.txt"
        expected = [Polyhedron(vertices=v) for v in ast.literal_eval(data.read_text())]
        self.assertEqual(Counter(map(normal_form, extremal)), Counter(map(normal_form, expected)))

    def test_extremal_mode(self):
        result = delta_classification(m=2, Delta=5, mode="delta_ext")
        maximum = max(len(p.integral_points()) for p in self.result)
        expected = [p for p in self.result if len(p.integral_points()) == maximum]
        self.assertEqual(Counter(map(normal_form, result)), Counter(map(normal_form, expected)))

    def test_original_sage_sources(self):
        # GAP (a groups dependency) brings passagemath-repl transitively.
        # Check that the Python port agrees with the unmodified Sage code.
        from sage.repl.preparse import preparse_file

        namespace = {}
        exec("from sage.all__sagemath_polyhedra import *\n"
             "from sage.all__sagemath_combinat import *", namespace)
        source_dir = Path(__file__).resolve().parent

        def load_source(filename):
            path = source_dir / filename
            exec(compile(preparse_file(path.read_text()), str(path), "exec"), namespace)

        namespace["load"] = load_source
        load_source("delta-classification.sage")
        original = namespace["delta_classification"](2, 5, "delta")
        self.assertEqual(Counter(map(normal_form, original)), Counter(map(normal_form, self.result)))


class PrimitiveColumnsTest(unittest.TestCase):
    def test_fresh_classification_preserves_primitive_maxima(self):
        # Bypass database files: incomplete sandwich enumeration lost these examples.
        for delta, expected in ((19, 23), (23, 27)):
            with self.subTest(delta=delta):
                result = delta_classification(2, delta, "delta")
                maximum = max(
                    sum(gcd(*map(int, v)) == 1 for v in p.integral_points()) // 2
                    for p in result
                )
                self.assertEqual(maximum, expected)

    def setUp(self):
        self.repo = Path(__file__).resolve().parent
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(temporary.name)
        Path("data").mkdir()

    def test_cached_columns_are_preserved(self):
        # Include a nonvertex primitive point: hull conversion would lose it.
        columns = [[(1, 0), (1, 1), (1, 2)]]
        cache = Path("data/dim_2_delta_5_simple.txt")
        cache.write_text(repr(columns))
        before = cache.stat().st_mtime_ns
        self.assertEqual(delta_classification(2, 5, "delta_simple"), columns)
        self.assertEqual(cache.stat().st_mtime_ns, before)

    def test_generates_missing_columns(self):
        shutil.copy2(self.repo / "data/dim_2_delta_5_maximal.txt", "data")
        expected = ast.literal_eval(
            (self.repo / "data/dim_2_delta_5_simple.txt").read_text())
        result = delta_classification(2, 5, "delta_simple")
        self.assertEqual(result, expected)
        self.assertEqual(ast.literal_eval(
            Path("data/dim_2_delta_5_simple.txt").read_text()), expected)

    def test_polytope_wrapper_redirects_before_writing(self):
        with self.assertRaisesRegex(ValueError, "Use delta_classification"):
            lattice_polytopes_with_given_dimension_and_delta(2, 5, "delta_simple")
        self.assertEqual(list(Path("data").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
