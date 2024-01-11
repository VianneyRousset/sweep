#!/usr/bin/env python

import unittest
from vsweep import SweepResult
import numpy as np


class TestSweepResult(unittest.TestCase):
    """Test the SweepResult."""

    def test_convertion_to_ndarray(self):
        """Check that a SweepResult can be converted to an ndarray."""

        x = np.random.random(10)
        y = np.random.random(10)

        result = SweepResult(x, y)

        self.assertTrue(np.all(np.array(result) == np.array([x, y])))

    def test_equal(self):
        """Check equal operator."""

        x1 = np.random.random(10)
        x2 = np.random.random(10)
        y1 = np.random.random(10)
        y2 = np.random.random(10)

        self.assertEqual(SweepResult(x1, y1), SweepResult(x1, y1))
        self.assertNotEqual(SweepResult(x1, y1), SweepResult(x2, y1))
        self.assertNotEqual(SweepResult(x1, y1), SweepResult(x1, y2))
        self.assertNotEqual(SweepResult(x1, y1), SweepResult(x2, y2))

    def test_concatenate(self):
        """Check that SweepResults can be concatenate using +."""

        x1 = np.random.random(10)
        y1 = np.random.random(10)
        result1 = SweepResult(x1, y1)

        x2 = np.random.random(10)
        y2 = np.random.random(10)
        result2 = SweepResult(x2, y2)

        expected_result = SweepResult(
            x=np.concatenate([x1, x2]), y=np.concatenate([y1, y2])
        )

        self.assertEqual(result1 + result2, expected_result)

    def test_unpacking(self):
        """Check that a SweepResult can be unpacked."""

        x = list(np.random.random(10))
        y = list(np.random.random(10))

        result = SweepResult(x, y)

        x0, y0 = result

        self.assertEqual(x0, x)
        self.assertEqual(y0, y)

    def test_attributes(self):
        """Check that result x and y can be accessed."""

        x = list(np.random.random(10))
        y = list(np.random.random(10))

        result = SweepResult(x, y)

        self.assertEqual(result.x, x)
        self.assertEqual(result.y, y)

    def test_non_numerical_attributes(self):
        """Check that result x and y can be accessed."""

        x1 = [None, "abc"]
        y1 = [[2, 3, 4, 5], ()]
        result1 = SweepResult(x1, y1)

        x2 = [[1, 2, 3]]
        y2 = [42]
        result2 = SweepResult(x2, y2)

        expected_result = SweepResult(x=[*x1, *x2], y=[*y1, *y2])

        self.assertEqual(result1 + result2, expected_result)

    def test_none_y(self):
        """Check that a None can be stored as y."""

        result = SweepResult(42, None)

        self.assertEqual(result.y, [None])
