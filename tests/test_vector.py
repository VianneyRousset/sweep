#!/usr/bin/env python

import unittest
from vsweep import VectorSweep, LinearSweep, LogSweep
import numpy as np


class TestVectorSweep(unittest.TestCase):
    """Test VectorSweep."""

    def test_single_vector_sweep(self):
        """Check that x follows the given vector and y are stored."""

        sweep = VectorSweep([1, 2, 3])

        for step in sweep:
            step.y = step.x**2

        self.assertEqual(sweep.result.x, list([1, 2, 3]))
        self.assertEqual(sweep.result.y, list([1, 4, 9]))

    def test_multiple_passes(self):
        """Check x-values with multiple passes."""

        sweep = VectorSweep([2, 3, 4], npasses=3)

        for step in sweep:
            pass

        self.assertEqual(sweep.result.x, [2, 3, 4, 2, 3, 4, 2, 3, 4])

    def test_roundtrip(self):
        """Check x-values with roundtrip."""

        sweep = VectorSweep([1, 2, 3], roundtrip=True)

        for step in sweep:
            pass

        self.assertEqual(sweep.result.x, [1, 2, 3, 2, 1])

    def test_roundtrip_multiple_passes(self):
        """Check x-values with multiple passes and roundtrip."""

        sweep = VectorSweep([2, 3, 4], roundtrip=True, npasses=3)

        for step in sweep:
            pass

        self.assertEqual(sweep.result.x, [2, 3, 4, 3, 2, 3, 4, 3, 2, 3, 4, 3, 2])

    def test_len(self):
        """Check that the length can be retrieved."""

        sweep = VectorSweep([1, 2, 3, 4, 5])

        self.assertEqual(len(sweep), 5)


class TestLinearSweep(unittest.TestCase):
    """Test LinearSweep."""

    def test_nsteps(self):
        """Check x-values with nsteps."""

        sweep = LinearSweep(2, 7, nsteps=6)

        for step in sweep:
            pass

        self.assertEqual(sweep.result.x, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_nsteps_roundtrip_multiple_passes(self):
        """Check x-values with nsteps, roundstrip and multiple passes."""

        sweep = LinearSweep(3, 5, nsteps=3, roundtrip=True, npasses=3)

        for step in sweep:
            pass

        self.assertEqual(
            sweep.result.x,
            [3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0],
        )

    def test_max_step_size(self):
        """Check x-values step size is less or equal to max_step_size."""

        for max_step_size in [0.1, 1, 10]:
            sweep = LinearSweep(3, 5, max_step_size=max_step_size)

            for step in sweep:
                pass

            for x0, x1 in zip(sweep.result.x[:-1], sweep.result.x[1:]):
                self.assertGreater(x1 - x0, 0)
                self.assertLessEqual(x1 - x0, 1.001 * max_step_size)

    def test_invalid_max_step_size(self):
        """Check ValueError is raise if max_step_size not > 0."""

        with self.assertRaises(ValueError):
            LinearSweep(2, 5, max_step_size=0)

        with self.assertRaises(ValueError):
            LinearSweep(2, 5, max_step_size=-1)

    def test_reverse(self):
        """Check x-values with nsteps when start > stop."""

        sweep = LinearSweep(7, 2, nsteps=6)

        for step in sweep:
            pass

        self.assertEqual(sweep.result.x, [7.0, 6.0, 5.0, 4.0, 3.0, 2.0])

    def test_reverse_nsteps_roundtrip_multiple_passes(self):
        """Check x-values with nsteps, roundstrip and multiple passes when start > stop."""

        sweep = LinearSweep(5, 3, nsteps=3, roundtrip=True, npasses=3)

        for step in sweep:
            pass

        self.assertEqual(
            sweep.result.x,
            [5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0],
        )

    def test_reverse_max_step_size(self):
        """Check x-values with max_step_size when start > stop."""

        for max_step_size in [0.1, 1, 10]:
            sweep = LinearSweep(5, 3, max_step_size=max_step_size)

            for step in sweep:
                pass

            for x0, x1 in zip(sweep.result.x[:-1], sweep.result.x[1:]):
                self.assertLess(x1 - x0, 0)
                self.assertLessEqual(abs(x1 - x0), 1.001 * max_step_size)

    def test_update_is_called(self):
        """Check that the given update function is called."""

        check = False

        def update(step):
            nonlocal check

            self.assertFalse(check)
            check = True

        sweep = LinearSweep(0, 10, nsteps=11, update=update)

        for step in sweep:
            self.assertTrue(check)
            check = False


class TestLogSweep(unittest.TestCase):
    """Test LogSweep."""

    def test_nsteps(self):
        """Check x-values with nsteps."""

        sweep = LogSweep(10, 1_000_000, nsteps=6)

        for step in sweep:
            pass

        self.assertEqual(sweep.result.x, [10, 100, 1000, 10_000, 100_000, 1_000_000])

    def test_nsteps_roundtrip_multiple_passes(self):
        """Check x-values with nsteps, roundstrip and multiple passes."""

        sweep = LogSweep(10, 1000, nsteps=3, roundtrip=True, npasses=3)

        for step in sweep:
            pass

        self.assertEqual(
            sweep.result.x,
            [10, 100, 1000, 100, 10, 100, 1000, 100, 10, 100, 1000, 100, 10],
        )

    def test_max_step_factor(self):
        """Check x-values step factor is less or equal to max_step_factor."""

        for max_step_factor in [1.2, 5, 10000]:
            sweep = LogSweep(10, 1000, max_step_factor=max_step_factor)

            for step in sweep:
                pass

            for x0, x1 in zip(sweep.result.x[:-1], sweep.result.x[1:]):
                self.assertGreater(x1 / x0, 1)
                self.assertLessEqual(x1 / x0, 1.001 * max_step_factor)

    def test_invalid_max_step_factor(self):
        """Check ValueError is raise if max_step_factor <= 1."""

        with self.assertRaises(ValueError):
            LogSweep(10, 1000, max_step_factor=1)

        with self.assertRaises(ValueError):
            LogSweep(10, 100, max_step_factor=0.5)

    def test_reverse(self):
        """Check x-values with nsteps when start > stop."""

        sweep = LogSweep(1000, 10, nsteps=3)

        for step in sweep:
            pass

        self.assertEqual(sweep.result.x, [1000.0, 100.0, 10.0])

    def test_reverse_nsteps_roundtrip_multiple_passes(self):
        """Check x-values with nsteps, roundstrip and multiple passes when start > stop."""

        sweep = LogSweep(1000, 10, nsteps=3, roundtrip=True, npasses=3)

        for step in sweep:
            pass

        self.assertEqual(
            sweep.result.x,
            [
                1000.0,
                100.0,
                10.0,
                100.0,
                1000.0,
                100.0,
                10.0,
                100.0,
                1000.0,
                100.0,
                10.0,
                100.0,
                1000.0,
            ],
        )

    def test_reverse_max_step_factor(self):
        """Check x-values with max_step_factor when start > stop."""

        for max_step_factor in [1.2, 5, 10000]:
            sweep = LogSweep(1000, 100, max_step_factor=max_step_factor)

            for step in sweep:
                pass

            for x0, x1 in zip(sweep.result.x[:-1], sweep.result.x[1:]):
                self.assertGreater(x0 / x1, 1)
                self.assertLessEqual(x0 / x1, 1.001 * max_step_factor)

    def test_update_is_called(self):
        """Check that the given update function is called."""

        check = False

        def update(step):
            nonlocal check

            self.assertFalse(check)
            check = True

        sweep = LogSweep(10, 1_000_000, nsteps=6, update=update)

        for step in sweep:
            self.assertTrue(check)
            check = False
