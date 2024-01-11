#!/usr/bin/env python

import unittest
from vsweep import AdaptiveSweep
import numpy as np


class TestAdaptiveSweep(unittest.TestCase):
    """Test AdaptiveSweep."""

    def test_basic_adaptive_sweep(self):
        """Check x and y values."""

        sweep = AdaptiveSweep(-8, 5, 1000)

        for step in sweep:
            step.y = step.x**2

        # check we have the right number of steps
        self.assertEqual(len(sweep.result.x), 1000)

        # check that x is monotonically increasing
        for increment in np.diff(sweep.result.x):
            self.assertGreater(increment, 0)

        # check start is stop reached
        self.assertEqual(sweep.result.x[0], -8)

        # check that stop has been reached
        self.assertEqual(sweep.result.x[-1], 5)

        # check y values
        for x, y in zip(sweep.result.x, sweep.result.y):
            self.assertEqual(y, x**2)

    def test_point_density(self):
        """Check that more steps has been allocated to the region of interest."""

        def gaussian(x, x0, sigma):
            return np.exp(-(((x - x0) / sigma) ** 2))

        sweep = AdaptiveSweep(-5, 8, 1000)

        for step in sweep:
            step.y = gaussian(step.x, 2, 0.1)

        roi = (np.array(sweep.result.x) >= 0) & (np.array(sweep.result.x) <= 4)

        density_inside_roi = np.sum(roi) / 4
        density_outside_roi = np.sum(~roi) / (5 + 4)

        self.assertGreater(density_inside_roi, density_outside_roi)
