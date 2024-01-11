"""
Definition of AdaptiveSweep
"""


import numpy as np
from .base import Sweep, SweepStep, SweepResult
from abc import ABC, abstractmethod
from itertools import chain, repeat
from adaptive import Learner1D
from time import time
from math import ceil

# TODO test all step_size (also factor < 1)
#      test start > stop and stop > start
#      check update is called
#      check multidimensionnal ?


class AdaptiveSweep(Sweep):
    """
    Dynamicly select step x-values base on the adaptive lib

    https://github.com/python-adaptive/adaptive.

    A sweep is an iterable object that returns a step at each iteration.

    The result of a sweep can be accessed with the `result` property.

    TODO should npoints be renamed nsteps ?
    """

    def __init__(self, start, stop, npoints, update=None):
        """
        Initializae the sweep.

        Parameters
        ----------
        start : float
            The start x-value of the sweep.
        stop : float
            The stop x-value of the sweep.
        npoints : int
            The number of points.
        update : function
            An optionnal function method can be passed to be called at each step.
        """

        super().__init__(update=update)

        self.start = start
        self.stop = stop
        self.npoints = npoints
        self._learner = None
        self._step = None

    @property
    def result(self):
        """The result of the sweep."""

        # sort data along x
        data = np.array(sorted(self._learner.data.items(), key=lambda v: v[0]))

        return SweepResult(*data.T)

    def __iter__(self):
        def nope(x):
            pass

        self._learner = Learner1D(nope, (self.start, self.stop))

        # reset result
        self._result = SweepResult([], [])
        self._step = None

        return self

    def __next__(self):
        # save previous step result
        if self._step is not None:
            self._learner.tell(self._step.x, self._step.y)

        if self._learner.npoints >= self.npoints:
            raise StopIteration()

        self._step = SweepStep(x=self._learner.ask(1)[0][0])

        if self.update is not None:
            self.update(self._step.x)

        # may change x and y
        return self._step

    def __len__(self):
        return self.npoints
