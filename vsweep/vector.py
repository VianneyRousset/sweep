"""
Implemented as iterators -> can be cascaded
"""

from .base import Sweep, SweepStep, SweepResult

import numpy as np
from itertools import chain, repeat
from time import time
from math import ceil

# TODO test all step_size (also factor < 1)
#      test start > stop and stop > start
#      check update is called
#      check multidimensionnal ?


class VectorSweep(Sweep):
    """
    Sweep iterating over a vector of x values.

    A sweep is an iterable object that returns a step at each iteration.

    The result of a sweep can be accessed with the `result` property.
    """

    def __init__(self, vector, roundtrip=False, npasses=1, update=None):
        """
        Initializae the sweep.

        Parameters
        ----------
        vector : list
            The vector of x values.
        roundtrip : bool
            If True, the vector will be repeated in both directions.
        npasses : int
            The number of times the vector will be repeated.
        update : function
            An optionnal function method can be passed to be called at each step.
        """

        super().__init__(update=update)

        self.vector = list(vector)
        self.roundtrip = roundtrip
        self.npasses = npasses

        self._result = None
        self._step = None
        self._iterator = None
        self._start_time = None

    @property
    def result(self):
        """The result of the sweep."""
        return self._result

    def __iter__(self):
        self._start_time = time()

        # reset result
        self._result = SweepResult([], [])
        self._step = None

        # create roundtrip and avoid repetitions of the last elements
        if self.roundtrip:
            # prepare passes
            vector = self.vector

            # create roundtrip and discard last element [1,2,3,4,5,4,3,2]
            vector = list(chain(vector[::1], vector[-2:0:-1]))

            # repeat the roundtrip for each pass
            vector = chain(*repeat(list(vector), self.npasses))

            # put back the last element
            vector = list(chain(vector, self.vector[:1]))

            self._iterator = enumerate(vector)

        else:
            # repeat the vector for each pass
            self._iterator = enumerate(chain(*repeat(self.vector, self.npasses)))

        return self

    def __next__(self):
        # save data of the previous step
        if self._step is not None:
            self._result = self._result + [self._step.x, self._step.y]

        # create next step
        elapsed_time = time() - self._start_time
        n, x = next(self._iterator)
        self._step = SweepStep(
            x=x, step_number=n, total=len(self), elapsed_time=elapsed_time
        )

        # call update
        if self.update:
            self.update(self._step)

        # may change x and y
        return self._step

    def __len__(self):
        total = len(self.vector) * self.npasses

        if self.roundtrip:
            total = total * 2

        return total


class LinearSweep(VectorSweep):
    """
    Uniformly spaced sweep.

    A sweep is an iterable object that returns a step at each iteration.

    The result of a sweep can be accessed with the `result` property.
    """

    def __init__(
        self,
        start,
        stop,
        nsteps=None,
        max_step_size=None,
        roundtrip=False,
        npasses=1,
        update=None,
    ):
        """
        Initializae the sweep.

        Parameters
        ----------
        start : float
            The start x-value of the sweep.
        stop : float
            The stop x-value of the sweep.
        nsteps : int
            The number of steps.
        max_step_size : float
            x value maximum absolute difference between two steps. If provided,
            nsteps is ignored and the number of steps is computed to guarantee
            that each absolute difference between two steps is less or equal to
            max_step_size.
        roundtrip : bool
            If True, the vector will be repeated in both directions.
        npasses : int
            The number of times the vector will be repeated.
        update : function
            An optionnal function method can be passed to be called at each
            step.
        """

        # save previous step result
        if max_step_size is not None:
            if max_step_size <= 0:
                raise ValueError("max_step_size must be > 0")

            nsteps = int(ceil(abs((stop - start) / max_step_size))) + 1

        vector = np.linspace(start, stop, nsteps)

        super().__init__(
            vector=vector, roundtrip=roundtrip, npasses=npasses, update=update
        )


class LogSweep(VectorSweep):
    """
    Logarithmically spaced sweep.

    A sweep is an iterable object that returns a step at each iteration.

    The result of a sweep can be accessed with the `result` property.
    """

    def __init__(
        self,
        start,
        stop,
        nsteps=None,
        max_step_factor=None,
        roundtrip=False,
        npasses=1,
        update=None,
    ):
        """
        Initializae the sweep.

        Parameters
        ----------
        start : float
            The start x-value of the sweep.
        stop : float
            The stop x-value of the sweep.
        nsteps : int
            The number of steps.
        max_step_factor : float
            x value maximum increase between two steps. If provided, nsteps is
            ignored and the number of steps is computed to guarantee that each
            increment is less or equal to max_step_size.
        roundtrip : bool
            If True, the vector will be repeated in both directions.
        npasses : int
            The number of times the vector will be repeated.
        update : function
            An optionnal function method can be passed to be called at each step.
        """

        # save previous step result
        if max_step_factor is not None:
            if max_step_factor <= 1:
                raise ValueError("max_step_factor must be > 1")

            nsteps = ceil(abs(np.log(stop / start)) / np.log(max_step_factor)) + 1

        vector = np.logspace(np.log10(start), np.log10(stop), nsteps)

        super().__init__(
            vector=vector, roundtrip=roundtrip, npasses=npasses, update=update
        )
