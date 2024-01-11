"""
Definition of a `Sweep`, `SweepStep` and `SweepResult`.

Iteration over a `Sweep` object yield `SweepSteps` with a `x` value. The value
measured at each step must be recorded in `SweepStep.y`. Sweep result is given
by `Sweep.result` as a `SweepResult` object containing both axis `SweepResult.x`
and `SwepResult.y`.
"""

import numpy as np
from abc import ABC, abstractmethod


class Sweep(ABC):
    """
    Base class for sweep.

    A sweep is an iterable object that returns a step at each iteration.

    The result of a sweep can be accessed with the `result` property.
    """

    def __init__(self, update=None):
        """
        Initializae the sweep.

        Parameters
        ----------
        update : function
            An optionnal function method can be passed to be called at each step.
        """
        self.update = update

    @property
    @abstractmethod
    def result(self):
        """The result of the sweep."""
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


class SweepStep:
    """
    Store the informations of a step of a sweep.

    Attributes
    ----------
    x : float
        The x value of the step.
    y : float
        The y value of the step. The latter is must be set by the user.
    step_number : int
        The number of the step.
    total : int
        The total number of steps in the sweep.
    elapsed_time : float
        The elapsed time when since the beginning of the sweep.
    """

    def __init__(self, x, y=None, step_number=None, total=None, elapsed_time=None):
        """
        Initializae the step.

        Parameters
        ----------
        x : float
            The x value of the step.
        y : float
            The y value of the step. The latter is must be set by the user.
        step_number : int
            The number of the step.
        total : int
            The total number of steps in the sweep.
        elapsed_time : float
            The elapsed time when since the beginning of the sweep.
        """

        self.x = x
        self.y = y
        self.n = step_number
        self.total = total
        self.elapsed_time = elapsed_time

    def __repr__(self):
        def tabulated(info):
            w = max(len(k) for k, v in info.items())
            fmt = "{:>" + str(w) + "} : {:e}"
            return "\n".join(fmt.format(k, v) for k, v in info.items())

        # format nicely the step number
        step = None
        if step is not None:
            if self.total is not None:
                step = f"{self.n} / {self.total}"
            else:
                step = f"{self.n} / ?"

        info = {
            "x": self.x,
            "y": self.y,
            "step": step,
            "elapsed time": self.elapsed_time,
        }

        # filter None info
        info = {k: v for k, v in info.items() if v is not None}

        return f"--- Sweep step ---\n" + tabulated(info)


class SweepResult:
    """
    Store the result of a sweep.

    Attributes
    ----------
    x : list
        The list of x values.
    y : list
        The list of y values.
    """

    # used a replacement for None
    _novalue = object()

    def __init__(self, x, y=_novalue):
        """
        Initializae the result.

        Parameters
        ----------
        x : list or SweepResult
            The list of x values.
        y : list
            The list of y values.
        """

        if y is self._novalue:
            x, y = x

        try:
            self.x = list(x)
            self.y = list(y)

        except TypeError:
            self.x = [x]
            self.y = [y]

    def __array__(self):
        return np.array([self.x, self.y])

    def __add__(self, other):
        other = SweepResult(*other)
        return SweepResult(self.x + other.x, self.y + other.y)

    def __iter__(self):
        return iter([self.x, self.y])

    def __eq__(self, other):
        other = SweepResult(other)
        return self.x == other.x and self.y == other.y
