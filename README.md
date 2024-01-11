# vsweep

Cute way to perform sweeps

## Installation

```bash
pip install git+https://github.com/VianneyRousset/vsweep
```

## Usage

Vsweep provides `Sweep` objects that can be used to perform sweeps. They are iterators that returns steps with a provided `x` value and records the `y` value given by the user. The result can then be retrieved using `sweep.result`.

### Basic linear sweep

Here is an example that perform a linear sweep for a measurement:

```python3
from vsweep import LinearSweep

sweep = LinearSweep(0, 5, 1000)

def measure(x):
  # do some measurement (here fake data is generated)
  return x**2

for step in sweep:
  step.y = measure(step.x)

x, y = sweep.result
plt.plot(x, y)
```

### Types of sweeps

The following sweeps are available:

- `LinearSweep` follows a linspace distribution of the x values.
- `LogSweep` follows a logspace distribution of the x values.
- `VectorSweep` for an arbitrary list of x values.
- `AdaptiveSweep` intelligently selects the best x values using the [adaptive](https://github.com/python-adaptive/adaptive) library.

![Types of sweeps](visuals/sweeps.png?raw=true)
