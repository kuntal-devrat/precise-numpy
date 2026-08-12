"""
precise-numpy: High-performance NumPy-compatible interval arrays with
guaranteed numerical error bounds, powered by Rust SIMD.

Every element carries an error bound (midpoint +/- radius),
and every operation propagates these bounds correctly using hardware-directed rounding.

Example:
    >>> import precise_numpy as pnp
    >>> a = pnp.array([1.0, 2.0, 3.0])
    >>> b = pnp.array([4.0, 5.0, 6.0])
    >>> c = a + b
    >>> print(c)
    IntervalArray([5, 7, 9])
    >>> print(c.max_relative_error())
    0.0
"""

from precise_numpy._precise_numpy import (
    IntervalArray,
    array,
    zeros,
    ones,
    full,
    linspace,
    arange,
    num_threads,
    __version__,
)

__all__ = [
    "IntervalArray",
    "array",
    "zeros",
    "ones",
    "full",
    "linspace",
    "arange",
    "num_threads",
    "__version__",
]
