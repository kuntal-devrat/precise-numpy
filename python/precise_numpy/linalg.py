"""Linear algebra helpers, mirroring numpy.linalg's core API."""

from precise_numpy._precise_numpy import det, inv, solve, lstsq, eig, svd, pinv
from precise_numpy import _as_array


def norm(a, ord=None, axis=None):
    """Frobenius (L2) norm of the array as an interval (midpoint, radius)."""
    return _as_array(a).norm()


__all__ = ["det", "inv", "solve", "lstsq", "eig", "svd", "pinv", "norm"]
