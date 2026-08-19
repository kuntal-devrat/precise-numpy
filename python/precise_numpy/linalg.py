"""Linear algebra helpers, mirroring numpy.linalg's core API."""

from precise_numpy._precise_numpy import det, inv, solve, lstsq, eig, svd, pinv
from precise_numpy import _as_array, abs_, power, sqrt, sum_, max, min


def _reduce_norm(a, ord, axis):
    """Norm of `a` along the given axis for a supported `ord`.

    Returns an IntervalArray (one interval per slice). The interval
    arithmetic is rigorous: the result encloses the norm of every matrix
    in the input's interval family.
    """
    if ord is None or ord == 2:
        return sqrt(sum_(power(a, 2), axis=axis))
    if ord == 1:
        return sum_(abs_(a), axis=axis)
    if ord == float("inf"):
        return max(abs_(a), axis=axis)
    if ord == float("-inf"):
        return min(abs_(a), axis=axis)
    if ord == 0:
        raise ValueError("ord=0 is not supported (count of nonzero elements)")
    raise ValueError("unsupported ord %r for vector norm along an axis" % (ord,))


def norm(a, ord=None, axis=None):
    """Matrix or vector norm of the array.

    Matches ``numpy.linalg.norm`` for the supported orders:

    - ``axis=None`` (default): Frobenius norm for matrices, L2 for vectors.
      For a matrix, ``ord`` may be ``None``/``'fro'`` (Frobenius), ``1``
      (max column abs-sum), ``-1`` (min column abs-sum), ``inf`` (max row
      abs-sum), ``-inf`` (min row abs-sum), ``2`` (spectral norm, largest
      singular value) or ``'nuc'`` (sum of singular values).
    - ``axis=k``: vector norm along that axis, ``ord`` in
      {``None``, ``2``, ``1``, ``inf``, ``-inf``}.

    Scalar results are returned as ``(midpoint, radius)`` so the error
    bound is never hidden; axis reductions return an ``IntervalArray``.
    """
    a = _as_array(a)
    if axis is None:
        if a.ndim == 1:
            if ord in (None, 2):
                return a.norm()
            if ord == 1:
                return sum_(abs_(a))
            if ord == float("inf"):
                return max(abs_(a))
            if ord == float("-inf"):
                return min(abs_(a))
            raise ValueError("unsupported ord %r for a vector norm" % (ord,))
        if ord is None or ord == "fro":
            return a.norm()
        if ord == 1:
            return max(sum_(abs_(a), axis=0))
        if ord == -1:
            return min(sum_(abs_(a), axis=0))
        if ord == float("inf"):
            return max(sum_(abs_(a), axis=1))
        if ord == float("-inf"):
            return min(sum_(abs_(a), axis=1))
        if ord == 2:
            s = svd(a)[1]
            return (s.values()[0], s.radii()[0])
        if ord == "nuc":
            ss = svd(a)[1].sum()
            return (ss[0], ss[1])
        raise ValueError("unsupported ord %r for a matrix norm" % (ord,))
    if isinstance(axis, (list, tuple)):
        raise ValueError("multiple axes are not supported; pass a single int axis")
    if not isinstance(axis, int):
        raise ValueError("axis must be an integer or None")
    return _reduce_norm(a, ord, axis)


__all__ = ["det", "inv", "solve", "lstsq", "eig", "svd", "pinv", "norm"]