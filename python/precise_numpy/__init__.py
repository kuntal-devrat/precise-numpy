"""
precise-numpy: NumPy-compatible interval arrays with guaranteed error bounds.

Every element is an interval represented as (midpoint, radius). Arithmetic
propagates both values using hardware-directed rounding, so results carry a
rigorous bound on the accumulated floating-point error.

Example:
    >>> import precise_numpy as pnp
    >>> a = pnp.array([0.1, 0.2, 0.3])
    >>> b = a + pnp.array([0.4, 0.5, 0.6])
    >>> b.sum()
    (2.1, 4.440892098500626e-16)
"""

from precise_numpy._precise_numpy import (
    BoolArray,
    IntervalArray,
    array,
    zeros,
    ones,
    empty,
    full,
    eye,
    identity,
    diag,
    linspace,
    arange,
    concatenate,
    stack,
    vstack,
    hstack,
    split,
    where_impl,
    from_raw_parts,
    num_threads,
    seed,
    rand,
    random_sample,
    randn,
    randint,
    uniform,
    normal,
    det,
    inv,
    solve,
    lstsq,
    eig,
    svd,
    pinv,
    __version__,
)

import struct as _struct


def _as_array(x):
    """Coerce an IntervalArray, scalar, or (midpoint, radius) tuple."""
    if isinstance(x, IntervalArray):
        return x
    if isinstance(x, (tuple, list)):
        return array(x)
    return array([x])


def asarray(a, error=0.0):
    """Convert the input to an IntervalArray."""
    return array(a, error=error)


def _binary(fname):
    def wrapper(a, b):
        return getattr(_as_array(a), fname)(_as_array(b))

    wrapper.__name__ = fname
    return wrapper


def _unary(fname):
    def wrapper(a):
        return getattr(_as_array(a), fname)()

    wrapper.__name__ = fname
    return wrapper


# NumPy-style aliases for IntervalArray methods.
abs_ = _unary("abs")
absolute = abs_  # numpy-compatible alias
sign = _unary("sign")
floor = _unary("floor")
ceil = _unary("ceil")
trunc = _unary("trunc")
round_ = _unary("round")
sqrt = _unary("sqrt")
exp = _unary("exp")
ln = _unary("ln")
log = ln  # numpy-compatible alias
log2 = _unary("log2")
log10 = _unary("log10")
sin = _unary("sin")
cos = _unary("cos")
tan = _unary("tan")
nan_to_num = _unary("nan_to_num")
isnan = _unary("isnan")
isinf = _unary("isinf")
isfinite = _unary("isfinite")
nonzero = _unary("nonzero")
sort = _unary("sort")
argsort = _unary("argsort")
norm = _unary("norm")
transpose = _unary("transpose")
maximum = _binary("maximum")
minimum = _binary("minimum")
power = _binary("power")
dot = _binary("dot")
matmul = _binary("matmul")


def clip(a, a_min, a_max):
    """Clip interval midpoints to [a_min, a_max] (radii preserved)."""
    return _as_array(a).clip(a_min, a_max)


def reshape(a, *shape):
    """Reshape an array; a single dimension may be -1 (inferred)."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        dims = tuple(shape[0])
    else:
        dims = tuple(shape)
    return _as_array(a).reshape(dims)


def sum_(a, axis=None):
    """Sum over all elements or along one axis."""
    return _as_array(a).sum(axis=axis)


def mean(a, axis=None):
    return _as_array(a).mean(axis=axis)


def prod(a, axis=None):
    return _as_array(a).prod(axis=axis)


def var(a, axis=None):
    return _as_array(a).var(axis=axis)


def std(a, axis=None):
    return _as_array(a).std(axis=axis)


def max(a, axis=None):
    return _as_array(a).max(axis=axis)


def min(a, axis=None):
    return _as_array(a).min(axis=axis)


def argmax(a, axis=None):
    return _as_array(a).argmax(axis=axis)


def argmin(a, axis=None):
    return _as_array(a).argmin(axis=axis)


def all(a, axis=None):
    return _as_array(a).all(axis=axis)


def any(a, axis=None):
    return _as_array(a).any(axis=axis)


def where(condition, x, y):
    """Select elements from x/y according to condition (broadcast together)."""
    return where_impl(condition, x, y)


# ── File I/O (implemented in Python) ────────────────────────────────────

_MAGIC = b"PNAV1"


def save(fname, arr):
    """Save an IntervalArray to a binary file (midpoints and radii)."""
    if not isinstance(arr, IntervalArray):
        raise TypeError("save() requires an IntervalArray")
    shape = list(arr.shape)
    mid = arr.values()
    rad = arr.radii()
    with open(fname, "wb") as f:
        f.write(_MAGIC)
        f.write(_struct.pack("<I", len(shape)))
        for d in shape:
            f.write(_struct.pack("<Q", d))
        for m, r in zip(mid, rad):
            f.write(_struct.pack("<dd", m, r))


def load(fname):
    """Load an IntervalArray saved with `save`."""
    with open(fname, "rb") as f:
        magic = f.read(5)
        if magic != _MAGIC:
            raise ValueError("not a precise_numpy save file")
        (ndim,) = _struct.unpack("<I", f.read(4))
        shape = list(_struct.unpack("<" + "Q" * ndim, f.read(8 * ndim)))
        total = 1
        for d in shape:
            total *= d
        data = f.read(16 * total)
        if len(data) != 16 * total:
            raise ValueError("file truncated")
        vals = _struct.unpack("<" + "dd" * total, data)
        mid = list(vals[0::2])
        rad = list(vals[1::2])
    return from_raw_parts(mid, rad, shape)


def savetxt(fname, arr, fmt="%.17g"):
    """Write midpoints/radii to a text file (one interval per line)."""
    if not isinstance(arr, IntervalArray):
        raise TypeError("savetxt() requires an IntervalArray")
    shape = " ".join(str(d) for d in arr.shape)
    with open(fname, "w") as f:
        f.write("# precise_numpy shape %s\n" % shape)
        for m, r in zip(arr.values(), arr.radii()):
            f.write((fmt + " " + fmt + "\n") % (m, r))


def loadtxt(fname):
    """Read an IntervalArray written by `savetxt`."""
    shape = None
    mid, rad = [], []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("# precise_numpy shape"):
                    shape = [int(tok) for tok in line.replace("# precise_numpy shape", "").split()]
                continue
            m, r = line.split()
            mid.append(float(m))
            rad.append(float(r))
    if shape is None:
        raise ValueError("not a precise_numpy savetxt file (missing shape header)")
    return from_raw_parts(mid, rad, shape)


__all__ = [
    "BoolArray",
    "IntervalArray",
    "array",
    "asarray",
    "zeros",
    "ones",
    "empty",
    "full",
    "eye",
    "identity",
    "diag",
    "linspace",
    "arange",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "split",
    "where",
    "from_raw_parts",
    "num_threads",
    "seed",
    "rand",
    "random_sample",
    "randn",
    "randint",
    "uniform",
    "normal",
    "det",
    "inv",
    "solve",
    "lstsq",
    "eig",
    "svd",
    "pinv",
    "save",
    "load",
    "savetxt",
    "loadtxt",
    "abs_",
    "absolute",
    "sign",
    "floor",
    "ceil",
    "trunc",
    "round_",
    "sqrt",
    "exp",
    "ln",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "nan_to_num",
    "isnan",
    "isinf",
    "isfinite",
    "nonzero",
    "sort",
    "argsort",
    "norm",
    "transpose",
    "maximum",
    "minimum",
    "power",
    "dot",
    "matmul",
    "clip",
    "reshape",
    "sum_",
    "mean",
    "prod",
    "var",
    "std",
    "max",
    "min",
    "argmax",
    "argmin",
    "all",
    "any",
    "__version__",
]
