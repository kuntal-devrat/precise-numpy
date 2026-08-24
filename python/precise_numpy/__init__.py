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

import builtins
import math
import struct as _struct

import numpy as _np

from precise_numpy._precise_numpy import (
    BoolArray,
    IntervalArray,
    __version__,
    arange,
    array,
    cholesky,
    concatenate,
    cond,
    det,
    diag,
    eig,
    empty,
    eye,
    from_raw_parts,
    full,
    hstack,
    identity,
    inv,
    linspace,
    lstsq,
    matrix_power,
    matrix_rank,
    normal,
    num_threads,
    ones,
    pinv,
    rand,
    randint,
    randn,
    random_sample,
    seed,
    solve,
    split,
    stack,
    svd,
    uniform,
    vstack,
    where_impl,
    zeros,
)


def _as_array(x):
    """Coerce an IntervalArray, scalar, or (midpoint, radius) tuple."""
    if isinstance(x, IntervalArray):
        return x
    if isinstance(x, (tuple, list)):
        return array(x)
    return array([x])


# -----------------------------------------------------------------------
# NumPy interop (NEP18): np.asarray(IntervalArray) returns midpoints.
# -----------------------------------------------------------------------


def _to_numpy(self, dtype=None):
    """NEP18 __array__ protocol: convert midpoints to a numpy ndarray."""
    import numpy as _np

    return _np.asarray(self.values(), dtype=dtype).reshape(self.shape)


IntervalArray.__array__ = _to_numpy


def to_numpy(a, dtype=None):
    """Explicit conversion: IntervalArray -> numpy ndarray (midpoints).

    Parameters
    ----------
    a : IntervalArray or scalar-convertible.
    dtype : numpy dtype, optional
        Passed to ``numpy.asarray``. Defaults to float64.
    """
    arr = _as_array(a)
    return _np.asarray(arr.values(), dtype=dtype).reshape(arr.shape)


def from_numpy(x, error=0.0):
    """Convert a numpy ndarray (or array-like) to an IntervalArray.

    The returned array stores ``x``'s values as midpoints.  ``error`` may
    be a scalar (uniform radius) or a broadcastable array of radii.

    Parameters
    ----------
    x : array-like or scalar.
    error : float or array-like, optional
        Uniform (scalar) or per-element radius to add.
    """
    import numpy as _np

    x = _np.asarray(x, dtype=_np.float64)
    mids = x.ravel().tolist()
    n = x.size
    if hasattr(error, "__iter__") and not isinstance(error, str):
        arr = _np.asarray(error, dtype=float)
        rads = arr.reshape(x.shape).ravel().tolist()
    else:
        rads = [float(error)] * n
    return from_raw_parts(mids, rads, list(x.shape))


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
        for m, r in zip(mid, rad, strict=True):
            f.write(_struct.pack("<dd", m, r))


def load(fname):
    """Load an IntervalArray saved with `save`."""
    with open(fname, "rb") as f:
        magic = f.read(5)
        if magic != _MAGIC:
            raise ValueError("not a precise_numpy save file")
        (ndim,) = _struct.unpack("<I", f.read(4))
        shape = list(_struct.unpack("<" + "Q" * ndim, f.read(8 * ndim)))
        total = math.prod(shape)
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
        f.write(f"# precise_numpy shape {shape}\n")
        for m, r in zip(arr.values(), arr.radii(), strict=True):
            f.write(f"{fmt % m} {fmt % r}\n")


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
    "cholesky",
    "matrix_power",
    "matrix_rank",
    "cond",
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
    "to_numpy",
    "from_numpy",
    "astype",
    "save_npy",
    "load_npy",
]

# -----------------------------------------------------------------------
# astype: convert IntervalArray to plain ndarray or keep interval view
# -----------------------------------------------------------------------


def astype(a, dtype=None):
    """Convert an IntervalArray to another representation.

    Parameters
    ----------
    a : IntervalArray or scalar-convertible.
    dtype : numpy dtype or string, optional
        - ``None`` / ``'interval64'`` / ``'interval'`` : returns the IntervalArray itself.
        - ``float`` / ``np.float64`` etc. : returns a plain ndarray of midpoints.
        - ``int`` / ``np.int64`` etc. : returns a plain ndarray of midpoints cast to int.

    The radii are silently discarded when converting to a plain ndarray.
    """
    arr = _as_array(a)
    if dtype is None:
        return arr
    try:
        dt = _np.dtype(dtype)
    except TypeError:
        return arr
    name = dt.name.lower()
    if "interval" in name:
        return arr
    return _np.asarray(arr.values(), dtype=dt).reshape(arr.shape)


# -----------------------------------------------------------------------
# numpy-compatible I/O (.npy format: combined mid+rad array)
# -----------------------------------------------------------------------


def save_npy(fname, arr):
    """Save an IntervalArray as a numpy ``.npy`` file.

    The stored array has dtype float64 and shape ``(*shape, 2)`` where the
    last dimension holds ``[midpoint, radius]``.  It can be loaded with
    ``load_npy`` or inspected with ``np.load`` (midpoints are at index 0
    along the last axis).
    """
    if not isinstance(arr, IntervalArray):
        raise TypeError("save_npy requires an IntervalArray")
    mids = _np.asarray(arr.values(), dtype=_np.float64).reshape(arr.shape)
    rads = _np.asarray(arr.radii(), dtype=_np.float64).reshape(arr.shape)
    combined = _np.stack([mids, rads], axis=-1)
    _np.save(fname, combined)


def load_npy(fname):
    """Load an IntervalArray saved with ``save_npy``."""
    combined = _np.load(fname)
    if combined.ndim < 1 or combined.shape[-1] != 2:
        raise ValueError("not a precise_numpy .npy file (last dim must be 2)")
    shape = list(combined.shape[:-1])
    mids = combined[..., 0].ravel().tolist()
    rads = combined[..., 1].ravel().tolist()
    return from_raw_parts(mids, rads, shape)


# -----------------------------------------------------------------------
# NumPy ufunc protocol (NEP-13): np.add, np.exp, np.sin, … on pnp arrays
# -----------------------------------------------------------------------


def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """Handle numpy ufunc calls involving IntervalArrays."""
    if method != "__call__":
        return NotImplemented
    if kwargs.get("out", None) is not None:
        return NotImplemented  # out= kwarg not yet supported
    if kwargs.get("where", False) is not False:
        return NotImplemented

    name = ufunc.__name__

    unary = {
        "negative": lambda a: -a,
        "positive": lambda a: a,
        "absolute": lambda a: abs_(a),
        "rint": lambda a: round_(a),
        "sign": lambda a: sign(a),
        "exp": lambda a: exp(a),
        "exp2": lambda a: exp(a) ** 2,
        "log": lambda a: ln(a),
        "log2": lambda a: log2(a),
        "log10": lambda a: log10(a),
        "sqrt": lambda a: sqrt(a),
        "sin": lambda a: sin(a),
        "cos": lambda a: cos(a),
        "tan": lambda a: tan(a),
        "isnan": lambda a: isnan(a),
        "isinf": lambda a: isinf(a),
        "isfinite": lambda a: isfinite(a),
    }

    binary_fwd = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "true_divide": lambda a, b: a / b,
        "divide": lambda a, b: a / b,
        "power": lambda a, b: power(a, b),
        "matmul": lambda a, b: matmul(a, b),
    }

    binary_rev = {
        "add": lambda other, self: self + other,
        "multiply": lambda other, self: self * other,
        "subtract": lambda other, self: other - self,
        "true_divide": lambda other, self: full(self.shape, float(other)) / self,
        "divide": lambda other, self: full(self.shape, float(other)) / self,
        "power": lambda other, self: power(self, other),
    }

    if name in unary and len(inputs) == 1 and isinstance(inputs[0], IntervalArray):
        return unary[name](inputs[0])

    if len(inputs) != 2:
        return NotImplemented

    a, b = inputs
    if isinstance(a, IntervalArray) and name in binary_fwd:
        return binary_fwd[name](a, b)
    if isinstance(b, IntervalArray) and isinstance(a, IntervalArray) and name in binary_fwd:
        return binary_fwd[name](a, b)
    if isinstance(b, IntervalArray) and name in binary_rev:
        return binary_rev[name](a, b)

    return NotImplemented


IntervalArray.__array_ufunc__ = __array_ufunc__


# -----------------------------------------------------------------------
# NumPy function protocol (NEP-18): np.linalg.norm, np.concatenate, …
# -----------------------------------------------------------------------

_HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    def decorator(implementation):
        _HANDLED_FUNCTIONS[numpy_function] = implementation
        return implementation

    return decorator


@implements(_np.linalg.norm)
def _np_linalg_norm(x, ord=None, axis=None, keepdims=False):
    if not isinstance(x, IntervalArray):
        return NotImplemented
    import precise_numpy.linalg as _la

    result = _la.norm(x, ord=ord, axis=axis)
    if keepdims and isinstance(result, IntervalArray):
        # Make reduced dimensions size-1
        if axis is None:
            shape = [1] * x.ndim
            return result.reshape(shape)
        if isinstance(axis, (list, tuple)):
            new_shape = list(x.shape)
            for ax in axis:
                new_shape[ax] = 1
            return result.reshape(new_shape)
        new_shape = list(x.shape)
        new_shape[axis] = 1
        return result.reshape(new_shape)
    return result


@implements(_np.linalg.det)
def _np_linalg_det(x):
    if not isinstance(x, IntervalArray):
        return NotImplemented
    return det(x)


@implements(_np.linalg.inv)
def _np_linalg_inv(x):
    if not isinstance(x, IntervalArray):
        return NotImplemented
    return inv(x)


@implements(_np.linalg.solve)
def _np_linalg_solve(a, b):
    if not isinstance(a, IntervalArray) and not isinstance(b, IntervalArray):
        return NotImplemented
    return solve(a, b)


@implements(_np.linalg.lstsq)
def _np_linalg_lstsq(a, b, rcond="warn"):
    if not isinstance(a, IntervalArray) and not isinstance(b, IntervalArray):
        return NotImplemented
    return lstsq(a, b)


@implements(_np.linalg.svd)
def _np_linalg_svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    if not full_matrices:
        raise ValueError("precise_numpy only supports full_matrices=True for svd")
    u, s, vt = svd(a)
    if not compute_uv:
        return s
    return u, s, vt


@implements(_np.linalg.eig)
def _np_linalg_eig(a):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    return eig(a)


@implements(_np.linalg.eigvals)
def _np_linalg_eigvals(a):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    evals, _ = eig(a)
    return evals


@implements(_np.linalg.pinv)
def _np_linalg_pinv(a, rcond=1e-15, hermitian=False):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    return pinv(a)


@implements(_np.concatenate)
def _np_concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    if out is not None or dtype is not None:
        return NotImplemented
    arys = [arr if isinstance(arr, IntervalArray) else array(arr) for arr in arrays]
    return concatenate(arys, axis=axis)


@implements(_np.stack)
def _np_stack(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    if out is not None or dtype is not None:
        return NotImplemented
    arys = [arr if isinstance(arr, IntervalArray) else array(arr) for arr in arrays]
    return stack(arys, axis=axis)


@implements(_np.sum)
def _np_sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if out is not None or dtype is not None or initial is not None:
        return NotImplemented
    if where is not True:
        return NotImplemented
    return sum_(a, axis=axis)


@implements(_np.mean)
def _np_mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    if out is not None or dtype is not None:
        return NotImplemented
    return mean(a, axis=axis)


@implements(_np.std)
def _np_std(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    if out is not None or dtype is not None:
        return NotImplemented
    return std(a, axis=axis)


@implements(_np.var)
def _np_var(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    if out is not None or dtype is not None:
        return NotImplemented
    return var(a, axis=axis)


@implements(_np.max)
def _np_max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    if out is not None or initial is not None:
        return NotImplemented
    return max(a, axis=axis)


@implements(_np.min)
def _np_min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    if out is not None or initial is not None:
        return NotImplemented
    return min(a, axis=axis)


@implements(_np.all)
def _np_all(a, axis=None, out=None, keepdims=False, where=True):
    if out is not None:
        return NotImplemented
    return all(a, axis=axis)


@implements(_np.any)
def _np_any(a, axis=None, out=None, keepdims=False, where=True):
    if out is not None:
        return NotImplemented
    return any(a, axis=axis)


@implements(_np.transpose)
def _np_transpose(a, axes=None):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    return a.transpose()


@implements(_np.reshape)
def _np_reshape(a, newshape, order="C"):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    if isinstance(newshape, int):
        newshape = (newshape,)
    return reshape(a, *newshape)


@implements(_np.squeeze)
def _np_squeeze(a, axis=None):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    if axis is None:
        return a.reshape([d for d in a.shape if d != 1])
    return a.reshape(
        [1 if i == axis else d for i, d in enumerate(a.shape) if not (i == axis and d == 1)]
    )


@implements(_np.expand_dims)
def _np_expand_dims(a, axis):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    shape = list(a.shape)
    if axis < 0:
        axis += len(shape) + 1
    shape.insert(axis, 1)
    return a.reshape(shape)


@implements(_np.rollaxis)
def _np_rollaxis(a, axis, start=0):
    if not isinstance(a, IntervalArray):
        return NotImplemented
    # simplified: just use transpose for single-axis roll
    return a.transpose()


def __array_function__(self, func, types, args, kwargs):
    """NEP-18: handle numpy function calls on IntervalArrays."""
    if not builtins.all(issubclass(t, IntervalArray) for t in types):
        return NotImplemented
    if func not in _HANDLED_FUNCTIONS:
        return NotImplemented
    return _HANDLED_FUNCTIONS[func](*args, **kwargs)


IntervalArray.__array_function__ = __array_function__
