<div align="center">

  <img src="assets/logo.png" alt="precise-numpy logo" width="380" />

  # precise-numpy

  **Interval arithmetic with guaranteed error bounds for Python, powered by Rust SIMD.**

  [![PyPI Version](https://img.shields.io/pypi/v/precise-numpy.svg?color=007ec6)](https://pypi.org/project/precise-numpy/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/precise-numpy.svg?color=3776ab)](https://pypi.org/project/precise-numpy/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

</div>

---

## Overview

`precise-numpy` is a Python library for numerical computing with provable error bounds.
Every element is an interval stored as **(midpoint, radius)**; every operation propagates
both values with hardware-directed rounding (`MXCSR` on x86_64), so results carry a rigorous
enclosure of the true mathematical value.

This is not a drop-in replacement for NumPy — arrays track two `f64` values per element and
intervals behave differently from plain floats (see [Semantics](#semantics)). It is a fast,
NumPy-flavored API for interval arithmetic.

Key capabilities:
- **Hardware-directed rounding** for guaranteed mathematical enclosures.
- **Rust SIMD acceleration** (AVX-512, AVX2/FMA, NEON) with Rayon parallelism.
- **NumPy-like API**: creation, indexing, broadcasting, reductions, stacking, linalg, random, I/O.
- **GIL release** during long computations.

---

## Installation

```bash
pip install precise-numpy
```

Build from source:

```bash
git clone https://github.com/your-org/precise-numpy.git
cd precise-numpy
maturin develop --release
```

Requires Python >= 3.10 and a Rust toolchain. Wheels are built for CPython (abi3).

---

## Quick Start

```python
import precise_numpy as pnp

# Create interval arrays with error bounds
a = pnp.array([0.1, 0.2, 0.3], error=1e-4)
b = pnp.array([0.4, 0.5, 0.6], error=1e-4)

# Arithmetic propagates both midpoints and radii
c = a + b
print(c.radii())            # [2e-4, 2e-4, 2e-4]
print(c.max_radius())       # 2e-4

# Reductions return (midpoint, radius) tuples
mid, err = c.sum()
print(f"sum = {mid} +/- {err}")

# Inspect error growth
print("max relative error:", c.max_relative_error())
```

### Semantics

- **Values**: `get(i)`, `a[0]`, `sum()`, etc. return a `(midpoint, radius)` tuple.
  To get raw float arrays use `a.values()` / `a.radii()`.
- **`==` / `!=`** mean *interval overlap*; `<, <=, >, >=` use strict endpoint ordering
  (`a.hi < b.lo`, etc.).
- **Division by zero** raises a `RuntimeWarning` and produces the entire real line
  (radius `inf`) for the affected element.
- **`len(a)`** is `a.shape[0]`, matching NumPy.
- **Reductions** (`sum`, `mean`, ...) support an optional `axis` argument; scalar results
  come back as `(midpoint, radius)` tuples, axis results as `IntervalArray`s.
- **dtype** is `"interval64"`, `itemsize` is 16 (two `f64`s per element).
- **Rounding** (`round`) uses round-half-to-even, like Python and NumPy.
- **`reshape(-1, ...)`** infers the missing dimension; `transpose`/`.t` support 1D/2D.

---

## API Reference

### Array creation

`array(values, error=0.0)`, `asarray`, `zeros(shape)`, `ones`, `empty`, `full(shape, value, error=0.0)`,
`eye(n, m=None, k=0)`, `identity(n)`, `diag(v)`, `arange(start, stop, step=1.0)`,
`linspace(start, stop, num=50, endpoint=True)`, `from_raw_parts(midpoints, radii, shape)`.

### Properties & conversions

`shape`, `ndim`, `size`, `dtype`, `itemsize`, `nbytes`, `strides`, `t`,
`values()`, `radii()`, `tolist()`, `get(i)`, `item(i)`, `midpoint(i)`, `radius(i)`,
`copy()`, `flatten()`, `ravel()`, `reshape(*shape)`, `transpose()`, `sort()`, `argsort()`.

### Indexing

Integer (with negatives), slices (`a[1:4]`, `a[::-1]`), fancy integer lists (`a[[0, 2]]`),
boolean masks (`a[a > 2.0]`, `a[BoolArray]`), and 2D tuples (`m[1, :]`, `m[:, 0]`).
Assignment works for integers, slices, and lists.

### Arithmetic & math

`+ - * / ** @`, in-place variants, scalar operands on either side, NumPy broadcasting.
`sin, cos, tan, exp, ln, log2, log10, sqrt, abs, floor, ceil, trunc, round,
clip(a_min, a_max), sign, nan_to_num, power(x, y), maximum(x, y), minimum(x, y)`.

### Comparisons

`==`, `!=` (overlap), `<, <=, >, >=` (strict ordering) return `BoolArray` values with
`& | ^ ~`, `any()`, `all()`, `sum()`, `count_nonzero()`, `tolist()`.

### Reductions

`sum, mean, prod, var, std, min, max, argmin, argmax, all, any` — each with an optional
`axis`; plus `min_val()`, `max_val()`, `cumsum(axis=None)`, `norm()` (L2).

### Stacking & selection

`concatenate(arrays, axis=0)`, `stack`, `vstack`, `hstack`, `split(a, sections_or_indices, axis=0)`,
`where(condition, x, y)`, `nonzero(a)` (tuple of index lists per axis).

### Linear algebra (`precise_numpy.linalg`)

`det`, `inv`, `solve(A, b)`, `lstsq(A, b)`, `pinv`, `eig`, `svd`, `norm`.

Notes: `eig` (symmetric, Jacobi) and `eig_general`-style Hessenberg QR with real
eigenvalues only — complex pairs raise `NotImplementedError`/`ValueError`. `svd` returns
the thin decomposition `U (m, k), s (k,), Vh (k, n)` with `k = min(m, n)`. All linalg
routines operate on midpoints (radius 0 results) except `solve`, which uses interval LU.

### Random (`precise_numpy.random`)

`seed`, `rand(*size)`, `random_sample(*size)`, `random(*size)`, `randn(*size)`,
`randint(low, high, *size)`, `uniform(low, high, *size)`, `normal(loc=0.0, scale=1.0, *size)`.
Deterministic xoshiro256** generator; no `size` argument returns a Python float.

### File I/O

`save(path, arr)` / `load(path)` (binary) and `savetxt(path, arr, fmt="%.17g")` /
`loadtxt(path)` (text with a shape header). Both round-trip midpoints and radii.

---

## Error-Bound Example: AI Quantization Audit

```python
import precise_numpy as pnp

# Wrap input data with interval error bounds
X = pnp.array(floats, error=1e-4).reshape([128, 64])
W = pnp.array(weights).reshape([64, 64])

# Propagate error through a linear layer
Q = X.matmul(W)

print("Max Radius Error:", Q.max_radius())
print("Max Relative Error:", Q.max_relative_error())
```

See [`examples/quantization_safety_audit.py`](examples/quantization_safety_audit.py) and
[`examples/precision_audit_test.py`](examples/precision_audit_test.py).

---

## Performance Notes

Each element stores two `f64` values and each op enforces directed rounding, so the library
is roughly 1.1x–1.7x slower than plain NumPy element-wise ops on large arrays while
tracking rigorous error bounds — and it is often faster than NumPy on small arrays because
the PyO3 extension avoids per-call Python dispatch overhead. See
[`examples/full_benchmark.py`](examples/full_benchmark.py).

---

## Technical Architecture

1. **Structure-of-Arrays (SoA) buffer**: midpoints and radii in contiguous, 64-byte aligned
   memory (`AlignedBuffer`), enabling direct SIMD loads.
2. **Single-pass streaming SIMD**: radius bounds fused into one pass over both buffers
   (AVX-512, AVX2+FMA, NEON).
3. **Shared buffers**: `IntervalArray` clones, slices, and reshapes are O(1) reference-counted.
4. **GIL release**: long-running kernels run under `py.allow_threads()` with Rayon parallelism.
5. **No external dependencies at runtime**; pure Rust + pyo3.

---

## Testing

```bash
cargo test          # Rust unit tests (arithmetic, linalg, reduction, extra, random)
python -m unittest tests/python/test_api.py   # Python integration tests
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.