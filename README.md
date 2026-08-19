<div align="center">

  <img src="assets/logo.png" alt="precise-numpy logo" width="320" />

  # precise-numpy

  **Interval arithmetic with guaranteed error bounds for Python, powered by Rust.**

[![PyPI Version](https://img.shields.io/pypi/v/precise-numpy.svg?color=007ec6)](https://pypi.org/project/precise-numpy/)
[![Python Versions](https://img.shields.io/pypi/pyversions/precise-numpy.svg?color=3776ab)](https://pypi.org/project/precise-numpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Beta](https://img.shields.io/badge/Status-Beta-yellow.svg)](#status)

[Quickstart](#quickstart) · [API](#api-reference) · [Benchmarks](#performance) · [Correctness](#correctness) · [Changelog](https://github.com/kuntal-devrat/precise-numpy/releases)

</div>

---

## What You Can Verify

Need guaranteed error bounds on numerical computing? Browse the **capability grid** below — every operation returns a rigorous `(midpoint, radius)` enclosure over the true mathematical value.

| Domain | What it does | Status |
|--------|-------------|--------|
| Core arithmetic | `+ - * / ** @`, in-place, broadcasting, scalar both sides | Available |
| Math functions | `sin, cos, tan, exp, ln, log2, log10, sqrt, abs, floor, ceil, trunc, round, sign, clip, power` | Available |
| Reductions | `sum, mean, prod, var, std, min, max, argmin, argmax, all, any` — with optional `axis` | Available |
| Linear algebra | `det, inv, solve, lstsq, pinv, cholesky, matrix_power, cond, matrix_rank` | Available |
| Eigen / SVD | `eig` (symmetric + general), `svd` — rigorous eigenvalue, eigenvector, singular-value and singular-vector enclosures | Available |
| Norms | `norm(a, ord, axis)` — numpy-compatible `ord` in {None/2/'fro', 1, -1, inf, -inf, 'nuc'} | Available |
| Random | `rand, randn, randint, uniform, normal, random_sample` — xoshiro256\*\*, reproducible | Available |
| Interop | `__array_ufunc__` (NEP-13), `__array_function__` (NEP-18), `to_numpy`, `from_numpy`, `save_npy`, `load_npy`, `astype` | Available |
| I/O | `.pn` (binary mid+rad), `.npy` (numpy-compatible), `.txt` (text with shape header) | Available |

---

## Stack

| Piece | What it does |
|-------|-------------|
| Rust core | All arithmetic kernels, SIMD-amenable SoA buffer, interval primitives with directed rounding |
| GEMM radius pass | 4 parallel `dgemm` calls (midpoint + 3 radius terms) + one global RTN inflation; 1000×1000 matmul ≈ 0.14s |
| Exact error tracking | FMA residuals + TwoSum per-product rounding errors; no silent rounding loss |
| Rayon parallelism | Row-block decomposition over threads; GIL released during long computations |
| PyO3 abi3 wheel | CPython >= 3.10, single wheel for Linux/macOS/Windows |

---

## precise-numpy vs NumPy

| | precise-numpy | NumPy |
|---|---|---|
| Element type | Interval `(midpoint, radius)` — two `f64` per element | `float64` — one `f64` per element |
| Memory per element | 16 bytes | 8 bytes |
| Arithmetic result | Rigorous enclosure guaranteed | Floating-point result; no error bound |
| matmul (1000³) | ≈ 0.14s (4 dgemm passes) | ≈ 0.02s (1 dgemm pass) |
| Writeable views | Not supported (copy on write) | Full strided views, in-place ops |
| Drop-in replacement | No — result types are intervals | Yes — returns `float64` |
| Guarantee | Every result contains the true mathematical value | No formal guarantee |

---

## Installation

```bash
pip install precise-numpy
```

Requires Python >= 3.10. Build from source:

```bash
git clone https://github.com/kuntal-devrat/precise-numpy.git
cd precise-numpy
maturin develop --release
```

---

## Quick Start

```python
import precise_numpy as pnp
import numpy as np

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

# NumPy ufuncs work via NEP-13
r = np.exp(pnp.array([0.0, 1.0]))
print(r.values())  # [1.0, e]

# np.linalg works via NEP-18
A = pnp.array([[1.0, 2.0], [3.0, 4.0]])
print(np.linalg.norm(A))           # Frobenius norm: (mid, rad)
print(np.linalg.eigvals(A))        # eigenvalue intervals
print(np.linalg.svd(A))            # U, singular values, Vt (all interval)
```

---

## API Reference

### Array creation

`array(values, error=0.0)`, `asarray`, `zeros(shape)`, `ones`, `empty`, `full(shape, value, error=0.0)`,
`eye(n, m=None, k=0)`, `identity(n)`, `diag(v)`, `arange(start, stop, step=1.0)`,
`linspace(start, stop, num=50, endpoint=True)`, `from_raw_parts(midpoints, radii, shape)`.

`empty(shape)` zeros memory on purpose — uninitialised radii would violate the rigorous enclosure guarantee.

`to_numpy(a, dtype=None)`, `from_numpy(x, error=scalar_or_array)`, `astype(a, dtype)`, `save_npy(path, arr)`, `load_npy(path)`.

### Properties & conversions

`shape`, `ndim`, `size`, `dtype` (`"interval64"`), `itemsize` (16), `nbytes`, `strides`, `t`,
`values()`, `radii()`, `tolist()`, `get(i)`, `item(i)`, `midpoint(i)`, `radius(i)`,
`copy()`, `flatten()`, `ravel()`, `reshape(*shape)`, `transpose()`, `sort()`, `argsort()`.

`__array__(dtype=None)` (NEP-18) — returns a plain `float64` ndarray of midpoints so `np.asarray(pnp_array)` and `np + pnp_array` work naturally.

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

`sum, mean, prod, var, std, min, max, argmin, argmax, all, any` — each with an optional `axis`;
plus `min_val()`, `max_val()`, `cumsum(axis=None)`, `norm(ord=None, axis=None)`
with numpy-compatible `ord` in {None/2/'fro', 1, -1, inf, -inf, 'nuc'} and int `axis`.

### Stacking & selection

`concatenate(arrays, axis=0)`, `stack`, `vstack`, `hstack`, `split(a, sections_or_indices, axis=0)`,
`where(condition, x, y)`, `nonzero(a)` (tuple of index lists per axis).

### Linear algebra (`precise_numpy.linalg`)

`det`, `inv`, `solve(A, b)`, `lstsq(A, b)`, `pinv`, `eig`, `svd`, `norm`, `cholesky`, `matrix_power`, `matrix_rank`, `cond`.

**Rigor notes:**
`eig_symmetric` (Jacobi) — rigorous eigenvalue **and** eigenvector enclosures via residual bounds + Davis–Kahan.
`eig_general` (Hessenberg QR + Schur back-substitution) — rigorous eigenvalue radii via Bauer–Fike; eigenvector radii via perturbation bound (bounded by 1.0 at defective limit; `inf` for fully defective matrices).
`svd` — rigorous singular-value radii (cluster hull for degenerate clusters) + Davis–Kahan singular-vector radii.
`pinv` — interval reciprocals of singular values, sound for all conditioning levels.
`solve` — rigorous LU with partial pivoting + forward/back substitution.

### Random (`precise_numpy.random`)

`seed`, `rand(*size)`, `random_sample(*size)`, `random(*size)`, `randn(*size)`,
`randint(low, high, *size)`, `uniform(low, high, *size)`, `normal(loc=0.0, scale=1.0, *size)`.
Deterministic xoshiro256\*\*; `randint` scalar returns a Python `int`.

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

---

## Semantics

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
- **NEP-13 ufunc**: `np.add`, `np.multiply`, `np.negative`, `np.absolute`, `np.exp`, `np.log`,
  `np.sqrt`, `np.sin`, `np.cos`, `np.tan`, `np.isnan`, `np.isinf`, `np.isfinite`, `np.rint`,
  `np.sign`, `np.matmul` all work on `IntervalArray`.
- **NEP-18 function**: `np.linalg.norm`, `np.linalg.inv`, `np.linalg.solve`, `np.linalg.svd`,
  `np.linalg.eig`, `np.linalg.eigvals`, `np.linalg.pinv`, `np.sum`, `np.mean`, `np.std`, `np.var`,
  `np.max`, `np.min`, `np.all`, `np.any`, `np.concatenate`, `np.stack`, etc.

---

## Status

| Axis | State |
|------|-------|
| Correctness | 98 Rust + 97 Python tests. All rigor proofs hold. Degenerate inputs → `∞` radius or `ValueError`, never fake precision. |
| Performance | matmul 1000×1000: ~0.14s. Element-wise: ~1.1–2× vs numpy. 50 random triplets verified rigorous vs `float128`. |
| API surface | Creation, arithmetic, reductions, broadcasting, indexing, linalg, random, I/O, NEP-13/18 interop |
| Missing vs numpy | Complex intervals, `float32` intervals, writeable views, `np.fft`, `np.partition`, `np.take`, many ufuncs (`tanh`, `arccos`, `erf`) |
| Classification | **Beta** — not Alpha. No known correctness holes. Feature gaps remain. |

---

## Testing

```bash
cargo test                    # Rust unit tests
python -m unittest tests/python/test_api.py   # Python integration tests
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.