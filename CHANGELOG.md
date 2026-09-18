# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-09-18

### Fixed

- **Soundness:** Corrected TwoSum subtraction error residual in `sub_intervals_rigorous` to use `-bm` for exact TwoDiff residual evaluation, preventing under-estimated radii during cancellation.
- **Soundness:** Fixed matrix multiplication `matmul` radius accumulation to compute the full $R_A \cdot R_B$ matrix product via parallel DGEMM instead of accumulating only from the first row of $A$.
- **NumPy Interop:** Fixed `np.squeeze` / `squeeze` axis parameter handling to conform to NumPy standards (supports `axis=None`, `int`, `tuple`/`list`, negative axes, and raises `ValueError` for non-singleton dimensions).
- **Math:** Corrected `exp2` ufunc to compute $2^x$ (`2.0 ** a`) instead of $e^{2x}$.
- **Hardening:** Updated `rollaxis` to return `NotImplemented` to avoid invalid axis transpositions.

### Added

- Added `cumsum`, `squeeze`, and `exp2` top-level module exports and NEP-18 bindings.
- Added `IntervalArray.squeeze()` method.
- Added regression test suites for subtraction residual, matmul per-row radius accumulation, squeeze, and symbol exports.

### Performance

- Parallelized `add_scalar` and `mul_scalar` across threads using Rayon chunks for arrays $\ge 32,768$ elements.
- Cleaned unused `shape_of` and `shape_of_scalar` dead code.

## [1.0.0] - 2025-06-12

### Added

- Production release: first stable API.
- Missing `LICENSE` file (MIT).
- `SECURITY.md` and full project metadata.
- Hardened CI (lint, formatting, coverage) and release pipeline.

### Fixed

- `None`/newaxis setitem no longer panics (`a[None, 0] = 9`).
- Scalar division by a non-power-of-two now keeps the reciprocal rounding
  error in the radius (rigorous enclosure).
- `maximum`/`minimum` now propagate NaN/empty intervals instead of silently
  discarding them.
- Removed duplicated NEP-18 registrations in the Python layer.

### Changed

- Version bumped to `1.0.0`.
- Development Status classifier set to `5 - Production/Stable`.

[unreleased]: https://github.com/kuntal-devrat/precise-numpy/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/kuntal-devrat/precise-numpy/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/kuntal-devrat/precise-numpy/releases/tag/v1.0.0
