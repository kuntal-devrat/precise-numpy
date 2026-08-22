# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[unreleased]: https://github.com/kuntal-devrat/precise-numpy/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kuntal-devrat/precise-numpy/releases/tag/v1.0.0
