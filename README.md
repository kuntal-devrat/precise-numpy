<div align="center">

  <img src="assets/logo.png" alt="precise-numpy logo" width="380" />

  # precise-numpy

  **NumPy-compatible interval arrays with guaranteed numerical error bounds, powered by Rust SIMD.**

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
  [![Version](https://img.shields.io/badge/version-0.1.0-emerald.svg)](#)

</div>

---

## Overview

`precise-numpy` is a high-performance Python library for numerical computing with provable error bounds. Standard floating-point arrays (`float64`) accumulate rounding error and catastrophic cancellation without diagnostic warnings. `precise-numpy` uses hardware FPU directed rounding (`MXCSR` on x86_64 and `FPCR` on ARM64) to track mathematical bounds ($midpoint \pm radius$) across every array operation.

Key capabilities:
- **Drop-in NumPy Ergonomics**: Familiar array operations, indexing, broadcasting, and reductions.
- **Hardware-Directed Rounding**: Enforces strict IEEE 754 lower/upper bounds for guaranteed mathematical enclosure.
- **Rust SIMD Acceleration**: Single-pass streaming vectorization (AVX-512, AVX2/FMA, ARM NEON) and parallel execution via Rayon.
- **BLAS Matrix Multiplication**: Assembly-tuned GEMM microkernels for matrix operations.

---

## Killer Use Case: Floating-Point Drift & AI Quantization Auditor

ML models and quantitative trading algorithms often exhibit unpredictable drift across machines or quantized precision levels (e.g. FP32 vs FP16). `precise-numpy` enables exact numerical audits of neural network layers and scientific pipelines.

```python
import numpy as np
import precise_numpy as pnp

# Audit a Transformer Attention Layer under input noise (error = 1e-4)
X_raw = np.random.randn(128, 64)
W_q_raw = np.random.randn(64, 64) * 0.1

# Wrap input data with interval error bounds
X_pnp = pnp.array(X_raw.flatten().tolist(), error=1e-4).reshape([128, 64])
W_q_pnp = pnp.array(W_q_raw.flatten().tolist()).reshape([64, 64])

# Query projection with propagated error bounds
Q_pnp = X_pnp.matmul(W_q_pnp)

# Check maximum relative error amplification across the layer
print("Max Relative Error:", Q_pnp.max_relative_error())
print("Max Radius Error:", Q_pnp.max_radius())
```

*(See complete runnable audit script in [`examples/quantization_safety_audit.py`](examples/quantization_safety_audit.py))*

---

## Benchmarks

Benchmarked against standard single-float `numpy` on Python 3.11 (Intel / AMD AVX2 + FMA).

> **Understanding the Benchmarks:**
> - Standard NumPy operates on single 64-bit float arrays (`float64`) without error tracking.
> - `precise-numpy` maintains two contiguous 64-bit float arrays ($midpoint$ and $radius$) per operation and executes hardware rounding mode switches to guarantee error bounds.
> - Small array operations ($\le 1,000$ elements) benefit from PyO3 C-extension dispatch speed. Large array operations ($\ge 100,000$ elements) use single-pass streaming SIMD loops.

### Element-Wise Operations

| Array Size | Operation | Standard NumPy | precise-numpy | Ratio vs NumPy |
| :--- | :--- | ---: | ---: | ---: |
| 1,000 | Add | 1.4 µs | 1.7 µs | 1.21x |
| 1,000 | Subtract | 1.4 µs | 1.6 µs | 1.14x |
| 1,000 | Multiply | 1.3 µs | 1.7 µs | 1.31x |
| 100,000 | Add | 343.4 µs | 658.4 µs | 1.92x |
| 100,000 | Subtract | 343.4 µs | 658.4 µs | 1.92x |
| 100,000 | Multiply | 399.3 µs | 769.8 µs | 1.93x |
| 1,000,000 | Add | 2.8 ms | 4.1 ms | 1.45x |
| 1,000,000 | Subtract | 2.9 ms | 3.9 ms | 1.34x |
| 1,000,000 | Multiply | 2.9 ms | 4.7 ms | 1.61x |

### Reductions & Math Functions

| Array Size | Operation | Standard NumPy | precise-numpy | Ratio vs NumPy |
| :--- | :--- | ---: | ---: | ---: |
| 1,000 | mean | 6.8 µs | 600 ns | **0.09x (11x faster)** |
| 1,000 | sum | 2.7 µs | 800 ns | **0.30x (3.3x faster)** |
| 10,000 | sin | 136.6 µs | 112.9 µs | **0.83x (1.2x faster)** |
| 1,000,000 | sum | 875.0 µs | 775.6 µs | **0.89x (Faster)** |
| 1,000,000 | mean | 838.3 µs | 804.0 µs | **0.96x (Faster)** |

### Matrix Multiplication (`matmul`)

| Matrix Shape | Standard NumPy | precise-numpy | Notes |
| :--- | ---: | ---: | :--- |
| 64 × 64 | 13.4 µs | 50.5 µs | Assembly GEMM microkernel |
| 128 × 128 | 199.9 µs | 575.6 µs | 2.88x ratio vs OpenBLAS |
| 256 × 256 | 407.8 µs | 3.8 ms | Decomposed dual-pass GEMM |

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

---

## Quick Start

```python
import precise_numpy as pnp

# Create interval arrays with error bounds
a = pnp.array([1.0, 2.0, 3.0], error=0.01)
b = pnp.array([4.0, 5.0, 6.0], error=0.02)

# Arithmetic operations automatically propagate bounds
c = a + b
print(c)
# Output: IntervalArray([5.0+/-0.03, 7.0+/-0.03, 9.0+/-0.03])

# Inspect relative error
print("Max relative error:", c.max_relative_error())

# Scalar operations
d = a * 2.5 + 10.0

# Reductions
mid, err = c.sum()
print(f"Sum = {mid} ± {err}")
```

---

## Technical Architecture

1. **Structure-of-Arrays (SoA) Buffer**: Midpoints and radii are stored in contiguous, 64-byte aligned memory chunks (`AlignedBuffer`), allowing direct SIMD vector loads without interleaved packing overhead.
2. **Single-Pass Streaming SIMD (`mul_intervals_stream`)**: Fuses midpoint product $a_{mid} \cdot b_{mid}$ and radius error bound $|a_{mid}| b_{rad} + |b_{mid}| a_{rad} + a_{rad} b_{rad}$ into a single SIMD pass using AVX-512 and AVX2+FMA registers.
3. **Zero-Copy Reference Counting**: `IntervalArray` uses `Arc<AlignedBuffer>`, enabling $O(1)$ zero-copy slicing, clones, and reshaping.
4. **GIL Release**: Long computations drop the Python GIL via `py.allow_threads()`, enabling parallel multi-threaded computing with Rayon.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
