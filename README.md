# precise-numpy

High-performance NumPy-compatible interval arrays with guaranteed numerical error bounds, powered by Rust SIMD.

Every element carries an error bound (`midpoint ± radius`), and every operation propagates these bounds correctly using hardware-directed rounding.

## Install

```bash
pip install precise-numpy
```

## Quick Start

```python
import precise_numpy as pnp

a = pnp.array([1.0, 2.0, 3.0])
b = pnp.array([4.0, 5.0, 6.0])
c = a + b
print(c)               # IntervalArray([5.0, 7.0, 9.0])
print(c.max_relative_error())  # 0.0
```

## Features

- SIMD-accelerated interval arithmetic (AVX2/AVX512/NEON)
- Hardware-directed rounding for provably correct bounds
- NumPy-like API with zero-copy conversions
- Parallel execution via Rayon

## License

MIT
