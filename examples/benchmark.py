"""
Benchmark: precise-numpy vs NumPy
"""
import time
import sys

import precise_numpy as pnp
import numpy as np

def bench(label, fn, warmup=2, repeat=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    times.sort()
    return times[len(times) // 2]

def fmt_time(ns):
    if ns < 1_000:
        return f"{ns:.0f}ns"
    elif ns < 1_000_000:
        return f"{ns/1_000:.1f}us"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.1f}ms"
    else:
        return f"{ns/1_000_000_000:.2f}s"

def main():
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    print(f"{'='*70}")
    print(f"  precise-numpy vs NumPy Benchmark")
    print(f"  Python {sys.version}")
    print(f"{'='*70}")
    print(f"{'Size':>10} | {'Operation':>15} | {'NumPy':>10} | {'precise-numpy':>14} | {'Ratio':>8}")
    print(f"{'-'*70}")

    for n in sizes:
        np_a = np.random.rand(n).astype(np.float64)
        np_b = np.random.rand(n).astype(np.float64)
        pnp_a = pnp.array(np_a.tolist())
        pnp_b = pnp.array(np_b.tolist())

        ops = [
            ("Add", lambda a=np_a, b=np_b: a + b, lambda a=pnp_a, b=pnp_b: a + b),
            ("Subtract", lambda a=np_a, b=np_b: a - b, lambda a=pnp_a, b=pnp_b: a - b),
            ("Multiply", lambda a=np_a, b=np_b: a * b, lambda a=pnp_a, b=pnp_b: a * b),
        ]

        for i, (op_name, np_fn, pnp_fn) in enumerate(ops):
            np_time = bench(f"np_{op_name}", np_fn)
            pnp_time = bench(f"pnp_{op_name}", pnp_fn)
            ratio = pnp_time / np_time if np_time > 0 else float('inf')
            prefix = f"{n:>10,}" if i == 0 else f"{'':>10}"
            print(f"{prefix} | {op_name:>15} | {fmt_time(np_time):>10} | {fmt_time(pnp_time):>14} | {ratio:>7.2f}x")

        if n < sizes[-1]:
            print(f"{'-'*70}")

    print(f"{'='*70}")
    print("Note: precise-numpy includes error tracking overhead.")
    print("Every operation carries guaranteed numerical error bounds.")
    print("With AVX2/AVX512 SIMD, overhead decreases for larger arrays.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
