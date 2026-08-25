import time, sys, numpy as np, precise_numpy as pnp

def bench(fn, warmup=3, repeat=7):
    for _ in range(warmup): fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - start)
    times.sort()
    return times[len(times)//2]

def fmt(ns):
    if ns < 1000: return f'{ns:.0f}ns'
    if ns < 1e6: return f'{ns/1e3:.1f}us'
    if ns < 1e9: return f'{ns/1e6:.1f}ms'
    return f'{ns/1e9:.2f}s'

sep = '-' * 90
print('=' * 90)
print('  COMPREHENSIVE BENCHMARK: precise-numpy vs NumPy')
print('  Python', sys.version)
print('=' * 90)

header = f'{"Size":>10} | {"Op":>12} | {"NumPy":>10} | {"precise-numpy":>14} | {"Ratio":>8} | {"Verdict":>10}'

# ── ELEMENT-WISE OPS ──
print(f'\n{sep}')
print('  ELEMENT-WISE OPERATIONS')
print(sep)
print(header)
print(sep)

sizes = [1_000, 10_000, 100_000, 1_000_000]
for n in sizes:
    np_a = np.random.rand(n)
    np_b = np.random.rand(n)
    pnp_a = pnp.array(np_a.tolist())
    pnp_b = pnp.array(np_b.tolist())
    ops = [
        ('Add', lambda a=np_a, b=np_b: a + b, lambda a=pnp_a, b=pnp_b: a + b),
        ('Subtract', lambda a=np_a, b=np_b: a - b, lambda a=pnp_a, b=pnp_b: a - b),
        ('Multiply', lambda a=np_a, b=np_b: a * b, lambda a=pnp_a, b=pnp_b: a * b),
    ]
    for i, (name, np_fn, pnp_fn) in enumerate(ops):
        np_t = bench(np_fn)
        pnp_t = bench(pnp_fn)
        ratio = pnp_t / np_t
        verdict = 'CLOSE' if ratio < 3 else 'GAP' if ratio < 10 else 'BIG GAP'
        prefix = f'{n:>10,}' if i == 0 else ''
        print(f'{prefix} | {name:>12} | {fmt(np_t):>10} | {fmt(pnp_t):>14} | {ratio:>7.2f}x | {verdict:>10}')
    if n < sizes[-1]:
        print(sep)

# ── MATH FUNCTIONS ──
print(f'\n{sep}')
print('  MATH FUNCTIONS')
print(sep)
print(header)
print(sep)

for n in sizes:
    np_a = np.random.rand(n) * 2.0
    pnp_a = pnp.array(np_a.tolist())
    ops = [
        ('sin', lambda a=np_a: np.sin(a), lambda a=pnp_a: a.sin()),
        ('exp', lambda a=np_a: np.exp(a), lambda a=pnp_a: a.exp()),
        ('sqrt', lambda a=np_a: np.sqrt(a), lambda a=pnp_a: a.sqrt()),
    ]
    for i, (name, np_fn, pnp_fn) in enumerate(ops):
        np_t = bench(np_fn)
        pnp_t = bench(pnp_fn)
        ratio = pnp_t / np_t
        verdict = 'CLOSE' if ratio < 3 else 'GAP' if ratio < 10 else 'BIG GAP'
        prefix = f'{n:>10,}' if i == 0 else ''
        print(f'{prefix} | {name:>12} | {fmt(np_t):>10} | {fmt(pnp_t):>14} | {ratio:>7.2f}x | {verdict:>10}')
    if n < sizes[-1]:
        print(sep)

# ── REDUCTIONS ──
print(f'\n{sep}')
print('  REDUCTIONS')
print(sep)
print(header)
print(sep)

for n in sizes:
    np_a = np.random.rand(n)
    pnp_a = pnp.array(np_a.tolist())
    ops = [
        ('sum', lambda a=np_a: a.sum(), lambda a=pnp_a: a.sum()),
        ('mean', lambda a=np_a: a.mean(), lambda a=pnp_a: a.mean()),
    ]
    for i, (name, np_fn, pnp_fn) in enumerate(ops):
        np_t = bench(np_fn)
        pnp_t = bench(pnp_fn)
        ratio = pnp_t / np_t
        verdict = 'CLOSE' if ratio < 3 else 'GAP' if ratio < 10 else 'BIG GAP'
        prefix = f'{n:>10,}' if i == 0 else ''
        print(f'{prefix} | {name:>12} | {fmt(np_t):>10} | {fmt(pnp_t):>14} | {ratio:>7.2f}x | {verdict:>10}')
    if n < sizes[-1]:
        print(sep)

# ── MATRIX MULTIPLY ──
print(f'\n{sep}')
print('  MATRIX MULTIPLY')
print(sep)
print(header)
print(sep)

for sz in [64, 128, 256]:
    np_m1 = np.random.rand(sz, sz)
    np_m2 = np.random.rand(sz, sz)
    pnp_m1 = pnp.array(np_m1.flatten().tolist()).reshape([sz, sz])
    pnp_m2 = pnp.array(np_m2.flatten().tolist()).reshape([sz, sz])
    np_t = bench(lambda m1=np_m1, m2=np_m2: m1 @ m2)
    pnp_t = bench(lambda m1=pnp_m1, m2=pnp_m2: m1.matmul(m2))
    ratio = pnp_t / np_t
    verdict = 'CLOSE' if ratio < 3 else 'GAP' if ratio < 10 else 'BIG GAP'
    label = f'{sz}x{sz}'
    print(f'{label:>10} | {"matmul":>12} | {fmt(np_t):>10} | {fmt(pnp_t):>14} | {ratio:>7.2f}x | {verdict:>10}')

print(sep)
