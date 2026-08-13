import math
import precise_numpy as pnp

def verify_rigorous_precision():
    print("=" * 80)
    print("  RIGOROUS NUMERICAL PRECISION AUDIT")
    print("=" * 80)

    # Test 1: Division Directed Rounding Enclosure
    # Mathematically: 1.0 / 3.0 = 0.3333333333333333...
    # We want to make sure the interval contains the infinite expansion.
    a = pnp.array([1.0], error=0.0)
    b = pnp.array([3.0], error=0.0)
    c = a / b
    mid, rad = c.get(0)
    lo = mid - rad
    hi = mid + rad
    true_val = 1.0 / 3.0
    
    print("[Test 1] Enclosure of 1/3:")
    print(f"  Calculated Interval: [{lo:.20f}, {hi:.20f}]")
    print(f"  True value:           {true_val:.20f}")
    assert lo <= true_val <= hi, "Error: Enclosure failed!"
    print("  Status: PASSED (Enclosure guaranteed)")

    # Test 2: Catastrophic Drift Prevention
    # Compute: sum of 0.1 ten times.
    # Standard float64 accumulates error: 0.1 * 10 != 1.0
    np_sum = sum([0.1] * 10)
    pnp_arr = pnp.array([0.1] * 10, error=0.0)
    pnp_mid, pnp_rad = pnp_arr.sum()
    pnp_lo = pnp_mid - pnp_rad
    pnp_hi = pnp_mid + pnp_rad
    
    print("\n[Test 2] Catastrophic drift comparison (summing 0.1 ten times):")
    print(f"  Standard Float64 Sum: {np_sum:.20f} (Difference from 1.0: {np_sum - 1.0:.2e})")
    print(f"  precise-numpy Sum:    {pnp_mid:.20f} +/- {pnp_rad:.2e}")
    print(f"  Enclosure: [{pnp_lo:.20f}, {pnp_hi:.20f}]")
    assert pnp_lo <= 1.0 <= pnp_hi, "Error: Enclosure failed!"
    print("  Status: PASSED")

    # Test 3: Trig Monotonicity Peaks
    # Interval spanning pi/2. sin(x) should peak at 1.0.
    # If we evaluate sin([1.4, 1.7]), since it crosses pi/2 (~1.57), the upper bound MUST be exactly 1.0.
    a = pnp.array([1.55], error=0.15) # interval [1.4, 1.7]
    sin_a = a.sin()
    mid, rad = sin_a.get(0)
    lo, hi = mid - rad, mid + rad
    print("\n[Test 3] Monotonicity Peak check for sin(x) across pi/2:")
    print(f"  Input Interval: [1.4, 1.7]")
    print(f"  sin(Input) Enclosure: [{lo:.6f}, {hi:.6f}]")
    assert abs(hi - 1.0) < 1e-15 or hi >= 1.0, "Error: sin(x) upper bound should enclose peak at 1.0!"
    print("  Status: PASSED")

    # Test 4: Division Singularity Check
    # Division by an interval containing zero yields NaN interval (undefined/entire representation).
    a = pnp.array([1.0], error=0.0)
    b = pnp.array([0.0], error=0.1) # [-0.1, 0.1]
    res = a / b
    mid, rad = res.get(0)
    print("\n[Test 4] Division by interval containing zero:")
    print(f"  Input: 1.0 / [-0.1, 0.1]")
    print(f"  Output Interval: [{mid}, {rad}]")
    assert math.isnan(mid) and math.isnan(rad), "Error: Division by zero crossing should return NaN interval!"
    print("  Status: PASSED")

    # Test 5: Out of Domain Protection
    # log of negative values should yield NaN/NaN.
    a = pnp.array([-5.0], error=1.0)
    res = a.ln()
    mid, rad = res.get(0)
    print("\n[Test 5] Natural logarithm of negative interval:")
    print(f"  Input: ln([-6.0, -4.0])")
    print(f"  Output Midpoint: {mid}, Radius: {rad}")
    assert math.isnan(mid) and math.isnan(rad), "Error: Should be NaN interval!"
    print("  Status: PASSED")

    print("\n" + "=" * 80)
    print("  ALL PRECISION AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    verify_rigorous_precision()
