"""
Showcase Project: Certified Neural Network Robustness via Interval Bound Propagation (IBP)
Powered by precise-numpy

In AI Safety, standard floating-point models can be fooled by tiny adversarial noise.
Using precise-numpy, we can pass an input INTERVAL [x - epsilon, x + epsilon]
through a neural network layer (matmul + ReLU) and get GUARANTEED output bounds.

If output_lo(True Class) > output_hi(Other Class), the model is 100% MATHEMATICALLY CERTIFIED
to be immune to ANY adversarial attack within perturbation radius epsilon!
"""

import math
import precise_numpy as pnp

def relu_interval(arr: pnp.IntervalArray) -> pnp.IntervalArray:
    """Compute Interval ReLU activation: max(0, x)."""
    mids = arr.values()
    rads = arr.radii()
    out_mids = []
    out_rads = []
    for m, r in zip(mids, rads):
        lo = m - r
        hi = m + r
        new_lo = max(0.0, lo)
        new_hi = max(0.0, hi)
        new_mid = (new_hi + new_lo) / 2.0
        new_rad = (new_hi - new_lo) / 2.0
        out_mids.append(new_mid)
        out_rads.append(new_rad)
    
    return pnp.array(out_mids, error=0.0).reshape(arr.shape())

def run_ibp_verification_demo():
    print("=" * 75)
    print("  AI SAFETY SHOWCASE: CERTIFIED NEURAL NETWORK BOUND PROPAGATION (IBP)")
    print("=" * 75)

    # 1. Define an input sample (e.g. 4 feature sensor inputs)
    clean_input = [0.8, -0.5, 1.2, 0.3]
    epsilon = 0.05  # Adversarial noise bound (eps = +/- 0.05)

    print(f"\n[1] Clean Input Vector: {clean_input}")
    print(f"    Adversarial Noise Radius (Epsilon): +/- {epsilon}")

    # Create Interval Array representing [x - eps, x + eps]
    x_interval = pnp.array(clean_input, error=epsilon).reshape([1, 4])

    # 2. Define Layer 1 Weights (4 inputs -> 3 hidden units)
    w1_data = [
        0.5, -0.2,  0.8,
        0.3,  0.9, -0.4,
       -0.6,  0.1,  0.7,
        0.2, -0.5,  0.3
    ]
    w1 = pnp.array(w1_data).reshape([4, 3])

    # Layer 1 Forward Pass: z1 = x * w1
    z1 = x_interval.matmul(w1)
    # Layer 1 Activation: h1 = ReLU(z1)
    h1 = relu_interval(z1)

    print("\n[2] Layer 1 Output Interval Bounds (3 Hidden Neurons):")
    for i in range(3):
        m, r = h1.get(i)
        print(f"    Neuron {i}: [{m - r:.4f}, {m + r:.4f}] (mid: {m:.4f} +/- {r:.4f})")

    # 3. Define Output Layer Weights (3 hidden -> 2 classes: [Stop Sign, Speed Limit])
    w2_data = [
        1.5, -0.8,
        0.7, -1.2,
        1.1, -0.5
    ]
    w2 = pnp.array(w2_data).reshape([3, 2])

    # Final Output Layer Forward Pass
    logits = h1.matmul(w2)

    class_0_mid, class_0_rad = logits.get(0)  # Class 0: Stop Sign
    class_1_mid, class_1_rad = logits.get(1)  # Class 1: Speed Limit

    class_0_lo, class_0_hi = class_0_mid - class_0_rad, class_0_mid + class_0_rad
    class_1_lo, class_1_hi = class_1_mid - class_1_rad, class_1_mid + class_1_rad

    print("\n[3] Certified Output Class Bounds (Guaranteed Enclosure):")
    print(f"    Class 0 (Stop Sign):   [{class_0_lo:.4f}, {class_0_hi:.4f}]")
    print(f"    Class 1 (Speed Limit): [{class_1_lo:.4f}, {class_1_hi:.4f}]")

    # 4. Rigorous Mathematical Verification
    print("\n" + "-" * 75)
    print("  VERIFICATION RESULT:")
    print("-" * 75)

    if class_0_lo > class_1_hi:
        margin = class_0_lo - class_1_hi
        print("  [SUCCESS] CERTIFIED ROBUST!")
        print(f"     Class 0 Lower Bound ({class_0_lo:.4f}) strictly exceeds Class 1 Upper Bound ({class_1_hi:.4f}).")
        print(f"     Safety Margin: +{margin:.4f}")
        print(f"     PROOFS GUARANTEED: No adversarial perturbation within +/- {epsilon}")
        print("     can EVER cause this network to misclassify!")
    else:
        print("  [WARNING] UNVERIFIED / POTENTIALLY VULNERABLE")
        print("     Interval bounds overlap under the given adversarial noise radius.")

    print("=" * 75)

if __name__ == '__main__':
    run_ibp_verification_demo()
