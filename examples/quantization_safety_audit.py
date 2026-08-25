"""
Precision Auditor: Floating-Point Drift & Quantization Safety Scanner
Powered by precise-numpy

Detects exact lines of code where floating-point drift, catastrophic cancellation,
or numerical precision degradation occurs in AI models or scientific pipelines.
"""

import numpy as np
import precise_numpy as pnp

class PrecisionAuditor:
    def __init__(self, name="Numerical Model"):
        self.name = name
        self.steps = []

    def audit_step(self, step_name: str, np_result: np.ndarray, pnp_result: pnp.IntervalArray):
        """Compare standard NumPy float64 execution against precise-numpy error bounds."""
        max_rel_err = pnp_result.max_relative_error()
        max_rad = pnp_result.max_radius()
        
        # Calculate drift between float64 value and interval midpoint
        pnp_mids = np.array(pnp_result.values())
        max_drift = np.max(np.abs(np_result.flatten() - pnp_mids))

        status = "OK"
        if max_rel_err > 1e-3:
            status = "CRITICAL DRIFT"
        elif max_rel_err > 1e-6:
            status = "WARNING"

        self.steps.append({
            "step": step_name,
            "max_rel_error": max_rel_err,
            "max_radius": max_rad,
            "max_drift": max_drift,
            "status": status,
        })

    def print_report(self):
        print("\n" + "=" * 80)
        print(f"  PRECISION AUDIT REPORT: {self.name}")
        print("=" * 80)
        print(f"  {'Step / Operation':<30} | {'Max Rel Error':<15} | {'Max Radius':<12} | {'Status':<15}")
        print("-" * 80)

        for s in self.steps:
            err_str = f"{s['max_rel_error']:.2e}" if s['max_rel_error'] < float('inf') else "INF"
            rad_str = f"{s['max_radius']:.2e}"
            print(f"  {s['step']:<30} | {err_str:<15} | {rad_str:<12} | [{s['status']}]")

        print("=" * 80)
        
        warnings = [s for s in self.steps if s['status'] != "OK"]
        if warnings:
            print(f"\n  [ALERT] Found {len(warnings)} numerical instability warnings!")
            for w in warnings:
                print(f"   -> Step '{w['step']}': Relative Error = {w['max_rel_error']:.4e}")
        else:
            print("\n  [SUCCESS] All pipeline steps numerically stable within bounds.")
        print("=" * 80)


def run_quantization_safety_demo():
    print("Initializing Model Precision Audit...")
    auditor = PrecisionAuditor("Transformer Attention Layer (FP64 vs Interval)")

    # Simulated batch: 128 sequences, 64 hidden dim
    np.random.seed(42)
    X_raw = np.random.randn(128, 64)
    W_q_raw = np.random.randn(64, 64) * 0.1
    W_k_raw = np.random.randn(64, 64) * 0.1

    # 1. Inputs (with FP16 quantization noise estimation error = 1e-4)
    X_np = X_raw
    X_pnp = pnp.array(X_raw.flatten().tolist(), error=1e-4).reshape([128, 64])
    auditor.audit_step("1. Quantized Inputs", X_np, X_pnp)

    # 2. Query Projection (Matmul)
    W_q_pnp = pnp.array(W_q_raw.flatten().tolist()).reshape([64, 64])
    Q_np = X_np @ W_q_raw
    Q_pnp = X_pnp.matmul(W_q_pnp)
    auditor.audit_step("2. Query Projection (Matmul)", Q_np, Q_pnp)

    # 3. Key Projection (Matmul)
    W_k_pnp = pnp.array(W_k_raw.flatten().tolist()).reshape([64, 64])
    K_np = X_np @ W_k_raw
    K_pnp = X_pnp.matmul(W_k_pnp)
    auditor.audit_step("3. Key Projection (Matmul)", K_np, K_pnp)

    # 4. Attention Scores (Q @ K.T)
    K_T_pnp = K_pnp.transpose()
    Scores_np = Q_np @ K_np.T
    Scores_pnp = Q_pnp.matmul(K_T_pnp)
    auditor.audit_step("4. Attention Scores (Q @ K^T)", Scores_np, Scores_pnp)

    # 5. Scaling by 1 / sqrt(d_k)
    scale = 1.0 / np.sqrt(64.0)
    Scaled_np = Scores_np * scale
    Scaled_pnp = Scores_pnp.scale(scale)
    auditor.audit_step("5. Scale (1 / sqrt(64))", Scaled_np, Scaled_pnp)

    # Print Final Report
    auditor.print_report()

if __name__ == '__main__':
    run_quantization_safety_demo()
