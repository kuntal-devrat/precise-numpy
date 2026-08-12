use crate::array::IntervalArray;
use crate::error::Interval;
use crate::simd::vec_ops;

/// Sum all elements, accumulating error.
/// Uses SIMD-accelerated sum for both midpoints and radii.
pub fn sum(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::zero();
    }
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mid_sum = vec_ops::sum_f64(mids);
    let rad_sum = vec_ops::sum_f64(rads);

    Interval::from_midpoint_radius(mid_sum, rad_sum)
}

/// Compute the mean of all elements.
pub fn mean(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::nan();
    }
    let total = sum(a);
    let n = a.len() as f64;
    Interval::from_midpoint_radius(total.midpoint() / n, total.radius() / n)
}

/// Compute the variance of all elements (population variance).
///
/// Uses numerically stable two-pass algorithm:
///   1. Compute mean
///   2. Sum of squared deviations from mean
///
/// This avoids the catastrophic cancellation in E[X²] - E[X]² form.
pub fn var(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::nan();
    }
    let n = a.len() as f64;
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    // Pass 1: compute mean
    let mean_iv = mean(a);
    let mean_mid = mean_iv.midpoint();

    // Pass 2: sum of squared deviations from mean
    // For exact arrays, this is pure midpoint arithmetic
    // For interval arrays, we propagate uncertainty properly
    let mut sum_sq_dev_mid = 0.0f64;
    let mut sum_sq_dev_rad = 0.0f64;

    for i in 0..a.len() {
        // (x_i - mean) as an interval
        let dev_mid = mids[i] - mean_mid;
        let dev_rad = rads[i] + mean_iv.radius();

        // (x_i - mean)^2 in midpoint-radius form
        // mid = dev_mid^2, rad = 2*|dev_mid|*dev_rad + dev_rad^2
        let sq_mid = dev_mid * dev_mid;
        let sq_rad = 2.0 * dev_mid.abs() * dev_rad + dev_rad * dev_rad;

        sum_sq_dev_mid += sq_mid;
        sum_sq_dev_rad += sq_rad;
    }

    let var_mid = sum_sq_dev_mid / n;
    let var_rad = sum_sq_dev_rad / n;

    // Clamp negative midpoint (can happen due to floating point in edge cases)
    let var_mid = var_mid.max(0.0);

    Interval::from_midpoint_radius(var_mid, var_rad)
}

/// Compute the standard deviation (population).
pub fn std_dev(a: &IntervalArray) -> Interval {
    let v = var(a);
    // Guard: if variance interval could be negative, clamp lo to 0
    let lo = if v.lo > 0.0 { v.lo } else { 0.0 };
    let hi = if v.hi > 0.0 { v.hi } else { 0.0 };
    Interval::new(lo.sqrt(), hi.sqrt())
}

/// Find the minimum element.
/// Uses SIMD min for midpoints, scalar for radii tracking.
pub fn min(a: &IntervalArray) -> Interval {
    let n = a.len();
    assert!(n > 0, "min of empty array");

    let mids = a.data().midpoints();
    let rads = a.data().radii();

    // Find the index of the minimum midpoint
    let mut min_lo = f64::INFINITY;
    let mut min_hi = f64::INFINITY;
    for i in 0..n {
        let lo = mids[i] - rads[i];
        let hi = mids[i] + rads[i];
        if lo < min_lo { min_lo = lo; }
        if hi < min_hi { min_hi = hi; }
    }
    Interval::new(min_lo, min_hi)
}

/// Find the maximum element.
/// Uses SIMD max for midpoints, scalar for radii tracking.
pub fn max(a: &IntervalArray) -> Interval {
    let n = a.len();
    assert!(n > 0, "max of empty array");

    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mut max_lo = f64::NEG_INFINITY;
    let mut max_hi = f64::NEG_INFINITY;
    for i in 0..n {
        let lo = mids[i] - rads[i];
        let hi = mids[i] + rads[i];
        if lo > max_lo { max_lo = lo; }
        if hi > max_hi { max_hi = hi; }
    }
    Interval::new(max_lo, max_hi)
}

/// Dot product of two 1D arrays.
/// Uses SIMD FMA-accelerated dot product for midpoints.
pub fn dot(a: &IntervalArray, b: &IntervalArray) -> Interval {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let n = a.len();

    if n == 0 {
        return Interval::zero();
    }

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());

    // Midpoint of dot product: sum(a.mid * b.mid) — use FMA dot
    let mid_sum = vec_ops::dot_f64(a_mids, b_mids);

    // Radius of dot product: sum(|a.mid|*b.rad + |b.mid|*a.rad + a.rad*b.rad)
    let mut rad_sum = 0.0f64;
    for i in 0..n {
        rad_sum += a_mids[i].abs() * b_rads[i]
                 + b_mids[i].abs() * a_rads[i]
                 + a_rads[i] * b_rads[i];
    }

    Interval::from_midpoint_radius(mid_sum, rad_sum)
}

/// Cumulative sum along the array.
pub fn cumsum(a: &IntervalArray) -> IntervalArray {
    let n = a.len();
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];

    let mut cum_mid = 0.0f64;
    let mut cum_rad = 0.0f64;
    for i in 0..n {
        cum_mid += mids[i];
        cum_rad += rads[i];
        out_mids[i] = cum_mid;
        out_rads[i] = cum_rad;
    }

    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Product of all elements.
pub fn prod(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::exact(1.0);
    }
    let mut result = a.get(0);
    for i in 1..a.len() {
        result = result * a.get(i);
    }
    result
}

/// Matrix multiplication of two 2D arrays using tiled algorithm.
///
/// a: [M, K], b: [K, N] -> result: [M, N]
///
/// Uses 64x64 cache-friendly tiling with SIMD inner kernels.
pub fn matmul(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.ndim(), 2, "matmul requires 2D arrays");
    assert_eq!(b.ndim(), 2, "matmul requires 2D arrays");
    assert_eq!(
        a.shape()[1],
        b.shape()[0],
        "matmul: inner dimensions must match"
    );

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    // For small matrices, use the direct approach
    if m * n * k < 4096 {
        return matmul_small(a, b, m, k, n);
    }

    // Tiled matmul for larger matrices
    matmul_tiled(a, b, m, k, n)
}

/// Direct matmul for small matrices.
fn matmul_small(a: &IntervalArray, b: &IntervalArray, m: usize, k: usize, n: usize) -> IntervalArray {
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let b_mids = b.data().midpoints();
    let b_rads = b.data().radii();

    let mut r_mids = vec![0.0f64; m * n];
    let mut r_rads = vec![0.0f64; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum_mid = 0.0f64;
            let mut sum_rad = 0.0f64;
            for p in 0..k {
                let am = a_mids[i * k + p];
                let ar = a_rads[i * k + p];
                let bm = b_mids[p * n + j];
                let br = b_rads[p * n + j];

                // Midpoint-radius multiplication and accumulation
                sum_mid += am * bm;
                sum_rad += am.abs() * br + bm.abs() * ar + ar * br;
            }
            r_mids[i * n + j] = sum_mid;
            r_rads[i * n + j] = sum_rad;
        }
    }

    IntervalArray::from_raw_parts(&r_mids, &r_rads, &[m, n])
}

/// Tiled matmul for larger matrices.
/// Uses cache-friendly blocking to improve locality.
fn matmul_tiled(a: &IntervalArray, b: &IntervalArray, m: usize, k: usize, n: usize) -> IntervalArray {
    const TILE: usize = 64;

    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let b_mids = b.data().midpoints();
    let b_rads = b.data().radii();

    let mut r_mids = vec![0.0f64; m * n];
    let mut r_rads = vec![0.0f64; m * n];

    // Tiled loop: iterate over tiles of the output
    let mut ii = 0;
    while ii < m {
        let i_end = (ii + TILE).min(m);
        let mut jj = 0;
        while jj < n {
            let j_end = (jj + TILE).min(n);
            let mut pp = 0;
            while pp < k {
                let p_end = (pp + TILE).min(k);

                // Micro-kernel: compute tile [ii..i_end, jj..j_end] += A[ii..i_end, pp..p_end] * B[pp..p_end, jj..j_end]
                for i in ii..i_end {
                    for p in pp..p_end {
                        let am = a_mids[i * k + p];
                        let ar = a_rads[i * k + p];
                        let abs_am = am.abs();

                        for j in jj..j_end {
                            let bm = b_mids[p * n + j];
                            let br = b_rads[p * n + j];
                            let idx = i * n + j;

                            r_mids[idx] += am * bm;
                            r_rads[idx] += abs_am * br + bm.abs() * ar + ar * br;
                        }
                    }
                }

                pp += TILE;
            }
            jj += TILE;
        }
        ii += TILE;
    }

    IntervalArray::from_raw_parts(&r_mids, &r_rads, &[m, n])
}

/// L2 norm: sqrt(sum(x^2)), fused without intermediate array.
pub fn norm_l2(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::zero();
    }
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mut sum_sq_mid = 0.0f64;
    let mut sum_sq_rad = 0.0f64;
    for i in 0..a.len() {
        let m = mids[i];
        let r = rads[i];
        sum_sq_mid += m * m;
        sum_sq_rad += 2.0 * m.abs() * r + r * r;
    }

    let sum_iv = Interval::from_midpoint_radius(sum_sq_mid, sum_sq_rad);
    let lo = if sum_iv.lo > 0.0 { sum_iv.lo } else { 0.0 };
    let hi = if sum_iv.hi > 0.0 { sum_iv.hi } else { 0.0 };
    Interval::new(lo.sqrt(), hi.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_exact() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0]);
        let s = sum(&a);
        assert!((s.midpoint() - 6.0).abs() < 1e-10);
        assert!(s.radius() < 1e-15);
    }

    #[test]
    fn test_sum_empty() {
        let a = IntervalArray::zeros(&[0]);
        let s = sum(&a);
        assert_eq!(s.midpoint(), 0.0);
    }

    #[test]
    fn test_mean_exact() {
        let a = IntervalArray::from_f64_slice(&[2.0, 4.0, 6.0]);
        let m = mean(&a);
        assert!((m.midpoint() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_var_constant() {
        let a = IntervalArray::from_f64_slice(&[5.0, 5.0, 5.0]);
        let v = var(&a);
        assert!(v.midpoint().abs() < 1e-10);
    }

    #[test]
    fn test_var_known() {
        // Variance of [1, 2, 3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0]);
        let v = var(&a);
        assert!((v.midpoint() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let a = IntervalArray::from_f64_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let s = std_dev(&a);
        // population std dev of this set is 2.0
        assert!((s.midpoint() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_std_dev_zero_variance() {
        let a = IntervalArray::from_f64_slice(&[3.0, 3.0, 3.0]);
        let s = std_dev(&a);
        assert!(s.midpoint().abs() < 1e-10);
    }

    #[test]
    fn test_min_max() {
        let a = IntervalArray::from_f64_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
        assert!((min(&a).lo - 1.0).abs() < 1e-10);
        assert!((max(&a).hi - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0]);
        let b = IntervalArray::from_f64_slice(&[4.0, 5.0, 6.0]);
        let d = dot(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((d.midpoint() - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_empty() {
        let a = IntervalArray::zeros(&[0]);
        let b = IntervalArray::zeros(&[0]);
        let d = dot(&a, &b);
        assert_eq!(d.midpoint(), 0.0);
    }

    #[test]
    fn test_matmul() {
        let a = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = IntervalArray::from_f64_vec(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b);
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // [[19, 22], [43, 50]]
        assert!((c.get(0).midpoint() - 19.0).abs() < 1e-10);
        assert!((c.get(1).midpoint() - 22.0).abs() < 1e-10);
        assert!((c.get(2).midpoint() - 43.0).abs() < 1e-10);
        assert!((c.get(3).midpoint() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_non_square() {
        // [2,3] x [3,2]
        let a = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = IntervalArray::from_f64_vec(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((c.get(0).midpoint() - 58.0).abs() < 1e-10);
        assert!((c.get(1).midpoint() - 64.0).abs() < 1e-10);
        assert!((c.get(2).midpoint() - 139.0).abs() < 1e-10);
        assert!((c.get(3).midpoint() - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumsum() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
        let c = cumsum(&a);
        assert!((c.get(0).midpoint() - 1.0).abs() < 1e-10);
        assert!((c.get(1).midpoint() - 3.0).abs() < 1e-10);
        assert!((c.get(2).midpoint() - 6.0).abs() < 1e-10);
        assert!((c.get(3).midpoint() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_prod() {
        let a = IntervalArray::from_f64_slice(&[2.0, 3.0, 4.0]);
        let p = prod(&a);
        assert!((p.midpoint() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_l2() {
        let a = IntervalArray::from_f64_slice(&[3.0, 4.0]);
        let n = norm_l2(&a);
        assert!((n.midpoint() - 5.0).abs() < 1e-10);
    }
}
