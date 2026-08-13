use rayon::prelude::*;
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
pub fn var(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::nan();
    }
    let n = a.len() as f64;
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mean_iv = mean(a);
    let mean_mid = mean_iv.midpoint();

    let mut sum_sq_dev_mid = 0.0f64;
    let mut sum_sq_dev_rad = 0.0f64;

    for i in 0..a.len() {
        let dev_mid = mids[i] - mean_mid;
        let dev_rad = rads[i] + mean_iv.radius();

        let sq_mid = dev_mid * dev_mid;
        let sq_rad = 2.0 * dev_mid.abs() * dev_rad + dev_rad * dev_rad;

        sum_sq_dev_mid += sq_mid;
        sum_sq_dev_rad += sq_rad;
    }

    let var_mid = (sum_sq_dev_mid / n).max(0.0);
    let var_rad = sum_sq_dev_rad / n;

    Interval::from_midpoint_radius(var_mid, var_rad)
}

/// Compute the standard deviation (population).
pub fn std_dev(a: &IntervalArray) -> Interval {
    let v = var(a);
    let lo = if v.lo > 0.0 { v.lo } else { 0.0 };
    let hi = if v.hi > 0.0 { v.hi } else { 0.0 };
    Interval::new(lo.sqrt(), hi.sqrt())
}

/// Find the minimum element.
pub fn min(a: &IntervalArray) -> Interval {
    let n = a.len();
    assert!(n > 0, "min of empty array");

    let mids = a.data().midpoints();
    let rads = a.data().radii();

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
pub fn dot(a: &IntervalArray, b: &IntervalArray) -> Interval {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let n = a.len();

    if n == 0 {
        return Interval::zero();
    }

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());

    let mid_sum = vec_ops::dot_f64(a_mids, b_mids);

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

/// Multi-threaded Parallel GEMM matrix multiplication (`matrixmultiply`).
///
/// a: [M, K], b: [K, N] -> result: [M, N]
///
/// Uses parallel row-block decomposition over Rayon threads + assembly GEMM microkernels.
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

    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let b_mids = b.data().midpoints();
    let b_rads = b.data().radii();

    let mut r_mids = vec![0.0f64; m * n];
    let mut r_rads = vec![0.0f64; m * n];

    let a_is_exact = a.is_exact();
    let b_is_exact = b.is_exact();

    // Precompute absolute value matrices if needed for radii propagation
    let abs_a_mids: Vec<f64> = if !b_is_exact {
        a_mids.iter().map(|&x| x.abs()).collect()
    } else {
        vec![]
    };

    let abs_b_plus_rad: Vec<f64> = if !a_is_exact {
        b_mids.iter().zip(b_rads.iter()).map(|(&bm, &br)| bm.abs() + br).collect()
    } else {
        vec![]
    };

    // Use parallel row-block decomposition only for larger matrices (M >= 256)
    if m >= 256 {
        let row_chunk_size = 64;
        r_mids.par_chunks_mut(row_chunk_size * n)
            .zip(r_rads.par_chunks_mut(row_chunk_size * n))
            .enumerate()
            .for_each(|(chunk_idx, (m_chunk, r_chunk))| {
                let row_start = chunk_idx * row_chunk_size;
                let current_m = m_chunk.len() / n;

                let a_mid_ptr = unsafe { a_mids.as_ptr().add(row_start * k) };
                let a_rad_ptr = unsafe { a_rads.as_ptr().add(row_start * k) };

                // 1. C_mid = A_mid * B_mid
                unsafe {
                    matrixmultiply::dgemm(
                        current_m, k, n,
                        1.0,
                        a_mid_ptr, k as isize, 1,
                        b_mids.as_ptr(), n as isize, 1,
                        0.0,
                        m_chunk.as_mut_ptr(), n as isize, 1,
                    );
                }

                // 2. C_rad = |A_mid| * B_rad + A_rad * (|B_mid| + B_rad)
                if !b_is_exact {
                    let abs_a_ptr = unsafe { abs_a_mids.as_ptr().add(row_start * k) };
                    unsafe {
                        matrixmultiply::dgemm(
                            current_m, k, n,
                            1.0,
                            abs_a_ptr, k as isize, 1,
                            b_rads.as_ptr(), n as isize, 1,
                            0.0,
                            r_chunk.as_mut_ptr(), n as isize, 1,
                        );
                    }
                }

                if !a_is_exact {
                    let beta = if !b_is_exact { 1.0 } else { 0.0 };
                    unsafe {
                        matrixmultiply::dgemm(
                            current_m, k, n,
                            1.0,
                            a_rad_ptr, k as isize, 1,
                            abs_b_plus_rad.as_ptr(), n as isize, 1,
                            beta,
                            r_chunk.as_mut_ptr(), n as isize, 1,
                        );
                    }
                }
            });
    } else {
        // Single-threaded fast path to avoid Rayon thread scheduling overhead for small/medium matrices
        unsafe {
            matrixmultiply::dgemm(
                m, k, n,
                1.0,
                a_mids.as_ptr(), k as isize, 1,
                b_mids.as_ptr(), n as isize, 1,
                0.0,
                r_mids.as_mut_ptr(), n as isize, 1,
            );
        }

        if !b_is_exact {
            unsafe {
                matrixmultiply::dgemm(
                    m, k, n,
                    1.0,
                    abs_a_mids.as_ptr(), k as isize, 1,
                    b_rads.as_ptr(), n as isize, 1,
                    0.0,
                    r_rads.as_mut_ptr(), n as isize, 1,
                );
            }
        }

        if !a_is_exact {
            let beta = if !b_is_exact { 1.0 } else { 0.0 };
            unsafe {
                matrixmultiply::dgemm(
                    m, k, n,
                    1.0,
                    a_rads.as_ptr(), k as isize, 1,
                    abs_b_plus_rad.as_ptr(), n as isize, 1,
                    beta,
                    r_rads.as_mut_ptr(), n as isize, 1,
                );
            }
        }
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
    fn test_matmul() {
        let a = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = IntervalArray::from_f64_vec(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b);
        assert!((c.get(0).midpoint() - 19.0).abs() < 1e-10);
        assert!((c.get(1).midpoint() - 22.0).abs() < 1e-10);
        assert!((c.get(2).midpoint() - 43.0).abs() < 1e-10);
        assert!((c.get(3).midpoint() - 50.0).abs() < 1e-10);
    }
}
