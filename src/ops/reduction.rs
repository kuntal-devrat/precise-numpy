use crate::array::IntervalArray;
use crate::error::interval::{add_ru_chain, mul_ru, next_down, next_up};
use crate::error::Interval;
use rayon::prelude::*;

/// Exact TwoSum rounding error of fl(a + b), computed in round-to-nearest.
/// Returns e such that a + b = fl(a + b) + e exactly.
#[inline]
fn twosum_err(a: f64, b: f64, s: f64) -> f64 {
    let bv = s - a;
    let av = s - bv;
    let br = b - bv;
    let ar = a - av;
    ar + br
}

/// Sum all elements with rigorous error propagation: the radius includes
/// the rounding error of every partial sum (TwoSum tracking).
pub fn sum(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::zero();
    }
    let mids = a.data().midpoints();
    let rads = a.data().radii();
    let n = a.len();

    // Single pass: accumulate the midpoint in round-to-nearest with the
    // exact TwoSum rounding error of each partial sum, and accumulate the
    // radii/errors rounded up inline.
    let mut s = 0.0f64;
    let mut rad = 0.0f64;
    for i in 0..n {
        let t = s + mids[i];
        let err = twosum_err(s, mids[i], t);
        s = t;
        rad = add_ru_chain(add_ru_chain(rad, rads[i]), err.abs());
    }

    Interval::from_midpoint_radius(s, rad)
}

/// Compute the mean of all elements (rigorous: interval division).
pub fn mean(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::nan();
    }
    let total = sum(a);
    total / a.len() as f64
}

/// Compute the variance of all elements (population variance), rigorously.
pub fn var(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::nan();
    }
    let n = a.len() as f64;
    let mean_iv = mean(a);

    let mut sum_iv = Interval::zero();
    for i in 0..a.len() {
        let xi = a.get(i);
        let dev = xi - mean_iv;
        let sq = dev * dev;
        sum_iv = sum_iv + sq;
    }

    let v = sum_iv / n;
    // The squared deviation set is non-negative; clamp the lower bound.
    if v.lo < 0.0 {
        Interval::new(0.0, v.hi.max(0.0))
    } else {
        v
    }
}

/// Compute the standard deviation (population), rigorously.
pub fn std_dev(a: &IntervalArray) -> Interval {
    let v = var(a);
    if v.lo.is_nan() {
        return v;
    }
    let lo = if v.lo > 0.0 {
        next_down(v.lo.sqrt())
    } else {
        0.0
    };
    let hi = if v.hi > 0.0 {
        next_up(v.hi.sqrt())
    } else {
        0.0
    };
    Interval::new(lo, hi)
}

/// Find the minimum element (rigorous hull over the element sets).
pub fn min(a: &IntervalArray) -> Interval {
    let n = a.len();
    assert!(n > 0, "min of empty array");

    let mut min_lo = f64::INFINITY;
    let mut min_hi = f64::INFINITY;
    for i in 0..n {
        let iv = a.get(i);
        if iv.lo < min_lo {
            min_lo = iv.lo;
        }
        if iv.hi < min_hi {
            min_hi = iv.hi;
        }
    }
    Interval::new(min_lo, min_hi)
}

/// Find the maximum element (rigorous hull over the element sets).
pub fn max(a: &IntervalArray) -> Interval {
    let n = a.len();
    assert!(n > 0, "max of empty array");

    let mut max_lo = f64::NEG_INFINITY;
    let mut max_hi = f64::NEG_INFINITY;
    for i in 0..n {
        let iv = a.get(i);
        if iv.lo > max_lo {
            max_lo = iv.lo;
        }
        if iv.hi > max_hi {
            max_hi = iv.hi;
        }
    }
    Interval::new(max_lo, max_hi)
}

/// Dot product of two 1D arrays with rigorous error propagation.
pub fn dot(a: &IntervalArray, b: &IntervalArray) -> Interval {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let n = a.len();

    if n == 0 {
        return Interval::zero();
    }

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());

    // Single pass: products and TwoSum accumulation with exact errors,
    // accumulating all radius contributions rounded up inline.
    let mut s = 0.0f64;
    let mut rad = 0.0f64;
    for i in 0..n {
        let am = a_mids[i];
        let ar = a_rads[i];
        let bm = b_mids[i];
        let br = b_rads[i];
        let p = am * bm;
        let e1 = if p.is_finite() {
            am.mul_add(bm, -p).abs()
        } else {
            f64::INFINITY
        };
        let t = s + p;
        let e2 = twosum_err(s, p, t);
        s = t;
        rad = add_ru_chain(
            add_ru_chain(
                add_ru_chain(
                    add_ru_chain(rad, mul_ru(am.abs(), br)),
                    mul_ru(bm.abs(), ar),
                ),
                mul_ru(ar, br),
            ),
            add_ru_chain(e1, e2.abs()),
        );
    }

    Interval::from_midpoint_radius(s, rad)
}

/// Cumulative sum along the array with rigorous error propagation.
pub fn cumsum(a: &IntervalArray) -> IntervalArray {
    let n = a.len();
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];

    // Single pass: cumulative midpoints with exact per-step TwoSum errors,
    // and cumulative radii rounded up inline.
    let mut cum_mid = 0.0f64;
    let mut cum_rad = 0.0f64;
    for i in 0..n {
        let t = cum_mid + mids[i];
        let err = twosum_err(cum_mid, mids[i], t);
        cum_mid = t;
        cum_rad = add_ru_chain(add_ru_chain(cum_rad, rads[i]), err.abs());
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

/// Run `matrixmultiply::dgemm` over a parallel row-block decomposition.
/// Same semantics as `dgemm(C = alpha*A*B)` for row-major M×K · K×N.
#[inline]
fn parallel_dgemm(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], out: &mut [f64]) {
    if m == 0 || n == 0 {
        return;
    }
    if m >= 256 {
        let row_chunk_size = 64;
        out.par_chunks_mut(row_chunk_size * n)
            .enumerate()
            .for_each(|(chunk_idx, m_chunk)| {
                let row_start = chunk_idx * row_chunk_size;
                let current_m = m_chunk.len() / n;
                let a_ptr = unsafe { a.as_ptr().add(row_start * k) };
                unsafe {
                    matrixmultiply::dgemm(
                        current_m,
                        k,
                        n,
                        1.0,
                        a_ptr,
                        k as isize,
                        1,
                        b.as_ptr(),
                        n as isize,
                        1,
                        0.0,
                        m_chunk.as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
            });
    } else {
        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
}

/// Multi-threaded Parallel GEMM matrix multiplication (`matrixmultiply`).
///
/// a: [M, K], b: [K, N] -> result: [M, N]
///
/// The midpoint is a plain dgemm. The radius is built from four rigorous
/// pieces computed with the same assembly microkernels:
///
///   rad[i, j] = Σ_t (|a|_it·rad_b_tj + rad_a_it·|b|_tj + rad_a_it·rad_b_tj
///                    + |e1_itj| + |e2_itj|)
///
/// where e1/e2 are the exact product and summation rounding errors. Since
/// every term is non-negative, any round-to-nearest summation tree (dgemm
/// reassociates freely) underestimates the exact sum by at most
/// `k·2⁻⁵³` relative. We bound the error term by
/// `Σ|e1| + Σ|e2| ≤ 2⁻⁵³·(1 + k)·Σ_t |a|_it·|b|_tj` (standard TwoSum
/// partial-sum bound), computed as a third dgemm on the absolute values.
/// A single final outward inflation `(1 + 8k·2⁻⁵²)` then covers every
/// rounding along the chain, and `next_up` closes the last one.
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

    // Degenerate output: zero rows or zero columns (numpy returns an
    // empty result). This also avoids chunk sizes of zero in the
    // parallel path below.
    if m == 0 || n == 0 {
        return IntervalArray::zeros(&[m, n]);
    }

    // Absolute values of the midpoints (needed for the radius terms).
    let a_abs: Vec<f64> = a_mids.iter().map(|&x| x.abs()).collect();
    let b_abs: Vec<f64> = b_mids.iter().map(|&x| x.abs()).collect();

    let mut r_mids = vec![0.0f64; m * n];
    let mut r_rads = vec![0.0f64; m * n];
    let mut t_ab = vec![0.0f64; m * n];
    let mut t_ba = vec![0.0f64; m * n];
    let mut t_abs = vec![0.0f64; m * n];

    parallel_dgemm(m, k, n, a_mids, b_mids, &mut r_mids);
    parallel_dgemm(m, k, n, &a_abs, b_rads, &mut t_ab);
    parallel_dgemm(m, k, n, a_rads, &b_abs, &mut t_ba);
    parallel_dgemm(m, k, n, &a_abs, &b_abs, &mut t_abs);

    // Σ_t rad_a_t·rad_b_t: constant for every output element (RTN, then
    // inflated like the dgemm results below).
    let mut s_acc = 0.0f64;
    for t in 0..k {
        s_acc = s_acc + a_rads[t] * b_rads[t];
    }
    s_acc = next_up(s_acc * (1.0 + k as f64 * 2f64.powi(-53)));

    // Error-term coefficient and final inflation factor (see doc comment).
    let err_c = 2f64.powi(-53) * (k as f64 + 1.0);
    let inflate = 1.0 + 8.0 * k as f64 * 2f64.powi(-52);

    for idx in 0..m * n {
        let mid = r_mids[idx];
        let mut rad = (t_ab[idx] + t_ba[idx]) + (s_acc + err_c * t_abs[idx]);
        rad = next_up(rad * inflate);
        // Overflow/NaN anywhere in the chain: the interval becomes the
        // entire real line rather than an unsound finite enclosure.
        if !rad.is_finite() || !mid.is_finite() {
            rad = f64::INFINITY;
        }
        r_rads[idx] = rad;
    }

    IntervalArray::from_raw_parts(&r_mids, &r_rads, &[m, n])
}

/// L2 norm: sqrt(sum(x^2)), computed rigorously in interval space.
pub fn norm_l2(a: &IntervalArray) -> Interval {
    if a.is_empty() {
        return Interval::zero();
    }

    let mut sum_sq = Interval::zero();
    for i in 0..a.len() {
        let xi = a.get(i);
        let sq = xi * xi;
        sum_sq = sum_sq + sq;
    }

    let lo = if sum_sq.lo > 0.0 {
        next_down(sum_sq.lo.sqrt())
    } else {
        0.0
    };
    let hi = if sum_sq.hi > 0.0 {
        next_up(sum_sq.hi.sqrt())
    } else {
        0.0
    };
    Interval::new(lo, hi)
}

/// Result of a generalized dot/matmul: either a scalar interval (1D·1D) or
/// an array.
pub enum MatmulResult {
    Scalar(Interval),
    Array(IntervalArray),
}

/// NumPy-style `dot`: handles 1D·1D (scalar), 2D·1D, 1D·2D, and 2D·2D.
pub fn dot_general(a: &IntervalArray, b: &IntervalArray) -> Result<MatmulResult, String> {
    let (da, db) = (a.ndim(), b.ndim());
    match (da, db) {
        (1, 1) => {
            if a.len() != b.len() {
                return Err(format!(
                    "dot: shapes ({},) and ({},) not aligned",
                    a.len(),
                    b.len()
                ));
            }
            Ok(MatmulResult::Scalar(dot(a, b)))
        }
        (2, 1) => {
            let (m, k) = (a.shape()[0], a.shape()[1]);
            if k != b.len() {
                return Err(format!(
                    "dot: shapes ({}, {}) and ({},) not aligned",
                    m,
                    k,
                    b.len()
                ));
            }
            let b2 = b.reshape(&[k, 1]);
            let out = matmul(a, &b2);
            Ok(MatmulResult::Array(out.reshape(&[m])))
        }
        (1, 2) => {
            let (k, n) = (b.shape()[0], b.shape()[1]);
            if a.len() != k {
                return Err(format!(
                    "dot: shapes ({},) and ({}, {}) not aligned",
                    a.len(),
                    k,
                    n
                ));
            }
            let a2 = a.reshape(&[1, k]);
            Ok(MatmulResult::Array(matmul(&a2, b)))
        }
        (2, 2) => {
            let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
            if k != b.shape()[0] {
                return Err(format!(
                    "dot: shapes ({}, {}) and ({}, {}) not aligned",
                    m,
                    k,
                    b.shape()[0],
                    n
                ));
            }
            Ok(MatmulResult::Array(matmul(a, b)))
        }
        _ => Err(format!(
            "dot: unsupported dimensionalities {}D and {}D",
            da, db
        )),
    }
}

/// NumPy-style `matmul`: 1D vectors are promoted per numpy rules
/// (1D·1D -> scalar, 1D·2D -> 1D, 2D·1D -> 1D, 2D·2D -> 2D).
pub fn matmul_general(a: &IntervalArray, b: &IntervalArray) -> Result<MatmulResult, String> {
    let (da, db) = (a.ndim(), b.ndim());
    match (da, db) {
        (1, 1) => {
            if a.len() != b.len() {
                return Err(format!(
                    "matmul: shapes ({},) and ({},) not aligned",
                    a.len(),
                    b.len()
                ));
            }
            Ok(MatmulResult::Scalar(dot(a, b)))
        }
        (2, 1) => {
            let (m, k) = (a.shape()[0], a.shape()[1]);
            if k != b.len() {
                return Err(format!(
                    "matmul: shapes ({}, {}) and ({},) not aligned",
                    m,
                    k,
                    b.len()
                ));
            }
            let b2 = b.reshape(&[k, 1]);
            let out = matmul(a, &b2);
            Ok(MatmulResult::Array(out.reshape(&[m])))
        }
        (1, 2) => {
            let (k, n) = (b.shape()[0], b.shape()[1]);
            if a.len() != k {
                return Err(format!(
                    "matmul: shapes ({},) and ({}, {}) not aligned",
                    a.len(),
                    k,
                    n
                ));
            }
            let a2 = a.reshape(&[1, k]);
            Ok(MatmulResult::Array(matmul(&a2, b).reshape(&[n])))
        }
        (2, 2) => {
            let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
            if k != b.shape()[0] {
                return Err(format!(
                    "matmul: shapes ({}, {}) and ({}, {}) not aligned",
                    m,
                    k,
                    b.shape()[0],
                    n
                ));
            }
            Ok(MatmulResult::Array(matmul(a, b)))
        }
        _ => Err(format!(
            "matmul: unsupported dimensionalities {}D and {}D",
            da, db
        )),
    }
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

    #[test]
    fn test_matmul_general_1d() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0]);
        let b = IntervalArray::from_f64_slice(&[4.0, 5.0, 6.0]);
        match matmul_general(&a, &b).unwrap() {
            MatmulResult::Scalar(iv) => assert!((iv.midpoint() - 32.0).abs() < 1e-10),
            MatmulResult::Array(_) => panic!("expected scalar"),
        }
    }

    #[test]
    fn test_matmul_zero_dimensions() {
        // Regression: zero columns/rows previously panicked (chunk size 0).
        let a = IntervalArray::from_f64_vec(&vec![1.0; 300 * 2], &[300, 2]);
        let b = IntervalArray::from_f64_vec(&Vec::new(), &[2, 0]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[300, 0]);
        let d = IntervalArray::from_f64_vec(&Vec::new(), &[0, 2]);
        let e = IntervalArray::from_f64_vec(&vec![1.0; 2 * 3], &[2, 3]);
        assert_eq!(matmul(&d, &e).shape(), &[0, 3]);
    }

    #[test]
    fn test_dot_general_2d_1d() {
        let a = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = IntervalArray::from_f64_slice(&[5.0, 6.0]);
        match dot_general(&a, &b).unwrap() {
            MatmulResult::Scalar(_) => panic!("expected array"),
            MatmulResult::Array(out) => {
                assert_eq!(out.shape(), &[2]);
                assert!((out.get(0).midpoint() - 17.0).abs() < 1e-10);
                assert!((out.get(1).midpoint() - 39.0).abs() < 1e-10);
            }
        }
    }
}
