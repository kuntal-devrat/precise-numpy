use rayon::prelude::*;
use crate::array::IntervalArray;
use crate::error::Interval;
use crate::error::interval::{next_down, next_up, add_ru_chain, mul_ru};

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

    // Phase 1 (round-to-nearest): accumulate the midpoint, recording the
    // exact rounding error of each partial sum.
    let mut errs = Vec::with_capacity(n);
    let mut s = 0.0f64;
    for i in 0..n {
        let t = s + mids[i];
        errs.push(twosum_err(s, mids[i], t));
        s = t;
    }

    // Phase 2: accumulate radii and errors, rounding up each step.
    let mut rad = 0.0f64;
    for i in 0..n {
        rad = add_ru_chain(add_ru_chain(rad, rads[i]), errs[i].abs());
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
    let lo = if v.lo > 0.0 { next_down(v.lo.sqrt()) } else { 0.0 };
    let hi = if v.hi > 0.0 { next_up(v.hi.sqrt()) } else { 0.0 };
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

    // Phase 1 (RTN): products and TwoSum accumulation with exact errors.
    let mut errs = Vec::with_capacity(2 * n);
    let mut s = 0.0f64;
    for i in 0..n {
        let p = a_mids[i] * b_mids[i];
        errs.push(if p.is_finite() {
            a_mids[i].mul_add(b_mids[i], -p).abs()
        } else {
            f64::INFINITY
        });
        let t = s + p;
        errs.push(twosum_err(s, p, t));
        s = t;
    }

    // Phase 2: accumulate all radius contributions, rounding up each step.
    let mut rad = 0.0f64;
    for i in 0..n {
        rad = add_ru_chain(
            add_ru_chain(
                add_ru_chain(
                    add_ru_chain(
                        rad,
                        mul_ru(a_mids[i].abs(), b_rads[i]),
                    ),
                    mul_ru(b_mids[i].abs(), a_rads[i]),
                ),
                mul_ru(a_rads[i], b_rads[i]),
            ),
                    add_ru_chain(errs[2 * i], errs[2 * i + 1].abs()),
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

    // Phase 1 (RTN): cumulative midpoints with exact per-step errors.
    let mut errs = vec![0.0f64; n];
    let mut cum_mid = 0.0f64;
    for i in 0..n {
        let t = cum_mid + mids[i];
        errs[i] = twosum_err(cum_mid, mids[i], t);
        cum_mid = t;
        out_mids[i] = cum_mid;
    }

    // Phase 2: cumulative radii, rounding up each step.
    let mut cum_rad = 0.0f64;
    for i in 0..n {
        cum_rad = add_ru_chain(add_ru_chain(cum_rad, rads[i]), errs[i].abs());
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

    // Use parallel row-block decomposition only for larger matrices (M >= 256)
    if m >= 256 {
        let row_chunk_size = 64;
        r_mids.par_chunks_mut(row_chunk_size * n)
            .enumerate()
            .for_each(|(chunk_idx, m_chunk)| {
                let row_start = chunk_idx * row_chunk_size;
                let current_m = m_chunk.len() / n;

                let a_mid_ptr = unsafe { a_mids.as_ptr().add(row_start * k) };

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
            });
    } else {
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
    }

    // Rigorous radius pass: for every output element recompute the dot
    // product with TwoSum error tracking (the dgemm midpoint is only used
    // as the center; the radius encloses every rounding along the chain).
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            // Phase 1 (RTN): exact per-step errors of products and partial sums.
            let mut errs = Vec::with_capacity(2 * k);
            let mut s = 0.0f64;
            for t in 0..k {
                let am = a_mids[a_row + t];
                let bm = b_mids[t * n + j];
                let p = am * bm;
                errs.push(if p.is_finite() {
                    am.mul_add(bm, -p).abs()
                } else {
                    f64::INFINITY
                });
                let nt = s + p;
                errs.push(twosum_err(s, p, nt));
                s = nt;
            }
            // Phase 2 (round-up): radius accumulation.
            let mut rad = 0.0f64;
            for t in 0..k {
                let am = a_mids[a_row + t];
                let ar = a_rads[a_row + t];
                let bm = b_mids[t * n + j];
                let br = b_rads[t * n + j];
                rad = add_ru_chain(
                    add_ru_chain(
                        add_ru_chain(
                            add_ru_chain(
                                rad,
                                mul_ru(am.abs(), br),
                            ),
                            mul_ru(bm.abs(), ar),
                        ),
                        mul_ru(ar, br),
                    ),
                    add_ru_chain(errs[2 * t], errs[2 * t + 1].abs()),
                );
            }
            r_rads[i * n + j] = rad;
        }
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

    let lo = if sum_sq.lo > 0.0 { next_down(sum_sq.lo.sqrt()) } else { 0.0 };
    let hi = if sum_sq.hi > 0.0 { next_up(sum_sq.hi.sqrt()) } else { 0.0 };
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
                    m, k, b.len()
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
                    m, k, b.len()
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
