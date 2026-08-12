use crate::error::Interval;
use crate::array::IntervalArray;
use crate::simd::vec_ops;
use std::f64::consts::{PI, FRAC_PI_2};

/// Compute the interval enclosure of sin(x) for an interval x.
///
/// Uses the property that sin is monotonic on [-pi/2, pi/2] and
/// uses range reduction for general intervals.
pub fn sin_interval(iv: Interval) -> Interval {
    // If the interval is wider than 2*pi, result is [-1, 1]
    if iv.width() >= 2.0 * PI {
        return Interval::new(-1.0, 1.0);
    }

    // Evaluate sin at both endpoints and all critical points in between
    let lo = iv.lo;
    let hi = iv.hi;

    // Find all critical points (where sin' = cos = 0) in [lo, hi]
    let mut min_val = lo.sin().min(hi.sin());
    let mut max_val = lo.sin().max(hi.sin());

    // Critical points of sin are at pi/2 + k*pi
    let k_start = ((lo - FRAC_PI_2) / PI).floor() as i64;
    let k_end = ((hi - FRAC_PI_2) / PI).ceil() as i64;

    for k in k_start..=k_end {
        let cp = FRAC_PI_2 + k as f64 * PI;
        if cp >= lo && cp <= hi {
            let sv = cp.sin();
            min_val = min_val.min(sv);
            max_val = max_val.max(sv);
        }
    }

    Interval::new(min_val, max_val)
}

/// Compute the interval enclosure of cos(x) for an interval x.
pub fn cos_interval(iv: Interval) -> Interval {
    if iv.width() >= 2.0 * PI {
        return Interval::new(-1.0, 1.0);
    }

    let lo = iv.lo;
    let hi = iv.hi;

    let mut min_val = lo.cos().min(hi.cos());
    let mut max_val = lo.cos().max(hi.cos());

    // Critical points of cos are at k*pi
    let k_start = (lo / PI).floor() as i64;
    let k_end = (hi / PI).ceil() as i64;

    for k in k_start..=k_end {
        let cp = k as f64 * PI;
        if cp >= lo && cp <= hi {
            let cv = cp.cos();
            min_val = min_val.min(cv);
            max_val = max_val.max(cv);
        }
    }

    Interval::new(min_val, max_val)
}

/// Compute the interval enclosure of tan(x) for an interval x.
///
/// Returns entire if the interval contains a singularity.
pub fn tan_interval(iv: Interval) -> Interval {
    let lo = iv.lo;
    let hi = iv.hi;

    // Check if interval crosses any singularity (pi/2 + k*pi)
    let k_start = ((lo - FRAC_PI_2) / PI).floor() as i64;
    let k_end = ((hi - FRAC_PI_2) / PI).ceil() as i64;

    for k in k_start..=k_end {
        let sing = FRAC_PI_2 + k as f64 * PI;
        if sing > lo && sing < hi {
            return Interval::entire();
        }
    }

    let min = lo.tan().min(hi.tan());
    let max = lo.tan().max(hi.tan());
    Interval::new(min, max)
}

/// Compute the interval enclosure of exp(x) for an interval x.
///
/// exp is monotonically increasing, so just evaluate at endpoints.
pub fn exp_interval(iv: Interval) -> Interval {
    // Guard against overflow
    if iv.hi > 709.0 {
        return Interval::new(iv.lo.exp(), f64::INFINITY);
    }
    if iv.lo < -745.0 {
        return Interval::new(0.0, iv.hi.exp());
    }
    Interval::new(iv.lo.exp(), iv.hi.exp())
}

/// Compute the interval enclosure of ln(x) for an interval x.
///
/// ln is monotonically increasing. Returns NaN interval for non-positive input.
pub fn ln_interval(iv: Interval) -> Interval {
    if iv.hi <= 0.0 {
        return Interval::nan();
    }
    let lo = if iv.lo > 0.0 { iv.lo } else { f64::MIN_POSITIVE };
    Interval::new(lo.ln(), iv.hi.ln())
}

/// Compute the interval enclosure of log2(x) for an interval x.
pub fn log2_interval(iv: Interval) -> Interval {
    if iv.hi <= 0.0 {
        return Interval::nan();
    }
    let lo = if iv.lo > 0.0 { iv.lo } else { f64::MIN_POSITIVE };
    Interval::new(lo.log2(), iv.hi.log2())
}

/// Compute the interval enclosure of log10(x) for an interval x.
pub fn log10_interval(iv: Interval) -> Interval {
    if iv.hi <= 0.0 {
        return Interval::nan();
    }
    let lo = if iv.lo > 0.0 { iv.lo } else { f64::MIN_POSITIVE };
    Interval::new(lo.log10(), iv.hi.log10())
}

/// Compute the interval enclosure of sqrt(x) for an interval x.
///
/// sqrt is monotonically increasing for x >= 0.
pub fn sqrt_interval(iv: Interval) -> Interval {
    if iv.hi < 0.0 {
        return Interval::nan();
    }
    let lo = if iv.lo > 0.0 { iv.lo } else { 0.0 };
    Interval::new(lo.sqrt(), iv.hi.sqrt())
}

/// Compute the interval enclosure of abs(x) for an interval x.
pub fn abs_interval(iv: Interval) -> Interval {
    if iv.lo >= 0.0 {
        iv
    } else if iv.hi <= 0.0 {
        Interval::new(-iv.hi, -iv.lo)
    } else {
        // Interval spans zero: result is [0, max(|lo|, |hi|)]
        Interval::new(0.0, iv.lo.abs().max(iv.hi.abs()))
    }
}

/// Apply a math function to each element of an array.
///
/// For monotonic functions on exact arrays, uses a fast batch path.
/// Falls back to element-wise for intervals with nonzero radius.
pub fn apply_unary(a: &IntervalArray, f: fn(Interval) -> Interval) -> IntervalArray {
    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    // Parallelize large arrays
    if n >= vec_ops::PAR_THRESHOLD {
        return apply_unary_parallel(a, f);
    }

    let mids = a.data().midpoints();
    let rads = a.data().radii();
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    for i in 0..n {
        let iv = Interval::from_midpoint_radius(mids[i], rads[i]);
        let out = f(iv);
        r_mids[i] = out.midpoint();
        r_rads[i] = out.radius();
    }

    result
}

/// Parallel apply_unary using Rayon for large arrays.
fn apply_unary_parallel(a: &IntervalArray, f: fn(Interval) -> Interval) -> IntervalArray {
    use rayon::prelude::*;

    let n = a.len();
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];

    // Process in parallel chunks
    const CHUNK: usize = 4096;
    let chunks: Vec<(usize, usize)> = (0..n).step_by(CHUNK)
        .map(|start| (start, (start + CHUNK).min(n)))
        .collect();

    let results: Vec<(Vec<f64>, Vec<f64>)> = chunks.par_iter().map(|&(start, end)| {
        let len = end - start;
        let mut chunk_mids = vec![0.0f64; len];
        let mut chunk_rads = vec![0.0f64; len];
        for i in 0..len {
            let iv = Interval::from_midpoint_radius(mids[start + i], rads[start + i]);
            let out = f(iv);
            chunk_mids[i] = out.midpoint();
            chunk_rads[i] = out.radius();
        }
        (chunk_mids, chunk_rads)
    }).collect();

    // Stitch results back
    for (idx, &(start, end)) in chunks.iter().enumerate() {
        let (ref cm, ref cr) = results[idx];
        out_mids[start..end].copy_from_slice(cm);
        out_rads[start..end].copy_from_slice(cr);
    }

    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Optimized batch sin for exact arrays (zero radius).
/// Avoids interval overhead when all elements are exact.
pub fn sin_batch_exact(a: &IntervalArray) -> IntervalArray {
    let n = a.len();
    let mids = a.data().midpoints();
    let mut out_mids = vec![0.0f64; n];
    let out_rads = vec![0.0f64; n];

    for i in 0..n {
        out_mids[i] = mids[i].sin();
    }

    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Optimized batch exp for exact arrays.
pub fn exp_batch_exact(a: &IntervalArray) -> IntervalArray {
    let n = a.len();
    let mids = a.data().midpoints();
    let mut out_mids = vec![0.0f64; n];
    let out_rads = vec![0.0f64; n];

    for i in 0..n {
        out_mids[i] = mids[i].exp();
    }

    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Optimized batch sqrt for exact arrays.
pub fn sqrt_batch_exact(a: &IntervalArray) -> IntervalArray {
    let n = a.len();
    let mids = a.data().midpoints();
    let mut out_mids = vec![0.0f64; n];
    let out_rads = vec![0.0f64; n];

    for i in 0..n {
        out_mids[i] = if mids[i] >= 0.0 { mids[i].sqrt() } else { f64::NAN };
    }

    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Optimized batch ln for exact arrays.
pub fn ln_batch_exact(a: &IntervalArray) -> IntervalArray {
    let n = a.len();
    let mids = a.data().midpoints();
    let mut out_mids = vec![0.0f64; n];
    let out_rads = vec![0.0f64; n];

    for i in 0..n {
        out_mids[i] = if mids[i] > 0.0 { mids[i].ln() } else { f64::NEG_INFINITY };
    }

    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_exact() {
        let iv = Interval::exact(0.0);
        let r = sin_interval(iv);
        assert!((r.midpoint() - 0.0).abs() < 1e-10);
        assert!(r.width() < 1e-10);
    }

    #[test]
    fn test_sin_wide() {
        let iv = Interval::new(0.0, 10.0); // wider than 2*pi
        let r = sin_interval(iv);
        assert_eq!(r.lo, -1.0);
        assert_eq!(r.hi, 1.0);
    }

    #[test]
    fn test_cos_zero() {
        let iv = Interval::exact(0.0);
        let r = cos_interval(iv);
        assert!((r.midpoint() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_zero() {
        let iv = Interval::exact(0.0);
        let r = exp_interval(iv);
        assert!((r.midpoint() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_one() {
        let iv = Interval::exact(1.0);
        let r = ln_interval(iv);
        assert!((r.midpoint() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_negative_returns_nan() {
        let iv = Interval::exact(-1.0);
        let r = ln_interval(iv);
        assert!(r.lo.is_nan());
    }

    #[test]
    fn test_sqrt_four() {
        let iv = Interval::exact(4.0);
        let r = sqrt_interval(iv);
        assert!((r.midpoint() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_negative_returns_nan() {
        let iv = Interval::exact(-1.0);
        let r = sqrt_interval(iv);
        assert!(r.lo.is_nan());
    }

    #[test]
    fn test_abs_negative() {
        let iv = Interval::exact(-5.0);
        let r = abs_interval(iv);
        assert!((r.midpoint() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_abs_spanning_zero() {
        let iv = Interval::new(-3.0, 5.0);
        let r = abs_interval(iv);
        assert!((r.lo - 0.0).abs() < 1e-10);
        assert!((r.hi - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_unary_sin() {
        let arr = IntervalArray::from_f64_slice(&[0.0, std::f64::consts::FRAC_PI_2]);
        let result = apply_unary(&arr, sin_interval);
        assert!((result.get(0).midpoint() - 0.0).abs() < 1e-10);
        assert!((result.get(1).midpoint() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sin_batch_exact() {
        let arr = IntervalArray::from_f64_slice(&[0.0, std::f64::consts::FRAC_PI_2]);
        let result = sin_batch_exact(&arr);
        assert!((result.get(0).midpoint() - 0.0).abs() < 1e-10);
        assert!((result.get(1).midpoint() - 1.0).abs() < 1e-10);
    }
}
