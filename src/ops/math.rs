use crate::error::Interval;
use crate::array::IntervalArray;
use crate::error::interval::{
    next_down_n, next_up_n, sub_ru, LIBSM_ULP_ALLOWANCE,
};
use crate::simd::vec_ops;
use std::f64::consts::{PI, FRAC_PI_2};

/// Evaluate f(v) with round-to-nearest then expand downward by the libm
/// ulp allowance, so the result rigorously encloses the true value of
/// f(v) for a libm function accurate to within that many ulps.
#[inline]
fn eval_lo(f: fn(f64) -> f64, v: f64) -> f64 {
    next_down_n(f(v), LIBSM_ULP_ALLOWANCE)
}

#[inline]
fn eval_hi(f: fn(f64) -> f64, v: f64) -> f64 {
    next_up_n(f(v), LIBSM_ULP_ALLOWANCE)
}

/// Build the stored (mid, rad) pair for a libm function: the stored
/// center is the round-to-nearest eval of the input midpoint (matching
/// numpy), and the radius is the outward-rounded distance to the
/// enclosure endpoints. The eval is also added (with the ulp allowance)
/// as an enclosure candidate so the stored center is always rigorously
/// inside the enclosure.
#[inline]
fn centered(f: fn(f64) -> f64, m: f64, mut lo_e: f64, mut hi_e: f64) -> (f64, f64) {
    let fe = f(m);
    if fe.is_finite() {
        lo_e = lo_e.min(next_down_n(fe, LIBSM_ULP_ALLOWANCE));
        hi_e = hi_e.max(next_up_n(fe, LIBSM_ULP_ALLOWANCE));
    }
    centered_raw(fe, lo_e, hi_e)
}

/// Build the stored (mid, rad) pair for an exactly-representable function:
/// the center is the eval of the input midpoint; the radius is the
/// outward-rounded distance to the enclosure endpoints.
#[inline]
fn centered_raw(mut mid: f64, lo_e: f64, hi_e: f64) -> (f64, f64) {
    if lo_e.is_nan() || hi_e.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if !mid.is_finite() {
        mid = (lo_e + hi_e) * 0.5;
    }
    if mid < lo_e {
        mid = lo_e;
    }
    if mid > hi_e {
        mid = hi_e;
    }
    let rad = sub_ru(mid, lo_e).max(sub_ru(hi_e, mid));
    (mid, rad)
}

/// Compute the interval enclosure of sin(x) for an interval x.
///
/// Uses the property that sin is monotonic on [-pi/2, pi/2] and
/// uses range reduction for general intervals.
pub fn sin_interval(iv: Interval, m: f64) -> (f64, f64) {
    // If the interval is wider than 2*pi, result is [-1, 1]
    if iv.width() >= 2.0 * PI {
        return (0.0, 1.0);
    }

    // Evaluate sin at both endpoints and all critical points in between,
    // expanding each evaluation by the ulp allowance to cover libm error.
    let lo = iv.lo;
    let hi = iv.hi;

    let mut min_val = eval_lo(f64::sin, lo).min(eval_lo(f64::sin, hi));
    let mut max_val = eval_hi(f64::sin, lo).max(eval_hi(f64::sin, hi));

    // Critical points of sin are at pi/2 + k*pi
    let k_start = ((lo - FRAC_PI_2) / PI).floor() as i64;
    let k_end = ((hi - FRAC_PI_2) / PI).ceil() as i64;

    for k in k_start..=k_end {
        let cp = FRAC_PI_2 + k as f64 * PI;
        if cp >= lo && cp <= hi {
            let sv = cp.sin();
            min_val = min_val.min(next_down_n(sv, LIBSM_ULP_ALLOWANCE));
            max_val = max_val.max(next_up_n(sv, LIBSM_ULP_ALLOWANCE));
        }
    }

    if min_val.is_nan() || max_val.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    centered(f64::sin, m, min_val, max_val)
}

/// Compute the interval enclosure of cos(x) for an interval x.
pub fn cos_interval(iv: Interval, m: f64) -> (f64, f64) {
    if iv.width() >= 2.0 * PI {
        return (0.0, 1.0);
    }

    let lo = iv.lo;
    let hi = iv.hi;

    let mut min_val = eval_lo(f64::cos, lo).min(eval_lo(f64::cos, hi));
    let mut max_val = eval_hi(f64::cos, lo).max(eval_hi(f64::cos, hi));

    // Critical points of cos are at k*pi
    let k_start = (lo / PI).floor() as i64;
    let k_end = (hi / PI).ceil() as i64;

    for k in k_start..=k_end {
        let cp = k as f64 * PI;
        if cp >= lo && cp <= hi {
            let cv = cp.cos();
            min_val = min_val.min(next_down_n(cv, LIBSM_ULP_ALLOWANCE));
            max_val = max_val.max(next_up_n(cv, LIBSM_ULP_ALLOWANCE));
        }
    }

    if min_val.is_nan() || max_val.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    centered(f64::cos, m, min_val, max_val)
}

/// Compute the interval enclosure of tan(x) for an interval x.
///
/// Returns entire if the interval contains a singularity.
pub fn tan_interval(iv: Interval, m: f64) -> (f64, f64) {
    let lo = iv.lo;
    let hi = iv.hi;

    // Check if interval crosses any singularity (pi/2 + k*pi)
    let k_start = ((lo - FRAC_PI_2) / PI).floor() as i64;
    let k_end = ((hi - FRAC_PI_2) / PI).ceil() as i64;

    for k in k_start..=k_end {
        let sing = FRAC_PI_2 + k as f64 * PI;
        if sing > lo && sing < hi {
            return (0.0, f64::INFINITY);
        }
    }

    let min = eval_lo(f64::tan, lo).min(eval_lo(f64::tan, hi));
    let max = eval_hi(f64::tan, lo).max(eval_hi(f64::tan, hi));
    if min.is_nan() || max.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    centered(f64::tan, m, min, max)
}

/// Compute the interval enclosure of exp(x) for an interval x.
///
/// exp is monotonically increasing, so just evaluate at endpoints.
pub fn exp_interval(iv: Interval, m: f64) -> (f64, f64) {
    // Guard against overflow
    let hi = if iv.hi > 709.0 {
        f64::INFINITY
    } else {
        eval_hi(f64::exp, iv.hi)
    };
    let lo = if iv.lo < -745.0 {
        0.0
    } else {
        eval_lo(f64::exp, iv.lo).max(0.0)
    };
    centered(f64::exp, m, lo, hi)
}

/// Compute the interval enclosure of ln(x) for an interval x.
///
/// ln is monotonically increasing. Returns NaN interval for non-positive input.
pub fn ln_interval(iv: Interval, m: f64) -> (f64, f64) {
    if iv.hi <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let lo = if iv.lo > 0.0 { iv.lo } else { f64::MIN_POSITIVE };
    centered(f64::ln, m, eval_lo(f64::ln, lo), eval_hi(f64::ln, iv.hi))
}

/// Compute the interval enclosure of log2(x) for an interval x.
pub fn log2_interval(iv: Interval, m: f64) -> (f64, f64) {
    if iv.hi <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let lo = if iv.lo > 0.0 { iv.lo } else { f64::MIN_POSITIVE };
    centered(f64::log2, m, eval_lo(f64::log2, lo), eval_hi(f64::log2, iv.hi))
}

/// Compute the interval enclosure of log10(x) for an interval x.
pub fn log10_interval(iv: Interval, m: f64) -> (f64, f64) {
    if iv.hi <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let lo = if iv.lo > 0.0 { iv.lo } else { f64::MIN_POSITIVE };
    centered(f64::log10, m, eval_lo(f64::log10, lo), eval_hi(f64::log10, iv.hi))
}

/// Compute the interval enclosure of sqrt(x) for an interval x.
///
/// sqrt is monotonically increasing for x >= 0.
pub fn sqrt_interval(iv: Interval, m: f64) -> (f64, f64) {
    if iv.hi < 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let lo = if iv.lo > 0.0 { iv.lo } else { 0.0 };
    centered(f64::sqrt, m, eval_lo(f64::sqrt, lo), eval_hi(f64::sqrt, iv.hi))
}

/// Compute the interval enclosure of abs(x) for an interval x.
pub fn abs_interval(iv: Interval, m: f64) -> (f64, f64) {
    let (lo_e, hi_e) = if iv.lo >= 0.0 {
        (iv.lo, iv.hi)
    } else if iv.hi <= 0.0 {
        (-iv.hi, -iv.lo)
    } else {
        // Interval spans zero: result is [0, max(|lo|, |hi|)]
        (0.0, iv.lo.abs().max(iv.hi.abs()))
    };
    centered_raw(m.abs(), lo_e, hi_e)
}

/// Apply a math function to each element of an array.
///
/// Each element is stored as (eval of input midpoint, outward-rounded
/// distance to the enclosure endpoints), so the stored center matches
/// the round-to-nearest numpy evaluation.
pub fn apply_unary<F>(a: &IntervalArray, f: F) -> IntervalArray
where
    F: Fn(Interval, f64) -> (f64, f64) + Sync,
{
    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    // Parallelize large arrays
    if n >= vec_ops::PAR_THRESHOLD {
        return apply_unary_parallel(a, &f);
    }

    let mids = a.data().midpoints();
    let rads = a.data().radii();
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    for i in 0..n {
        let iv = Interval::from_midpoint_radius(mids[i], rads[i]);
        let (mid, rad) = f(iv, mids[i]);
        r_mids[i] = mid;
        r_rads[i] = rad;
    }

    result
}

/// Parallel apply_unary using Rayon for large arrays.
fn apply_unary_parallel<F>(a: &IntervalArray, f: &F) -> IntervalArray
where
    F: Fn(Interval, f64) -> (f64, f64) + Sync,
{
    use rayon::prelude::*;

    let n = a.len();
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mut result = IntervalArray::zeros(a.shape());

    // Process in parallel chunks
    const CHUNK: usize = 4096;
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();
    r_mids
        .par_chunks_mut(CHUNK)
        .zip(r_rads.par_chunks_mut(CHUNK))
        .enumerate()
        .for_each(|(chunk_idx, (om, or))| {
            let start = chunk_idx * CHUNK;
            for i in 0..om.len() {
                let iv = Interval::from_midpoint_radius(mids[start + i], rads[start + i]);
                let (mid, rad) = f(iv, mids[start + i]);
                om[i] = mid;
                or[i] = rad;
            }
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_exact() {
        let iv = Interval::exact(0.0);
        let (mid, rad) = sin_interval(iv, 0.0);
        assert!((mid - 0.0).abs() < 1e-10);
        assert!(rad < 1e-10);
    }

    #[test]
    fn test_sin_wide() {
        let iv = Interval::new(0.0, 10.0); // wider than 2*pi
        let (mid, rad) = sin_interval(iv, 5.0);
        assert_eq!(mid, 0.0);
        assert_eq!(rad, 1.0);
    }

    #[test]
    fn test_cos_zero() {
        let iv = Interval::exact(0.0);
        let (mid, _) = cos_interval(iv, 0.0);
        assert!((mid - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_zero() {
        let iv = Interval::exact(0.0);
        let (mid, _) = exp_interval(iv, 0.0);
        assert!((mid - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_one() {
        let iv = Interval::exact(1.0);
        let (mid, _) = ln_interval(iv, 1.0);
        assert!((mid - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_negative_returns_nan() {
        let iv = Interval::exact(-1.0);
        let (mid, _) = ln_interval(iv, -1.0);
        assert!(mid.is_nan());
    }

    #[test]
    fn test_sqrt_four() {
        let iv = Interval::exact(4.0);
        let (mid, rad) = sqrt_interval(iv, 4.0);
        assert_eq!(mid, 2.0);
        assert!(rad < 1e-10);
    }

    #[test]
    fn test_sqrt_negative_returns_nan() {
        let iv = Interval::exact(-1.0);
        let (mid, _) = sqrt_interval(iv, -1.0);
        assert!(mid.is_nan());
    }

    #[test]
    fn test_abs_negative() {
        let iv = Interval::exact(-5.0);
        let (mid, _) = abs_interval(iv, -5.0);
        assert!((mid - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_abs_spanning_zero() {
        let iv = Interval::new(-3.0, 5.0);
        let (mid, rad) = abs_interval(iv, 1.0);
        assert!((mid - 1.0).abs() < 1e-10);
        assert!(mid - rad <= 0.0 + 1e-12);
        assert!(mid + rad >= 5.0 - 1e-12);
    }

    #[test]
    fn test_apply_unary_sin() {
        let arr = IntervalArray::from_f64_slice(&[0.0, std::f64::consts::FRAC_PI_2]);
        let result = apply_unary(&arr, sin_interval);
        assert!((result.get(0).midpoint() - 0.0).abs() < 1e-10);
        assert!((result.get(1).midpoint() - 1.0).abs() < 1e-10);
    }
}
