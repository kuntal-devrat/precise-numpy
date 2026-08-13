use rayon::prelude::*;
use crate::array::IntervalArray;
use crate::error::Interval;
use crate::simd::vec_ops;

/// SIMD & parallel-accelerated single-pass interval addition.
pub fn add_arrays(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.len(), b.len(), "array lengths must match for add");
    assert_eq!(a.shape(), b.shape(), "shapes must match for add");

    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    // Exact array fast path: if both arrays have zero error, radii remain zero!
    if a.is_exact() && b.is_exact() {
        vec_ops::add_f64(a_mids, b_mids, r_mids);
        return result;
    }

    // Exact b fast path: radius is unchanged!
    if b.is_exact() {
        vec_ops::add_f64(a_mids, b_mids, r_mids);
        r_rads.copy_from_slice(a_rads);
        return result;
    }

    // Exact a fast path: radius is unchanged!
    if a.is_exact() {
        vec_ops::add_f64(a_mids, b_mids, r_mids);
        r_rads.copy_from_slice(b_rads);
        return result;
    }

    if n >= vec_ops::PAR_THRESHOLD {
        const CHUNK: usize = 8192;
        r_mids.par_chunks_mut(CHUNK)
            .zip(r_rads.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (rm, rr))| {
                let start = chunk_idx * CHUNK;
                let end = start + rm.len();
                vec_ops::add_intervals_stream(
                    &a_mids[start..end],
                    &a_rads[start..end],
                    &b_mids[start..end],
                    &b_rads[start..end],
                    rm,
                    rr,
                );
            });
    } else {
        vec_ops::add_intervals_stream(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
    }

    result
}

/// SIMD & parallel-accelerated single-pass interval subtraction.
pub fn sub_arrays(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.len(), b.len(), "array lengths must match for sub");
    assert_eq!(a.shape(), b.shape(), "shapes must match for sub");

    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    // Exact array fast path
    if a.is_exact() && b.is_exact() {
        vec_ops::sub_f64(a_mids, b_mids, r_mids);
        return result;
    }

    // Exact b fast path: radius is unchanged!
    if b.is_exact() {
        vec_ops::sub_f64(a_mids, b_mids, r_mids);
        r_rads.copy_from_slice(a_rads);
        return result;
    }

    // Exact a fast path: radius is unchanged!
    if a.is_exact() {
        vec_ops::sub_f64(a_mids, b_mids, r_mids);
        r_rads.copy_from_slice(b_rads);
        return result;
    }

    if n >= vec_ops::PAR_THRESHOLD {
        const CHUNK: usize = 8192;
        r_mids.par_chunks_mut(CHUNK)
            .zip(r_rads.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (rm, rr))| {
                let start = chunk_idx * CHUNK;
                let end = start + rm.len();
                vec_ops::sub_intervals_stream(
                    &a_mids[start..end],
                    &a_rads[start..end],
                    &b_mids[start..end],
                    &b_rads[start..end],
                    rm,
                    rr,
                );
            });
    } else {
        vec_ops::sub_intervals_stream(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
    }

    result
}

/// SIMD & parallel-accelerated single-pass streaming interval multiplication.
///
/// Uses `mul_intervals_stream` super-instruction:
/// Computes both midpoints and radii in SIMD vector registers in a SINGLE pass.
/// ZERO auxiliary memory allocations!
pub fn mul_arrays(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.len(), b.len(), "array lengths must match for mul");
    assert_eq!(a.shape(), b.shape(), "shapes must match for mul");

    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    // Exact array fast path: if both arrays are exact, radii remain zero!
    if a.is_exact() && b.is_exact() {
        vec_ops::mul_f64(a_mids, b_mids, r_mids);
        return result;
    }

    // Exact b fast path: radius = |b_mid| * a_rad
    if b.is_exact() {
        vec_ops::mul_f64(a_mids, b_mids, r_mids);
        vec_ops::abs_mul_f64(b_mids, a_rads, r_rads);
        return result;
    }

    // Exact a fast path: radius = |a_mid| * b_rad
    if a.is_exact() {
        vec_ops::mul_f64(a_mids, b_mids, r_mids);
        vec_ops::abs_mul_f64(a_mids, b_rads, r_rads);
        return result;
    }

    if n >= vec_ops::PAR_THRESHOLD {
        const CHUNK: usize = 8192;
        r_mids.par_chunks_mut(CHUNK)
            .zip(r_rads.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (rm, rr))| {
                let start = chunk_idx * CHUNK;
                let end = start + rm.len();
                vec_ops::mul_intervals_stream(
                    &a_mids[start..end],
                    &a_rads[start..end],
                    &b_mids[start..end],
                    &b_rads[start..end],
                    rm,
                    rr,
                );
            });
    } else {
        vec_ops::mul_intervals_stream(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
    }

    result
}

/// SIMD & parallel-accelerated interval division.
pub fn div_arrays(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.len(), b.len(), "array lengths must match for div");
    assert_eq!(a.shape(), b.shape(), "shapes must match for div");

    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    // Fast check for zero crossing
    let mut has_zero_crossing = false;
    for i in 0..n {
        if b_mids[i] - b_rads[i] <= 0.0 && b_mids[i] + b_rads[i] >= 0.0 {
            has_zero_crossing = true;
            break;
        }
    }

    // Exact array fast path: if both arrays have zero error, radii remain zero!
    if a.is_exact() && b.is_exact() {
        vec_ops::div_f64(a_mids, b_mids, r_mids);
        return result;
    }

    // Exact b fast path: c_rad = a_rad / |b_mid|
    if b.is_exact() {
        let mut has_zero = false;
        for i in 0..n {
            if b_mids[i] == 0.0 {
                has_zero = true;
                break;
            }
        }
        if !has_zero {
            vec_ops::div_f64(a_mids, b_mids, r_mids);
            let abs_b: Vec<f64> = b_mids.iter().map(|&x| x.abs()).collect();
            vec_ops::div_f64(a_rads, &abs_b, r_rads);
            return result;
        }
    }

    // Exact a fast path: c_rad = |a_mid| * b_rad / |b_mid^2 - b_rad^2|
    if a.is_exact() && !has_zero_crossing {
        vec_ops::div_f64(a_mids, b_mids, r_mids);
        for i in 0..n {
            let am = a_mids[i];
            let bm = b_mids[i];
            let br = b_rads[i];
            let denom = (bm * bm - br * br).abs();
            r_rads[i] = (am.abs() * br) / denom;
        }
        return result;
    }

    if has_zero_crossing {
        for i in 0..n {
            let a_iv = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
            let b_iv = Interval::from_midpoint_radius(b_mids[i], b_rads[i]);
            let r = a_iv / b_iv;
            r_mids[i] = r.midpoint();
            r_rads[i] = r.radius();
        }
    } else {
        // Fast SIMD path: mid = a.mid / b.mid
        vec_ops::div_f64(a_mids, b_mids, r_mids);

        // Numerator & denominator via streaming ops
        for i in 0..n {
            let am = a_mids[i];
            let ar = a_rads[i];
            let bm = b_mids[i];
            let br = b_rads[i];
            let denom = (bm * bm - br * br).abs();
            let numer = am.abs() * br + bm.abs() * ar + ar * br;
            r_rads[i] = numer / denom;
        }
    }

    result
}

/// Bulk negation using SIMD.
pub fn neg_array(a: &IntervalArray) -> IntervalArray {
    let mut result = IntervalArray::zeros(a.shape());
    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    vec_ops::neg_f64(a_mids, r_mids);
    r_rads.copy_from_slice(a_rads);

    result
}

/// Add a scalar interval to every element using scalar-broadcast SIMD.
pub fn add_scalar(a: &IntervalArray, s: Interval) -> IntervalArray {
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    let s_mid = s.midpoint();
    let s_rad = s.radius();

    vec_ops::add_scalar_f64(a_mids, s_mid, r_mids);
    vec_ops::add_scalar_f64(a_rads, s_rad, r_rads);

    result
}

/// Multiply every element by a scalar interval using SIMD streaming super-instruction.
pub fn mul_scalar(a: &IntervalArray, s: Interval) -> IntervalArray {
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    let s_mid = s.midpoint();
    let s_rad = s.radius();

    vec_ops::mul_scalar_stream(a_mids, a_rads, s_mid, s_rad, r_mids, r_rads);

    result
}

/// Scale all elements by an exact scalar.
pub fn scale_array(a: &IntervalArray, scalar: f64) -> IntervalArray {
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    vec_ops::scale_f64(a_mids, scalar, r_mids);
    vec_ops::scale_f64(a_rads, scalar.abs(), r_rads);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_arrays() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0]);
        let b = IntervalArray::from_f64_slice(&[4.0, 5.0, 6.0]);
        let c = add_arrays(&a, &b);
        assert!((c.get(0).midpoint() - 5.0).abs() < 1e-10);
        assert!((c.get(2).midpoint() - 9.0).abs() < 1e-10);
        assert!(c.is_exact());
    }

    #[test]
    fn test_mul_arrays() {
        let a = IntervalArray::from_f64_slice(&[2.0, 3.0]);
        let b = IntervalArray::from_f64_slice(&[4.0, 5.0]);
        let c = mul_arrays(&a, &b);
        assert!((c.get(0).midpoint() - 8.0).abs() < 1e-10);
        assert!((c.get(1).midpoint() - 15.0).abs() < 1e-10);
    }
}
