use crate::array::IntervalArray;
use crate::error::Interval;
use crate::simd::vec_ops;
use rayon::prelude::*;

/// SIMD & parallel-accelerated interval addition with rigorous error
/// propagation: the radius includes the rounding error of the midpoint sum.
pub fn add_arrays(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.len(), b.len(), "array lengths must match for add");
    assert_eq!(a.shape(), b.shape(), "shapes must match for add");

    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    if n >= vec_ops::PAR_THRESHOLD {
        const CHUNK: usize = 8192;
        r_mids
            .par_chunks_mut(CHUNK)
            .zip(r_rads.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (rm, rr))| {
                let start = chunk_idx * CHUNK;
                let end = start + rm.len();
                vec_ops::add_intervals_rigorous(
                    &a_mids[start..end],
                    &a_rads[start..end],
                    &b_mids[start..end],
                    &b_rads[start..end],
                    rm,
                    rr,
                );
            });
    } else {
        vec_ops::add_intervals_rigorous(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
    }

    result
}

/// SIMD & parallel-accelerated interval subtraction with rigorous error
/// propagation: the radius includes the rounding error of the midpoint diff.
pub fn sub_arrays(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.len(), b.len(), "array lengths must match for sub");
    assert_eq!(a.shape(), b.shape(), "shapes must match for sub");

    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    if n >= vec_ops::PAR_THRESHOLD {
        const CHUNK: usize = 8192;
        r_mids
            .par_chunks_mut(CHUNK)
            .zip(r_rads.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (rm, rr))| {
                let start = chunk_idx * CHUNK;
                let end = start + rm.len();
                vec_ops::sub_intervals_rigorous(
                    &a_mids[start..end],
                    &a_rads[start..end],
                    &b_mids[start..end],
                    &b_rads[start..end],
                    rm,
                    rr,
                );
            });
    } else {
        vec_ops::sub_intervals_rigorous(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
    }

    result
}

/// SIMD & parallel-accelerated interval multiplication with rigorous error
/// propagation: the radius includes the rounding error of the midpoint
/// product.
pub fn mul_arrays(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.len(), b.len(), "array lengths must match for mul");
    assert_eq!(a.shape(), b.shape(), "shapes must match for mul");

    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    if n >= vec_ops::PAR_THRESHOLD {
        const CHUNK: usize = 8192;
        r_mids
            .par_chunks_mut(CHUNK)
            .zip(r_rads.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (rm, rr))| {
                let start = chunk_idx * CHUNK;
                let end = start + rm.len();
                vec_ops::mul_intervals_rigorous(
                    &a_mids[start..end],
                    &a_rads[start..end],
                    &b_mids[start..end],
                    &b_rads[start..end],
                    rm,
                    rr,
                );
            });
    } else {
        vec_ops::mul_intervals_rigorous(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
    }

    result
}

/// SIMD & parallel-accelerated interval division with rigorous error
/// propagation. Divisors whose interval contains zero fall back to the
/// full interval division (result becomes the entire real line). Returns
/// whether any divisor interval contained zero.
pub fn div_arrays(a: &IntervalArray, b: &IntervalArray) -> (IntervalArray, bool) {
    assert_eq!(a.len(), b.len(), "array lengths must match for div");
    assert_eq!(a.shape(), b.shape(), "shapes must match for div");

    let n = a.len();
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (b_mids, b_rads) = (b.data().midpoints(), b.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    // Check for zero crossing (single scan; also decides the fallback path)
    let mut has_zero_crossing = false;
    for i in 0..n {
        if b_mids[i] - b_rads[i] <= 0.0 && b_mids[i] + b_rads[i] >= 0.0 {
            has_zero_crossing = true;
            break;
        }
    }

    if has_zero_crossing {
        for i in 0..n {
            let a_iv = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
            let b_iv = Interval::from_midpoint_radius(b_mids[i], b_rads[i]);
            let r = a_iv / b_iv;
            r_mids[i] = r.midpoint();
            r_rads[i] = r.radius();
        }
    } else if n >= vec_ops::PAR_THRESHOLD {
        const CHUNK: usize = 8192;
        r_mids
            .par_chunks_mut(CHUNK)
            .zip(r_rads.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (rm, rr))| {
                let start = chunk_idx * CHUNK;
                let end = start + rm.len();
                vec_ops::div_intervals_rigorous(
                    &a_mids[start..end],
                    &a_rads[start..end],
                    &b_mids[start..end],
                    &b_rads[start..end],
                    rm,
                    rr,
                );
            });
    } else {
        vec_ops::div_intervals_rigorous(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
    }

    (result, has_zero_crossing)
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

/// Add a scalar interval to every element with rigorous error propagation.
pub fn add_scalar(a: &IntervalArray, s: Interval) -> IntervalArray {
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    let s_mid = s.midpoint();
    let s_rad = s.radius();
    let n = a.len();

    for i in 0..n {
        let am = a_mids[i];
        let t = am + s_mid;
        let bv = t - am;
        let av = t - bv;
        let br = s_mid - bv;
        let ar = am - av;
        let err = ar + br;
        r_mids[i] = t;
        r_rads[i] = crate::error::interval::add_ru_chain(
            crate::error::interval::add_ru_chain(a_rads[i], s_rad),
            err.abs(),
        );
    }

    result
}

/// Multiply every element by a scalar interval with rigorous error
/// propagation.
pub fn mul_scalar(a: &IntervalArray, s: Interval) -> IntervalArray {
    let mut result = IntervalArray::zeros(a.shape());

    let (a_mids, a_rads) = (a.data().midpoints(), a.data().radii());
    let (r_mids, r_rads) = result.data_mut().as_mut_slices();

    let s_mid = s.midpoint();
    let s_rad = s.radius();
    let n = a.len();

    for i in 0..n {
        let t = a_mids[i] * s_mid;
        let err = if t.is_finite() {
            a_mids[i].mul_add(s_mid, -t).abs()
        } else {
            f64::INFINITY
        };
        r_mids[i] = t;
        r_rads[i] = crate::error::interval::add_ru_chain(
            crate::error::interval::add_ru_chain(
                crate::error::interval::mul_ru(a_mids[i].abs(), s_rad),
                crate::error::interval::mul_ru(a_rads[i], s_mid.abs()),
            ),
            crate::error::interval::add_ru_chain(
                crate::error::interval::mul_ru(a_rads[i], s_rad),
                err,
            ),
        );
    }

    result
}

/// Scale all elements by an exact scalar (rigorous).
pub fn scale_array(a: &IntervalArray, scalar: f64) -> IntervalArray {
    mul_scalar(a, Interval::exact(scalar))
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
