use crate::array::IntervalArray;
use crate::error::Interval;
use crate::simd::vec_ops;
use rayon::prelude::*;

/// Parallel element-wise operation using Rayon work-stealing.
///
/// Works directly on SoA slices instead of collecting into Vec<Interval>.
pub fn par_apply_unary(
    a: &IntervalArray,
    f: impl Fn(Interval) -> Interval + Sync,
) -> IntervalArray {
    let n = a.len();
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];

    const CHUNK: usize = 4096;

    out_mids
        .par_chunks_mut(CHUNK)
        .zip(out_rads.par_chunks_mut(CHUNK))
        .enumerate()
        .for_each(|(chunk_idx, (om, or))| {
            let start = chunk_idx * CHUNK;
            for i in 0..om.len() {
                let iv = Interval::from_midpoint_radius(mids[start + i], rads[start + i]);
                let result = f(iv);
                om[i] = result.midpoint();
                or[i] = result.radius();
            }
        });

    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Parallel element-wise binary operation.
/// Works directly on SoA slices.
pub fn par_apply_binary(
    a: &IntervalArray,
    b: &IntervalArray,
    f: impl Fn(Interval, Interval) -> Interval + Sync,
) -> IntervalArray {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let b_mids = b.data().midpoints();
    let b_rads = b.data().radii();

    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];

    const CHUNK: usize = 4096;

    out_mids
        .par_chunks_mut(CHUNK)
        .zip(out_rads.par_chunks_mut(CHUNK))
        .enumerate()
        .for_each(|(chunk_idx, (om, or))| {
            let start = chunk_idx * CHUNK;
            for i in 0..om.len() {
                let a_iv = Interval::from_midpoint_radius(a_mids[start + i], a_rads[start + i]);
                let b_iv = Interval::from_midpoint_radius(b_mids[start + i], b_rads[start + i]);
                let result = f(a_iv, b_iv);
                om[i] = result.midpoint();
                or[i] = result.radius();
            }
        });

    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Parallel sum using Rayon with cache-friendly chunking.
/// Works directly on SoA slices.
pub fn par_sum(a: &IntervalArray) -> Interval {
    let mids = a.data().midpoints();
    let rads = a.data().radii();

    let (mid_sum, rad_sum) = mids
        .par_chunks(4096)
        .zip(rads.par_chunks(4096))
        .map(|(mc, rc)| {
            let m = vec_ops::sum_f64(mc);
            let r = vec_ops::sum_f64(rc);
            (m, r)
        })
        .reduce(|| (0.0f64, 0.0f64), |(m1, r1), (m2, r2)| (m1 + m2, r1 + r2));

    Interval::from_midpoint_radius(mid_sum, rad_sum)
}

/// Get the number of Rayon worker threads.
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_sum() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let s = par_sum(&a);
        assert!((s.midpoint() - 15.0).abs() < 1e-10);
        assert!(s.radius() < 1e-15);
    }

    #[test]
    fn test_par_apply_unary() {
        let a = IntervalArray::from_f64_slice(&[1.0, 4.0, 9.0]);
        let result = par_apply_unary(&a, |iv| {
            let mid = iv.midpoint();
            Interval::exact(mid.sqrt())
        });
        assert!((result.get(0).midpoint() - 1.0).abs() < 1e-10);
        assert!((result.get(1).midpoint() - 2.0).abs() < 1e-10);
        assert!((result.get(2).midpoint() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_num_threads() {
        let n = num_threads();
        assert!(n >= 1);
    }
}
