//! Interval comparisons. Two intervals are considered "equal" when they overlap.

use crate::array::IntervalArray;
use crate::error::Interval;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[inline]
fn cmp_interval(a: &Interval, b: &Interval, op: Cmp) -> bool {
    match op {
        Cmp::Eq => a.lo <= b.hi && b.lo <= a.hi,
        Cmp::Ne => a.lo > b.hi || b.lo > a.hi,
        Cmp::Lt => a.hi < b.lo,
        Cmp::Le => a.hi <= b.lo,
        Cmp::Gt => a.lo > b.hi,
        Cmp::Ge => a.lo >= b.hi,
    }
}

/// Element-wise comparison of two arrays (broadcast-compatible lengths).
pub fn compare_arrays(a: &IntervalArray, b: &IntervalArray, op: Cmp) -> Vec<bool> {
    assert_eq!(a.len(), b.len(), "comparison: length mismatch");
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let b_mids = b.data().midpoints();
    let b_rads = b.data().radii();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
        let bi = Interval::from_midpoint_radius(b_mids[i], b_rads[i]);
        out.push(cmp_interval(&ai, &bi, op));
    }
    out
}

/// Element-wise comparison of an array against an exact scalar.
pub fn compare_scalar(a: &IntervalArray, s: f64, op: Cmp) -> Vec<bool> {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
        let bi = Interval::exact(s);
        out.push(cmp_interval(&ai, &bi, op));
    }
    out
}

/// Whether any element of `a` (as intervals) overlaps `b`.
pub fn any_overlap(a: &IntervalArray, b: &IntervalArray) -> bool {
    compare_arrays(a, b, Cmp::Eq).iter().any(|&x| x)
}

/// Whether any element of `a` overlaps the exact scalar `s`.
pub fn any_overlap_scalar(a: &IntervalArray, s: f64) -> bool {
    compare_scalar(a, s, Cmp::Eq).iter().any(|&x| x)
}

/// Whether the scalar interval [s, s] overlaps any element of `a`.
pub fn any_overlap_inverse(a: &IntervalArray, s: f64) -> bool {
    any_overlap_scalar(a, s)
}

/// Element-wise NaN test (either endpoint is NaN).
pub fn is_nan_array(a: &IntervalArray) -> Vec<bool> {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(a_mids[i].is_nan() || a_rads[i].is_nan());
    }
    out
}

/// Element-wise infinity test (any endpoint infinite).
pub fn is_inf_array(a: &IntervalArray) -> Vec<bool> {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let m = a_mids[i];
        let r = a_rads[i];
        out.push((m - r).is_infinite() || (m + r).is_infinite());
    }
    out
}

/// Element-wise finiteness test.
pub fn is_finite_array(a: &IntervalArray) -> Vec<bool> {
    is_nan_array(a)
        .iter()
        .zip(is_inf_array(a).iter())
        .map(|(&n, &i)| !n && !i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap_eq() {
        let a = IntervalArray::from_f64_slice(&[0.5]);
        let b = IntervalArray::from_f64_slice(&[0.75]);
        let c = Interval::exact(0.5 + 0.25);
        assert_eq!(cmp_interval(&Interval::exact(0.5), &c, Cmp::Eq), false);
        assert_eq!(
            cmp_interval(&Interval::exact(0.5), &Interval::exact(0.5), Cmp::Eq),
            true
        );
        let ab = compare_arrays(&a, &b, Cmp::Lt);
        assert_eq!(ab, vec![true]);
    }

    #[test]
    fn test_wide_overlap() {
        let x = Interval::from_midpoint_radius(1.0, 0.5); // [0.5, 1.5]
        let y = Interval::from_midpoint_radius(1.2, 0.5); // [0.7, 1.7]
        assert_eq!(cmp_interval(&x, &y, Cmp::Eq), true);
        assert_eq!(cmp_interval(&x, &y, Cmp::Lt), false);
        assert_eq!(cmp_interval(&y, &x, Cmp::Lt), false);
    }

    #[test]
    fn test_disjoint() {
        let x = Interval::exact(1.0);
        let y = Interval::exact(3.0);
        assert_eq!(cmp_interval(&x, &y, Cmp::Eq), false);
        assert_eq!(cmp_interval(&x, &y, Cmp::Lt), true);
        assert_eq!(cmp_interval(&x, &y, Cmp::Le), true);
        assert_eq!(cmp_interval(&y, &x, Cmp::Gt), true);
        assert_eq!(cmp_interval(&y, &x, Cmp::Ge), true);
    }
}
