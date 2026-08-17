//! Additional array operations: power, rounding, clip, sort, concatenation,
//! axis reductions, and selection helpers.

use crate::array::IntervalArray;
use crate::error::Interval;
use crate::error::interval::{
    next_down, next_up, next_down_n, next_up_n, add_ru_chain, half_ulp, LIBSM_ULP_ALLOWANCE,
};

// ── Element-wise interval math ─────────────────────────────────────────

/// Interval power with an exact-integer fast path (repeated squaring with
/// directed rounding). For general exponents, the enclosure comes from
/// exp(b * ln(a)) with outward rounding, and the midpoint is refined with
/// the (accurately rounded) libm `powf`.
pub fn pow_interval(base: Interval, exp: Interval) -> Interval {
    if exp.is_exact() {
        let e = exp.lo;
        if e.fract() == 0.0 && e.abs() < 9.007_199_254_740_992e15 {
            let n = e as i64;
            if n == 0 {
                return Interval::exact(1.0);
            }
            let neg = n < 0;
            let k = n.unsigned_abs();
            let mut acc = Interval::exact(1.0);
            let mut x = base;
            let mut bits = k;
            while bits > 0 {
                if bits & 1 == 1 {
                    acc = acc * x;
                }
                bits >>= 1;
                if bits > 0 {
                    x = x * x;
                }
            }
            if neg {
                return acc.recip();
            }
            return acc;
        }
    }

    // base^exp = exp(exp * ln(base)) for base > 0
    if base.lo == 0.0 && base.hi == 0.0 {
        if exp.lo > 0.0 {
            return Interval::exact(0.0);
        }
        if exp.hi < 0.0 {
            return Interval::new(f64::INFINITY, f64::INFINITY);
        }
        return Interval::nan();
    }
    if base.hi <= 0.0 {
        return Interval::nan();
    }
    let ln_lo = if base.lo > 0.0 { base.lo } else { f64::MIN_POSITIVE };
    let ln_iv = Interval::new(
        next_down_n(ln_lo.ln(), LIBSM_ULP_ALLOWANCE),
        next_up_n(base.hi.ln(), LIBSM_ULP_ALLOWANCE),
    );
    let prod = exp * ln_iv;

    // Rigorous enclosure of exp(prod) with outward rounding (exp is not
    // correctly rounded on all platforms, hence the ulp allowance).
    let hi_raw = if prod.hi > 709.0 { f64::INFINITY } else { prod.hi.exp() };
    let lo_raw = if prod.lo < -745.0 { 0.0 } else { prod.lo.exp() };
    let lo = next_down_n(lo_raw, LIBSM_ULP_ALLOWANCE).max(0.0);
    let hi = next_up_n(hi_raw, LIBSM_ULP_ALLOWANCE);
    if lo.is_nan() || hi.is_nan() {
        return Interval::nan();
    }

    // Accurate midpoint via libm powf; the radius additionally covers the
    // powf libm error (up to LIBSM_ULP_ALLOWANCE ulps), so the interval is
    // centered tightly while remaining rigorous.
    let bm = base.midpoint();
    let em = exp.midpoint();
    let mut mid = if bm > 0.0 { bm.powf(em) } else { (lo + hi) * 0.5 };
    if mid.is_nan() || mid < lo || mid > hi {
        mid = (lo + hi) * 0.5;
        if mid.is_nan() {
            return Interval::new(lo, hi);
        }
    }
    let r1 = hi - mid;
    let e1 = crate::error::interval::two_sum_err(hi, -mid, r1);
    let r2 = mid - lo;
    let e2 = crate::error::interval::two_sum_err(mid, -lo, r2);
    let r3 = LIBSM_ULP_ALLOWANCE as f64 * half_ulp(mid);
    let rmax = r1.max(r2).max(r3);
    let mut emax = 0.0f64;
    if rmax == r1 {
        emax = emax.max(e1);
    }
    if rmax == r2 {
        emax = emax.max(e2);
    }
    let rad = add_ru_chain(rmax, emax);
    Interval::from_midpoint_radius(mid, rad)
}

/// Element-wise power of two arrays (must be broadcast to equal length).
pub fn pow_arrays(a: &IntervalArray, b: &IntervalArray) -> IntervalArray {
    assert_eq!(a.len(), b.len(), "power: length mismatch");
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let b_mids = b.data().midpoints();
    let b_rads = b.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for i in 0..n {
        let ai = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
        let bi = Interval::from_midpoint_radius(b_mids[i], b_rads[i]);
        let r = pow_interval(ai, bi);
        out_mids[i] = r.midpoint();
        out_rads[i] = r.radius();
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Array to an exact scalar power.
pub fn pow_scalar(a: &IntervalArray, e: f64) -> IntervalArray {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for i in 0..n {
        let ai = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
        let r = pow_interval(ai, Interval::exact(e));
        out_mids[i] = r.midpoint();
        out_rads[i] = r.radius();
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

/// Scalar to the array power.
pub fn rpow_scalar(a: &IntervalArray, base: f64) -> IntervalArray {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for i in 0..n {
        let ei = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
        let r = pow_interval(Interval::exact(base), ei);
        out_mids[i] = r.midpoint();
        out_rads[i] = r.radius();
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

// ── Rounding ───────────────────────────────────────────────────────────

pub fn floor_interval(iv: Interval, m: f64) -> (f64, f64) {
    centered_round(m.floor(), iv.lo.floor(), iv.hi.floor())
}

pub fn ceil_interval(iv: Interval, m: f64) -> (f64, f64) {
    centered_round(m.ceil(), iv.lo.ceil(), iv.hi.ceil())
}

pub fn trunc_interval(iv: Interval, m: f64) -> (f64, f64) {
    centered_round(m.trunc(), iv.lo.trunc(), iv.hi.trunc())
}

pub fn round_interval(iv: Interval, m: f64) -> (f64, f64) {
    centered_round(m.round_ties_even(), iv.lo.round_ties_even(), iv.hi.round_ties_even())
}

/// Build the stored (mid, rad) pair for an exactly-representable
/// rounding function: the center is the eval of the input midpoint and
/// the radius is the outward-rounded distance to the enclosure endpoints.
fn centered_round(mid: f64, lo_e: f64, hi_e: f64) -> (f64, f64) {
    if lo_e.is_nan() || hi_e.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let mut mid = mid;
    if !mid.is_finite() {
        mid = (lo_e + hi_e) * 0.5;
    }
    if mid < lo_e {
        mid = lo_e;
    }
    if mid > hi_e {
        mid = hi_e;
    }
    let rad = crate::error::interval::sub_ru(mid, lo_e)
        .max(crate::error::interval::sub_ru(hi_e, mid));
    (mid, rad)
}

// ── Clip / sign / nan_to_num ───────────────────────────────────────────

pub fn clip_interval(iv: Interval, lo: f64, hi: f64) -> Interval {
    let l = iv.lo.max(lo).min(hi);
    let h = iv.hi.max(lo).min(hi);
    Interval::new(l, h)
}

pub fn clip_array(a: &IntervalArray, lo: f64, hi: f64) -> IntervalArray {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for i in 0..n {
        let ai = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
        let r = clip_interval(ai, lo, hi);
        out_mids[i] = r.midpoint();
        out_rads[i] = r.radius();
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

pub fn sign_interval(iv: Interval) -> Interval {
    if iv.hi < 0.0 {
        Interval::exact(-1.0)
    } else if iv.lo > 0.0 {
        Interval::exact(1.0)
    } else {
        Interval::exact(0.0)
    }
}

pub fn sign_array(a: &IntervalArray) -> IntervalArray {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let out_rads = vec![0.0f64; n];
    for i in 0..n {
        let ai = Interval::from_midpoint_radius(a_mids[i], a_rads[i]);
        out_mids[i] = sign_interval(ai).midpoint();
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

pub fn nan_to_num_array(a: &IntervalArray) -> IntervalArray {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for i in 0..n {
        let mut m = a_mids[i];
        let mut r = a_rads[i];
        if m.is_nan() || r.is_nan() {
            m = 0.0;
            r = 0.0;
        } else {
            if m.is_infinite() {
                m = if m > 0.0 { f64::MAX } else { -f64::MAX };
            }
            if r.is_infinite() {
                r = f64::MAX;
            }
        }
        out_mids[i] = m;
        out_rads[i] = r;
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

// ── Sort / argsort ─────────────────────────────────────────────────────

/// Indices that sort the 1D array by midpoint (stable, radius tiebreak).
pub fn argsort(a: &IntervalArray) -> Vec<usize> {
    let n = a.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let mids = a.data().midpoints();
    let rads = a.data().radii();
    idx.sort_by(|&i, &j| {
        let (mi, ri) = (mids[i], rads[i]);
        let (mj, rj) = (mids[j], rads[j]);
        mi.partial_cmp(&mj).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ri.partial_cmp(&rj).unwrap_or(std::cmp::Ordering::Equal))
    });
    idx
}

pub fn sorted_array(a: &IntervalArray) -> IntervalArray {
    let idx = argsort(a);
    let n = a.len();
    let mids = a.data().midpoints();
    let rads = a.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for (k, &i) in idx.iter().enumerate() {
        out_mids[k] = mids[i];
        out_rads[k] = rads[i];
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape())
}

// ── Nonzero ────────────────────────────────────────────────────────────

/// Return per-axis index lists of nonzero elements (interval != 0).
pub fn nonzero(a: &IntervalArray) -> Vec<Vec<usize>> {
    let ndim = a.ndim();
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    let mids = a.data().midpoints();
    let rads = a.data().radii();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); ndim];
    for flat in 0..a.len() {
        if mids[flat] != 0.0 || rads[flat] != 0.0 {
            let idx = flat;
            for k in 0..ndim {
                out[k].push((idx / strides[k]) % shape[k]);
            }
        }
    }
    out
}

// ── Concatenation / stacking / splitting ───────────────────────────────

/// Concatenate arrays along an axis. All arrays must have the same ndim and
/// identical shapes except along `axis`.
pub fn concatenate(arrays: &[&IntervalArray], axis: usize) -> Result<IntervalArray, String> {
    if arrays.is_empty() {
        return Err("need at least one array to concatenate".to_string());
    }
    let ndim = arrays[0].ndim();
    if axis >= ndim {
        return Err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        ));
    }
    let mut out_shape = arrays[0].shape().to_vec();
    let mut total_axis = 0usize;
    for a in arrays {
        if a.ndim() != ndim {
            return Err("all input arrays must have the same number of dimensions".to_string());
        }
        for (j, &d) in a.shape().iter().enumerate() {
            if j != axis && d != out_shape[j] {
                return Err(format!(
                    "all the input array dimensions except for the concatenation axis must match exactly, but along dimension {} the array at index has size {} and the other size was {}",
                    j, d, out_shape[j]
                ));
            }
        }
        total_axis += a.shape()[axis];
    }
    out_shape[axis] = total_axis;

    let total: usize = out_shape.iter().product();
    let mut out_mids = vec![0.0f64; total];
    let mut out_rads = vec![0.0f64; total];
    let out_strides = IntervalArray::compute_strides_pub(&out_shape);

    let axis_stride = out_strides[axis];
    let mut out_offset = 0usize;
    for a in arrays {
        let inner = a.shape()[axis];
        let src_strides = a.strides();
        let src_shape = a.shape();
        for src_flat in 0..a.len() {
            let mut idx = src_flat;
            let mut out_flat = out_offset;
            let mut running = 1usize;
            for k in (0..ndim).rev() {
                let coord = (idx / running) % src_shape[k];
                idx -= coord * running;
                out_flat += coord * out_strides[k];
                running *= src_shape[k];
            }
            out_mids[out_flat] = a.data().midpoints()[src_flat];
            out_rads[out_flat] = a.data().radii()[src_flat];
        }
        out_offset += inner * axis_stride;
        let _ = src_strides;
    }

    Ok(IntervalArray::from_raw_parts(&out_mids, &out_rads, &out_shape))
}

/// Stack arrays along a new axis.
pub fn stack(arrays: &[&IntervalArray], axis: usize) -> Result<IntervalArray, String> {
    if arrays.is_empty() {
        return Err("need at least one array to stack".to_string());
    }
    let ndim = arrays[0].ndim();
    if axis > ndim {
        return Err(format!("axis {} is out of bounds", axis));
    }
    let reshaped: Vec<IntervalArray> = arrays
        .iter()
        .map(|a| {
            let mut s = a.shape().to_vec();
            s.insert(axis, 1);
            a.reshape(&s)
        })
        .collect();
    let refs: Vec<&IntervalArray> = reshaped.iter().collect();
    concatenate(&refs, axis)
}

/// Extract a contiguous range along one axis (materialized copy).
pub fn axis_range(a: &IntervalArray, axis: usize, start: usize, len: usize) -> IntervalArray {
    let mut shape = a.shape().to_vec();
    shape[axis] = len;
    let total: usize = shape.iter().product();
    let mut out_mids = vec![0.0f64; total];
    let mut out_rads = vec![0.0f64; total];
    let a_strides = a.strides();
    let a_shape = a.shape();
    let src_stride = a_strides[axis];
    for out_flat in 0..total {
        let mut idx = out_flat;
        let mut src_flat = start * src_stride;
        let mut running = 1usize;
        for k in (0..a.ndim()).rev() {
            let coord = (idx / running) % shape[k];
            idx -= coord * running;
            if k == axis {
                src_flat += coord * src_stride;
            } else if a_shape[k] > 1 {
                src_flat += coord * a_strides[k];
            }
            running *= shape[k];
        }
        out_mids[out_flat] = a.data().midpoints()[src_flat];
        out_rads[out_flat] = a.data().radii()[src_flat];
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, &shape)
}

/// Split an array into sections at the given indices along `axis`.
pub fn split(a: &IntervalArray, indices: &[usize], axis: usize) -> Result<Vec<IntervalArray>, String> {
    if axis >= a.ndim() {
        return Err(format!("axis {} is out of bounds", axis));
    }
    let dim = a.shape()[axis];
    for &i in indices {
        if i == 0 || i >= dim {
            return Err(format!(
                "array split does not result in an equal division; index {} out of range 1..={}",
                i,
                dim - 1
            ));
        }
    }
    let mut bounds: Vec<usize> = vec![0];
    bounds.extend_from_slice(indices);
    bounds.push(dim);
    let mut out = Vec::with_capacity(bounds.len() - 1);
    for w in bounds.windows(2) {
        out.push(axis_range(a, axis, w[0], w[1] - w[0]));
    }
    Ok(out)
}

// ── where() ────────────────────────────────────────────────────────────

/// Select elements from x/y based on a boolean mask (all same length).
pub fn where_select(cond: &[bool], x: &IntervalArray, y: &IntervalArray) -> IntervalArray {
    assert_eq!(x.len(), y.len(), "where: length mismatch");
    assert_eq!(x.len(), cond.len(), "where: condition length mismatch");
    let n = x.len();
    let x_mids = x.data().midpoints();
    let x_rads = x.data().radii();
    let y_mids = y.data().midpoints();
    let y_rads = y.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for i in 0..n {
        if cond[i] {
            out_mids[i] = x_mids[i];
            out_rads[i] = x_rads[i];
        } else {
            out_mids[i] = y_mids[i];
            out_rads[i] = y_rads[i];
        }
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, x.shape())
}

// ── Axis reductions ────────────────────────────────────────────────────

/// Reduce an array along one axis using an associative binary interval op.
pub fn reduce_axis<F: Fn(Interval, Interval) -> Interval + Sync>(
    a: &IntervalArray,
    axis: usize,
    init: Interval,
    f: F,
) -> IntervalArray {
    let ndim = a.ndim();
    assert!(axis < ndim, "axis out of bounds");
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    let dim = shape[axis];
    let mut out_shape = shape.clone();
    out_shape.remove(axis);
    let out_total: usize = out_shape.iter().product();
    let mut out_mids = vec![0.0f64; out_total];
    let mut out_rads = vec![0.0f64; out_total];

    for out_flat in 0..out_total {
        let mut idx = out_flat;
        let mut base = 0usize;
        let mut running = 1usize;
        for k in (0..out_shape.len()).rev() {
            let j = if k < axis { k } else { k + 1 };
            let coord = (idx / running) % out_shape[k];
            idx -= coord * running;
            base += coord * strides[j];
            running *= out_shape[k];
        }
        let mut acc = init;
        for t in 0..dim {
            acc = f(acc, a.get(base + t * strides[axis]));
        }
        out_mids[out_flat] = acc.midpoint();
        out_rads[out_flat] = acc.radius();
    }

    IntervalArray::from_raw_parts(&out_mids, &out_rads, &out_shape)
}

/// Sum along an axis.
pub fn sum_axis(a: &IntervalArray, axis: usize) -> IntervalArray {
    reduce_axis(a, axis, Interval::zero(), |acc, x| acc + x)
}

/// Product along an axis.
pub fn prod_axis(a: &IntervalArray, axis: usize) -> IntervalArray {
    reduce_axis(a, axis, Interval::exact(1.0), |acc, x| acc * x)
}

/// Min along an axis (hull of elementwise minima).
pub fn min_axis(a: &IntervalArray, axis: usize) -> IntervalArray {
    reduce_axis(a, axis, Interval::new(f64::INFINITY, f64::INFINITY), |acc, x| {
        Interval::new(acc.lo.min(x.lo), acc.hi.min(x.hi))
    })
}

/// Max along an axis.
pub fn max_axis(a: &IntervalArray, axis: usize) -> IntervalArray {
    reduce_axis(
        a,
        axis,
        Interval::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        |acc, x| Interval::new(acc.lo.max(x.lo), acc.hi.max(x.hi)),
    )
}

/// Mean along an axis (rigorous: interval division by the axis length).
pub fn mean_axis(a: &IntervalArray, axis: usize) -> IntervalArray {
    let dim = a.shape()[axis] as f64;
    let s = sum_axis(a, axis);
    let n = s.len();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for i in 0..n {
        let iv = s.get(i) / dim;
        out_mids[i] = iv.midpoint();
        out_rads[i] = iv.radius();
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, s.shape())
}

/// Population variance along an axis.
pub fn var_axis(a: &IntervalArray, axis: usize) -> IntervalArray {
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    let dim = shape[axis] as f64;
    let mut out_shape = shape.clone();
    out_shape.remove(axis);
    let out_total: usize = out_shape.iter().product();

    let mut mean_mids = vec![0.0f64; out_total];
    let mut mean_rads = vec![0.0f64; out_total];
    for out_flat in 0..out_total {
        let mut idx = out_flat;
        let mut base = 0usize;
        let mut running = 1usize;
        for k in (0..out_shape.len()).rev() {
            let j = if k < axis { k } else { k + 1 };
            let coord = (idx / running) % out_shape[k];
            idx -= coord * running;
            base += coord * strides[j];
            running *= out_shape[k];
        }
        let mut s = Interval::zero();
        for t in 0..shape[axis] {
            s = s + a.get(base + t * strides[axis]);
        }
        mean_mids[out_flat] = s.midpoint() / dim;
        mean_rads[out_flat] = s.radius() / dim;
    }

    let mut out_mids = vec![0.0f64; out_total];
    let mut out_rads = vec![0.0f64; out_total];
    for out_flat in 0..out_total {
        let mut idx = out_flat;
        let mut base = 0usize;
        let mut running = 1usize;
        for k in (0..out_shape.len()).rev() {
            let j = if k < axis { k } else { k + 1 };
            let coord = (idx / running) % out_shape[k];
            idx -= coord * running;
            base += coord * strides[j];
            running *= out_shape[k];
        }
        let mean_iv = Interval::from_midpoint_radius(mean_mids[out_flat], mean_rads[out_flat]);
        let mut sum_sq = Interval::zero();
        for t in 0..shape[axis] {
            let x = a.get(base + t * strides[axis]);
            let dev = x - mean_iv;
            sum_sq = sum_sq + dev * dev;
        }
        let v = sum_sq / dim;
        let v = if v.lo < 0.0 {
            Interval::new(0.0, v.hi.max(0.0))
        } else {
            v
        };
        out_mids[out_flat] = v.midpoint();
        out_rads[out_flat] = v.radius();
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, &out_shape)
}

/// Population standard deviation along an axis (rigorous).
pub fn std_axis(a: &IntervalArray, axis: usize) -> IntervalArray {
    let v = var_axis(a, axis);
    let n = v.len();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    for i in 0..n {
        let iv = v.get(i);
        let lo = if iv.lo > 0.0 { next_down(iv.lo.sqrt()) } else { 0.0 };
        let hi = if iv.hi > 0.0 { next_up(iv.hi.sqrt()) } else { 0.0 };
        let r = Interval::new(lo, hi);
        out_mids[i] = r.midpoint();
        out_rads[i] = r.radius();
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, v.shape())
}

/// Cumulative sum along an axis.
pub fn cumsum_axis(a: &IntervalArray, axis: usize) -> IntervalArray {
    let ndim = a.ndim();
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    let dim = shape[axis];
    let total: usize = shape.iter().product();
    let mut out_mids = vec![0.0f64; total];
    let mut out_rads = vec![0.0f64; total];
    let mut out_shape = shape.clone();
    out_shape.remove(axis);
    let out_total: usize = out_shape.iter().product();

    for out_flat in 0..out_total {
        let mut idx = out_flat;
        let mut base = 0usize;
        let mut running = 1usize;
        for k in (0..out_shape.len()).rev() {
            let j = if k < axis { k } else { k + 1 };
            let coord = (idx / running) % out_shape[k];
            idx -= coord * running;
            base += coord * strides[j];
            running *= out_shape[k];
        }
        let mut cum = Interval::zero();
        for t in 0..dim {
            let src = base + t * strides[axis];
            cum = cum + a.get(src);
            // Output flat index: out_flat's multi-index with `t` inserted at axis
            let mut mi = vec![0usize; ndim];
            let rem = out_flat;
            for k in 0..out_shape.len() {
                let j = if k < axis { k } else { k + 1 };
                let len_after: usize = out_shape[k + 1..].iter().product();
                mi[j] = (rem / len_after) % out_shape[k];
            }
            mi[axis] = t;
            let mut of = 0usize;
            let mut acc_stride = 1usize;
            for k in (0..ndim).rev() {
                of += mi[k] * acc_stride;
                acc_stride *= shape[k];
            }
            out_mids[of] = cum.midpoint();
            out_rads[of] = cum.radius();
        }
    }
    IntervalArray::from_raw_parts(&out_mids, &out_rads, &shape)
}

/// Index of the extreme element along an axis (by midpoint, then radius).
pub fn arg_extreme_axis(a: &IntervalArray, axis: usize, want_max: bool) -> Vec<usize> {
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    let dim = shape[axis];
    let mut out_shape = shape.clone();
    out_shape.remove(axis);
    let out_total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(out_total);

    let mids = a.data().midpoints();
    let rads = a.data().radii();

    for out_flat in 0..out_total {
        let mut idx = out_flat;
        let mut base = 0usize;
        let mut running = 1usize;
        for k in (0..out_shape.len()).rev() {
            let j = if k < axis { k } else { k + 1 };
            let coord = (idx / running) % out_shape[k];
            idx -= coord * running;
            base += coord * strides[j];
            running *= out_shape[k];
        }
        let mut best = base;
        for t in 1..dim {
            let cur = base + t * strides[axis];
            let better = if want_max {
                (mids[cur], rads[cur]) > (mids[best], rads[best])
            } else {
                (mids[cur], rads[cur]) < (mids[best], rads[best])
            };
            if better {
                best = cur;
            }
        }
        // index along axis: (best - base) / strides[axis]
        out.push((best - base) / strides[axis]);
    }
    out
}

/// Flat index of the extreme element (first occurrence).
pub fn arg_extreme_flat(a: &IntervalArray, want_max: bool) -> usize {
    let n = a.len();
    assert!(n > 0, "attempt to get argmax/argmin of an empty array");
    let mids = a.data().midpoints();
    let rads = a.data().radii();
    let mut best = 0usize;
    for i in 1..n {
        let better = if want_max {
            (mids[i], rads[i]) > (mids[best], rads[best])
        } else {
            (mids[i], rads[i]) < (mids[best], rads[best])
        };
        if better {
            best = i;
        }
    }
    best
}

/// Reduce a boolean predicate along an axis (all/any of nonzero-ness).
pub fn all_any_axis(a: &IntervalArray, axis: usize, all: bool) -> Vec<bool> {
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    let dim = shape[axis];
    let mut out_shape = shape.clone();
    out_shape.remove(axis);
    let out_total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(out_total);

    let mids = a.data().midpoints();
    let rads = a.data().radii();

    for out_flat in 0..out_total {
        let mut idx = out_flat;
        let mut base = 0usize;
        let mut running = 1usize;
        for k in (0..out_shape.len()).rev() {
            let j = if k < axis { k } else { k + 1 };
            let coord = (idx / running) % out_shape[k];
            idx -= coord * running;
            base += coord * strides[j];
            running *= out_shape[k];
        }
        let mut acc = all;
        for t in 0..dim {
            let src = base + t * strides[axis];
            let nonzero = mids[src] != 0.0 || rads[src] != 0.0;
            acc = if all { acc && nonzero } else { acc || nonzero };
            if !all && acc {
                break;
            }
            if all && !acc {
                break;
            }
        }
        out.push(acc);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_integer() {
        let x = Interval::exact(2.0);
        let r = pow_interval(x, Interval::exact(10.0));
        assert!((r.midpoint() - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_pow_negative_integer() {
        let x = Interval::exact(2.0);
        let r = pow_interval(x, Interval::exact(-2.0));
        assert!((r.midpoint() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_pow_zero() {
        let x = Interval::exact(5.0);
        let r = pow_interval(x, Interval::exact(0.0));
        assert_eq!(r.midpoint(), 1.0);
    }

    #[test]
    fn test_pow_fractional() {
        let x = Interval::exact(4.0);
        let r = pow_interval(x, Interval::exact(0.5));
        assert!((r.midpoint() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip() {
        let arr = IntervalArray::from_f64_slice(&[-2.0, 0.5, 3.0]);
        let c = clip_array(&arr, 0.0, 2.0);
        assert_eq!(c.get(0).midpoint(), 0.0);
        assert_eq!(c.get(1).midpoint(), 0.5);
        assert_eq!(c.get(2).midpoint(), 2.0);
    }

    #[test]
    fn test_sign() {
        let arr = IntervalArray::from_f64_slice(&[-2.0, 0.0, 3.0]);
        let s = sign_array(&arr);
        assert_eq!(s.get(0).midpoint(), -1.0);
        assert_eq!(s.get(1).midpoint(), 0.0);
        assert_eq!(s.get(2).midpoint(), 1.0);
    }

    #[test]
    fn test_sum_axis() {
        let a = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s0 = sum_axis(&a, 0);
        assert_eq!(s0.shape(), &[2]);
        assert!((s0.get(0).midpoint() - 4.0).abs() < 1e-10);
        assert!((s0.get(1).midpoint() - 6.0).abs() < 1e-10);
        let s1 = sum_axis(&a, 1);
        assert!((s1.get(0).midpoint() - 3.0).abs() < 1e-10);
        assert!((s1.get(1).midpoint() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_axis() {
        let a = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let m = mean_axis(&a, 0);
        assert!((m.get(0).midpoint() - 2.0).abs() < 1e-10);
        assert!((m.get(1).midpoint() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumsum_axis() {
        let a = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = cumsum_axis(&a, 1);
        assert!((c.get(0).midpoint() - 1.0).abs() < 1e-10);
        assert!((c.get(1).midpoint() - 3.0).abs() < 1e-10);
        assert!((c.get(2).midpoint() - 3.0).abs() < 1e-10);
        assert!((c.get(3).midpoint() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_arg_extreme() {
        let a = IntervalArray::from_f64_slice(&[3.0, 1.0, 2.0]);
        assert_eq!(arg_extreme_flat(&a, true), 0);
        assert_eq!(arg_extreme_flat(&a, false), 1);
        let m = IntervalArray::from_f64_vec(&[3.0, 1.0, 2.0, 4.0], &[2, 2]);
        let ax = arg_extreme_axis(&m, 0, true);
        assert_eq!(ax, vec![0, 1]);
        let ax = arg_extreme_axis(&m, 1, false);
        assert_eq!(ax, vec![1, 0]);
    }

    #[test]
    fn test_concatenate() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0]);
        let b = IntervalArray::from_f64_slice(&[3.0, 4.0, 5.0]);
        let refs = vec![&a, &b];
        let c = concatenate(&refs, 0).unwrap();
        assert_eq!(c.shape(), &[5]);
        assert_eq!(c.get(4).midpoint(), 5.0);
    }

    #[test]
    fn test_concatenate_2d_axis1() {
        let a = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = IntervalArray::from_f64_vec(&[5.0, 6.0], &[2, 1]);
        let refs = vec![&a, &b];
        let c = concatenate(&refs, 1).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.get(0).midpoint(), 1.0);
        assert_eq!(c.get(2).midpoint(), 5.0);
        assert_eq!(c.get(4).midpoint(), 4.0);
        assert_eq!(c.get(5).midpoint(), 6.0);
    }

    #[test]
    fn test_stack() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0]);
        let b = IntervalArray::from_f64_slice(&[3.0, 4.0]);
        let refs = vec![&a, &b];
        let c = stack(&refs, 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(2).midpoint(), 3.0);
    }

    #[test]
    fn test_split() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0, 4.0]);
        let parts = split(&a, &[2], 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 2);
        assert_eq!(parts[1].len(), 2);
        assert_eq!(parts[1].get(1).midpoint(), 4.0);
    }

    #[test]
    fn test_argsort() {
        let a = IntervalArray::from_f64_slice(&[3.0, 1.0, 2.0]);
        assert_eq!(argsort(&a), vec![1, 2, 0]);
        let s = sorted_array(&a);
        assert_eq!(s.get(0).midpoint(), 1.0);
    }

    #[test]
    fn test_nonzero() {
        let a = IntervalArray::from_f64_vec(&[0.0, 1.0, 0.0, 2.0], &[2, 2]);
        let nz = nonzero(&a);
        assert_eq!(nz[0], vec![0, 1]);
        assert_eq!(nz[1], vec![1, 1]);
    }

    #[test]
    fn test_where() {
        let x = IntervalArray::from_f64_slice(&[1.0, 2.0]);
        let y = IntervalArray::from_f64_slice(&[3.0, 4.0]);
        let cond = vec![true, false];
        let r = where_select(&cond, &x, &y);
        assert_eq!(r.get(0).midpoint(), 1.0);
        assert_eq!(r.get(1).midpoint(), 4.0);
    }
}
