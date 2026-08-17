//! NumPy-style shape broadcasting helpers.

use crate::array::IntervalArray;

/// Compute the broadcast shape of two shapes, or `None` if incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut out = vec![1usize; ndim];
    for k in 0..ndim {
        let da = if k >= ndim - a.len() {
            a[k - (ndim - a.len())]
        } else {
            1
        };
        let db = if k >= ndim - b.len() {
            b[k - (ndim - b.len())]
        } else {
            1
        };
        if da == db {
            out[k] = da;
        } else if da == 1 {
            out[k] = db;
        } else if db == 1 {
            out[k] = da;
        } else {
            return None;
        }
    }
    Some(out)
}

/// Compute the broadcast shape of a list of shapes, or `None` if incompatible.
pub fn broadcast_shapes_many(shapes: &[&[usize]]) -> Option<Vec<usize>> {
    let mut acc: Option<Vec<usize>> = None;
    for s in shapes {
        acc = match acc {
            None => Some(s.to_vec()),
            Some(cur) => broadcast_shapes(&cur, s),
        };
    }
    acc
}

/// Broadcast an array to a target shape by materializing repeated values.
pub fn broadcast_to(a: &IntervalArray, shape: &[usize]) -> IntervalArray {
    debug_assert_eq!(a.len() == 1 || broadcast_shapes(a.shape(), shape).is_some(), true);
    let target_total: usize = shape.iter().product();
    let a_total = a.len();
    let n_src = a.shape().len();
    let n_dst = shape.len();

    let mut out_mids = vec![0.0f64; target_total];
    let mut out_rads = vec![0.0f64; target_total];

    // Special fast paths
    if shape == a.shape() {
        out_mids.copy_from_slice(a.data().midpoints());
        out_rads.copy_from_slice(a.data().radii());
        return IntervalArray::from_raw_parts(&out_mids, &out_rads, shape);
    }
    if a_total == 1 {
        let m = a.data().midpoints()[0];
        let r = a.data().radii()[0];
        out_mids.fill(m);
        out_rads.fill(r);
        return IntervalArray::from_raw_parts(&out_mids, &out_rads, shape);
    }

    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let a_strides = a.strides();

    for out_flat in 0..target_total {
        // Map output flat index -> multi-index -> source multi-index -> source flat
        let mut idx = out_flat;
        let mut src_flat = 0usize;
        let mut running = 1usize;
        for k in (0..n_dst).rev() {
            let coord = (idx / running) % shape[k];
            idx -= coord * running;
            if k >= n_dst - n_src {
                let src_k = k - (n_dst - n_src);
                let src_dim = a.shape()[src_k];
                if src_dim > 1 {
                    src_flat += coord * a_strides[src_k];
                }
            }
            running *= shape[k];
        }
        out_mids[out_flat] = a_mids[src_flat];
        out_rads[out_flat] = a_rads[src_flat];
    }

    IntervalArray::from_raw_parts(&out_mids, &out_rads, shape)
}

/// Broadcast two arrays to their common shape. Panics on incompatible shapes.
pub fn broadcast_pair(a: &IntervalArray, b: &IntervalArray) -> (IntervalArray, IntervalArray) {
    let shape = broadcast_shapes(a.shape(), b.shape())
        .expect("operands could not be broadcast together");
    (broadcast_to(a, &shape), broadcast_to(b, &shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes_basic() {
        assert_eq!(broadcast_shapes(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 1]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[2], &[3]), None);
    }

    #[test]
    fn test_broadcast_to_vector() {
        let a = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0]);
        let b = broadcast_to(&a, &[2, 3]);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.get(0).midpoint(), 1.0);
        assert_eq!(b.get(3).midpoint(), 1.0);
        assert_eq!(b.get(5).midpoint(), 3.0);
    }

    #[test]
    fn test_broadcast_to_scalar() {
        let a = IntervalArray::from_f64_slice(&[7.0]);
        let b = broadcast_to(&a, &[2, 2]);
        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(b.get(3).midpoint(), 7.0);
    }
}
