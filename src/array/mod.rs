pub mod storage;

use std::sync::Arc;
use crate::error::Interval;
use storage::AlignedBuffer;

/// An N-dimensional array of intervals with shape, strides, and error tracking.
#[derive(Debug)]
pub struct IntervalArray {
    data: Arc<AlignedBuffer>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    total_len: usize,
}

impl IntervalArray {
    /// Create a 1D interval array from a Vec of Intervals.
    pub fn from_intervals(intervals: &[Interval]) -> Self {
        let len = intervals.len();
        let data = Arc::new(AlignedBuffer::from_intervals(intervals));
        let shape = vec![len];
        let strides = vec![1];
        Self {
            data,
            shape,
            strides,
            total_len: len,
        }
    }

    /// Create a 1D exact (zero-error) array from f64 values.
    pub fn from_f64_slice(values: &[f64]) -> Self {
        let len = values.len();
        let radii = vec![0.0f64; len];
        let data = Arc::new(AlignedBuffer::from_slices(values, &radii));
        Self {
            data,
            shape: vec![len],
            strides: vec![1],
            total_len: len,
        }
    }

    /// Create an N-dimensional exact array from f64 values with given shape.
    pub fn from_f64_vec(values: &[f64], shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(
            values.len(),
            total,
            "values length {} != product of shape {:?}",
            values.len(),
            shape
        );
        let radii = vec![0.0f64; total];
        let data = Arc::new(AlignedBuffer::from_slices(values, &radii));
        let strides = Self::compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
            total_len: total,
        }
    }

    /// Create directly from separate midpoint and radius slices with shape.
    pub fn from_raw_parts(midpoints: &[f64], radii: &[f64], shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(midpoints.len(), total);
        assert_eq!(radii.len(), total);
        let data = Arc::new(AlignedBuffer::from_slices(midpoints, radii));
        let strides = Self::compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
            total_len: total,
        }
    }

    /// Create an N-dimensional array of zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        let data = Arc::new(AlignedBuffer::new(total));
        let strides = Self::compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
            total_len: total,
        }
    }

    /// Create an N-dimensional array of ones (exact).
    pub fn ones(shape: &[usize]) -> Self {
        Self::full(shape, Interval::exact(1.0))
    }

    /// Create an N-dimensional array filled with a constant interval.
    pub fn full(shape: &[usize], value: Interval) -> Self {
        let total: usize = shape.iter().product();
        let mids = vec![value.midpoint(); total];
        let rads = vec![value.radius(); total];
        let data = Arc::new(AlignedBuffer::from_slices(&mids, &rads));
        let strides = Self::compute_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
            total_len: total,
        }
    }

    /// Create a 1D array with evenly spaced values.
    pub fn linspace(start: f64, stop: f64, num: usize) -> Self {
        if num == 0 {
            return Self::zeros(&[0]);
        }
        if num == 1 {
            return Self::from_f64_slice(&[start]);
        }
        let step = (stop - start) / (num - 1) as f64;
        let values: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
        Self::from_f64_slice(&values)
    }

    /// Create a 1D array with values from start to stop (exclusive) with given step.
    pub fn arange(start: f64, stop: f64, step: f64) -> Self {
        assert!(step != 0.0, "arange: step cannot be zero");
        let mut values = Vec::new();
        if step > 0.0 {
            let mut v = start;
            while v < stop {
                values.push(v);
                v += step;
            }
        } else {
            let mut v = start;
            while v > stop {
                values.push(v);
                v += step;
            }
        }
        Self::from_f64_slice(&values)
    }

    /// Compute strides from shape (row-major).
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        if ndim == 0 {
            return vec![];
        }
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements (cached).
    #[inline]
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Whether the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Shape of the array.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Strides of the array.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Whether this array has zero width (all exact intervals).
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.data.radii().iter().all(|&r| r == 0.0)
    }

    /// The maximum relative error across all elements.
    /// For elements with zero midpoint but nonzero radius, reports infinity.
    #[inline]
    pub fn max_relative_error(&self) -> f64 {
        let mids = self.data.midpoints();
        let rads = self.data.radii();
        let mut max_err = 0.0f64;
        for i in 0..self.total_len {
            if rads[i] == 0.0 {
                continue;
            }
            let abs_mid = mids[i].abs();
            if abs_mid > 0.0 {
                let rel = rads[i] / abs_mid;
                if rel > max_err {
                    max_err = rel;
                }
            } else {
                // Zero midpoint with nonzero radius: infinite relative error
                return f64::INFINITY;
            }
        }
        max_err
    }

    /// The maximum radius across all elements.
    #[inline]
    pub fn max_radius(&self) -> f64 {
        self.data
            .radii()
            .iter()
            .copied()
            .fold(0.0f64, f64::max)
    }

    /// Get interval at flat index.
    #[inline]
    pub fn get(&self, idx: usize) -> Interval {
        self.data.get_interval(idx)
    }

    /// Set interval at flat index.
    #[inline]
    pub fn set(&mut self, idx: usize, iv: Interval) {
        Arc::make_mut(&mut self.data).set_interval(idx, iv);
    }

    /// Get a reference to the underlying data buffer.
    #[inline]
    pub fn data(&self) -> &AlignedBuffer {
        &self.data
    }

    /// Get a mutable reference to the underlying data buffer.
    #[inline]
    pub fn data_mut(&mut self) -> &mut AlignedBuffer {
        Arc::make_mut(&mut self.data)
    }

    /// Reshape the array (must have same total elements).
    /// Zero-copy: only changes shape and strides metadata.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_total: usize = new_shape.iter().product();
        assert_eq!(
            self.total_len, new_total,
            "cannot reshape array of size {} into shape {:?}",
            self.total_len, new_shape
        );
        let new_strides = Self::compute_strides(new_shape);
        Self {
            data: Arc::clone(&self.data),
            shape: new_shape.to_vec(),
            strides: new_strides,
            total_len: new_total,
        }
    }

    /// Transpose 2D array.
    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose only supports 2D arrays");
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut result = Self::zeros(&[cols, rows]);
        for i in 0..rows {
            for j in 0..cols {
                let iv = self.data.get_interval(i * cols + j);
                result.set(j * rows + i, iv);
            }
        }
        result
    }
}

impl Clone for IntervalArray {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            total_len: self.total_len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f64_slice() {
        let arr = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.ndim(), 1);
        assert!(arr.is_exact());
        assert_eq!(arr.get(0).lo, 1.0);
    }

    #[test]
    fn test_zeros() {
        let arr = IntervalArray::zeros(&[3, 4]);
        assert_eq!(arr.len(), 12);
        assert_eq!(arr.shape(), &[3, 4]);
        assert!(arr.is_exact());
    }

    #[test]
    fn test_ones() {
        let arr = IntervalArray::ones(&[3]);
        assert_eq!(arr.len(), 3);
        assert!((arr.get(0).midpoint() - 1.0).abs() < 1e-15);
        assert!(arr.is_exact());
    }

    #[test]
    fn test_linspace() {
        let arr = IntervalArray::linspace(0.0, 1.0, 5);
        assert_eq!(arr.len(), 5);
        assert!((arr.get(0).midpoint() - 0.0).abs() < 1e-15);
        assert!((arr.get(4).midpoint() - 1.0).abs() < 1e-15);
        assert!((arr.get(2).midpoint() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_arange() {
        let arr = IntervalArray::arange(0.0, 5.0, 1.0);
        assert_eq!(arr.len(), 5);
        assert!((arr.get(0).midpoint() - 0.0).abs() < 1e-15);
        assert!((arr.get(4).midpoint() - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_empty_array() {
        let arr = IntervalArray::zeros(&[0]);
        assert_eq!(arr.len(), 0);
        assert!(arr.is_empty());
        assert!(arr.is_exact());
    }

    #[test]
    fn test_zero_dim_strides() {
        let strides = IntervalArray::compute_strides(&[]);
        assert!(strides.is_empty());
    }

    #[test]
    fn test_reshape() {
        let arr = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = arr.reshape(&[2, 3]);
        assert_eq!(reshaped.shape(), &[2, 3]);
        assert_eq!(reshaped.get(0).lo, 1.0);
        assert_eq!(reshaped.get(5).lo, 6.0);
    }

    #[test]
    fn test_transpose() {
        let arr = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t = arr.transpose();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.get(0).lo, 1.0);
        assert_eq!(t.get(1).lo, 4.0);
        assert_eq!(t.get(2).lo, 2.0);
    }

    #[test]
    fn test_clone() {
        let arr = IntervalArray::from_f64_slice(&[1.0, 2.0, 3.0]);
        let cloned = arr.clone();
        assert_eq!(cloned.len(), 3);
        assert_eq!(cloned.get(0).lo, 1.0);
        assert_eq!(cloned.get(2).lo, 3.0);
    }

    #[test]
    fn test_max_relative_error_zero_midpoint() {
        let arr = IntervalArray::from_intervals(&[Interval::from_midpoint_radius(0.0, 0.1)]);
        assert_eq!(arr.max_relative_error(), f64::INFINITY);
    }
}
