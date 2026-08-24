use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ptr::NonNull;
use std::slice;

use crate::error::Interval;

/// Aligned memory buffer storing intervals in structure-of-arrays layout.
///
/// Memory layout: [midpoint_0, ..., midpoint_n-1, radius_0, ..., radius_n-1]
///
/// SoA layout enables SIMD to operate on contiguous blocks of midpoints/radii.
#[derive(Debug)]
pub struct AlignedBuffer {
    ptr: NonNull<f64>,
    len: usize,
    layout: Option<Layout>,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate a new buffer for `len` intervals (stores 2*len f64 values).
    /// Zero-length buffers use a dangling pointer to avoid UB.
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                layout: None,
            };
        }
        let total = len.checked_mul(2).expect("capacity overflow");
        let size = total.checked_mul(std::mem::size_of::<f64>()).expect("capacity overflow");
        let layout = Layout::from_size_align(size, 64)
            .expect("invalid layout");
        let ptr = unsafe { alloc_zeroed(layout) as *mut f64 };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len,
            layout: Some(layout),
        }
    }

    /// Create from raw midpoint and radius slices (copies data).
    pub fn from_intervals(intervals: &[Interval]) -> Self {
        let len = intervals.len();
        if len == 0 {
            return Self::new(0);
        }
        let buf = Self::new(len);
        let ptr = buf.ptr.as_ptr();
        // Write midpoints and radii using raw pointer arithmetic
        // to avoid simultaneous mutable borrows.
        for (i, iv) in intervals.iter().enumerate() {
            unsafe {
                *ptr.add(i) = iv.midpoint();
                *ptr.add(len + i) = iv.radius();
            }
        }
        buf
    }

    /// Create directly from separate midpoint and radius slices (bulk memcpy).
    pub fn from_slices(midpoints: &[f64], radii: &[f64]) -> Self {
        assert_eq!(
            midpoints.len(),
            radii.len(),
            "midpoints and radii must have same length"
        );
        let len = midpoints.len();
        if len == 0 {
            return Self::new(0);
        }
        let buf = Self::new(len);
        let ptr = buf.ptr.as_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(midpoints.as_ptr(), ptr, len);
            std::ptr::copy_nonoverlapping(radii.as_ptr(), ptr.add(len), len);
        }
        buf
    }

    /// Number of intervals stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Raw pointer to the midpoint array.
    #[inline]
    pub fn midpoints_ptr(&self) -> *const f64 {
        self.ptr.as_ptr()
    }

    /// Raw pointer to the radius array.
    #[inline]
    pub fn radii_ptr(&self) -> *const f64 {
        if self.len == 0 {
            return self.ptr.as_ptr();
        }
        unsafe { self.ptr.as_ptr().add(self.len) }
    }

    /// Borrow midpoints as a slice.
    #[inline]
    pub fn midpoints(&self) -> &[f64] {
        if self.len == 0 {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Borrow radii as a slice.
    #[inline]
    pub fn radii(&self) -> &[f64] {
        if self.len == 0 {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.ptr.as_ptr().add(self.len), self.len) }
    }

    /// Borrow midpoints mutably as a slice.
    #[inline]
    pub fn midpoints_mut(&mut self) -> &mut [f64] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Borrow radii mutably as a slice.
    #[inline]
    pub fn radii_mut(&mut self) -> &mut [f64] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().add(self.len), self.len) }
    }

    /// Borrow both midpoints and radii mutably simultaneously.
    /// Uses raw pointer arithmetic to avoid double mutable borrow.
    #[inline]
    pub fn as_mut_slices(&mut self) -> (&mut [f64], &mut [f64]) {
        if self.len == 0 {
            return (&mut [], &mut []);
        }
        unsafe {
            let mids = slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len);
            let rads = slice::from_raw_parts_mut(self.ptr.as_ptr().add(self.len), self.len);
            (mids, rads)
        }
    }

    /// Extract interval at index.
    #[inline]
    pub fn get_interval(&self, idx: usize) -> Interval {
        debug_assert!(idx < self.len);
        let ptr = self.ptr.as_ptr();
        let mid = unsafe { *ptr.add(idx) };
        let rad = unsafe { *ptr.add(self.len + idx) };
        Interval::from_midpoint_radius(mid, rad)
    }

    /// Set interval at index.
    #[inline]
    pub fn set_interval(&mut self, idx: usize, iv: Interval) {
        debug_assert!(idx < self.len);
        let mid = iv.midpoint();
        let rad = iv.radius();
        let ptr = self.ptr.as_ptr();
        unsafe {
            *ptr.add(idx) = mid;
            *ptr.add(self.len + idx) = rad;
        }
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        if self.len == 0 {
            return Self::new(0);
        }
        let new_buf = Self::new(self.len);
        let total = self.len.checked_mul(2).expect("capacity overflow");
        let total_bytes = total.checked_mul(std::mem::size_of::<f64>()).expect("capacity overflow");
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr() as *const u8,
                new_buf.ptr.as_ptr() as *mut u8,
                total_bytes,
            );
        }
        new_buf
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() {
        let buf = AlignedBuffer::new(100);
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.midpoints().len(), 100);
        assert_eq!(buf.radii().len(), 100);
        assert!(buf.midpoints().iter().all(|&x| x == 0.0));
        assert!(buf.radii().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_zero_length_buffer() {
        let buf = AlignedBuffer::new(0);
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.midpoints().len(), 0);
        assert_eq!(buf.radii().len(), 0);
    }

    #[test]
    fn test_from_intervals() {
        let intervals = vec![
            Interval::from_midpoint_radius(1.0, 0.1),
            Interval::from_midpoint_radius(2.0, 0.2),
            Interval::from_midpoint_radius(3.0, 0.3),
        ];
        let buf = AlignedBuffer::from_intervals(&intervals);
        assert!((buf.midpoints()[0] - 1.0).abs() < 1e-10);
        assert!((buf.midpoints()[1] - 2.0).abs() < 1e-10);
        assert!((buf.radii()[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_from_slices() {
        let mids = vec![1.0, 2.0, 3.0];
        let rads = vec![0.1, 0.2, 0.3];
        let buf = AlignedBuffer::from_slices(&mids, &rads);
        assert!((buf.midpoints()[0] - 1.0).abs() < 1e-10);
        assert!((buf.radii()[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_set_get_interval() {
        let mut buf = AlignedBuffer::new(10);
        let iv = Interval::from_midpoint_radius(5.5, 0.05);
        buf.set_interval(3, iv);
        let got = buf.get_interval(3);
        assert!((got.midpoint() - 5.5).abs() < 1e-10);
        assert!((got.radius() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_clone() {
        let intervals = vec![
            Interval::from_midpoint_radius(1.0, 0.1),
            Interval::from_midpoint_radius(2.0, 0.2),
        ];
        let buf = AlignedBuffer::from_intervals(&intervals);
        let cloned = buf.clone();
        assert_eq!(cloned.len(), 2);
        assert!((cloned.midpoints()[0] - 1.0).abs() < 1e-10);
        assert!((cloned.radii()[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_clone_zero_length() {
        let buf = AlignedBuffer::new(0);
        let cloned = buf.clone();
        assert_eq!(cloned.len(), 0);
    }

    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn test_new_overflow() {
        // This will panic when calculating total if it overflows usize,
        // or when calculating size if it overflows usize.
        let _ = AlignedBuffer::new(usize::MAX / 16 + 1);
    }

    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn test_new_overflow_2() {
        let _ = AlignedBuffer::new(usize::MAX / 2 + 1);
    }
}
