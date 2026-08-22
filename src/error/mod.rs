pub mod interval;

pub use interval::Interval;

/// Propagate error through a chain of interval operations.
///
/// Given a sequence of operations, the final interval is guaranteed
/// to contain the true mathematical result.
#[inline]
pub fn propagate_add(a: Interval, b: Interval) -> Interval {
    a + b
}

#[inline]
pub fn propagate_sub(a: Interval, b: Interval) -> Interval {
    a - b
}

#[inline]
pub fn propagate_mul(a: Interval, b: Interval) -> Interval {
    a * b
}

#[inline]
pub fn propagate_div(a: Interval, b: Interval) -> Interval {
    a / b
}

/// Compute the maximum relative error across a slice of intervals.
#[inline]
pub fn max_relative_error(intervals: &[Interval]) -> f64 {
    intervals
        .iter()
        .map(|i| i.relative_error())
        .fold(0.0_f64, f64::max)
}

/// Compute the sum of all intervals, accumulating error.
#[inline]
pub fn accumulate_sum(intervals: &[Interval]) -> Interval {
    intervals
        .iter()
        .copied()
        .fold(Interval::zero(), |acc, i| acc + i)
}
