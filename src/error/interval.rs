use std::fmt;

/// A hardware-rounded interval [lo, hi] guaranteed to contain the true value.
///
/// Invariant: lo <= hi always holds.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// Exact interval containing a single value with zero width.
    #[inline]
    pub fn exact(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    /// Interval [lo, hi] with lo <= hi enforced.
    #[inline]
    pub fn new(lo: f64, hi: f64) -> Self {
        if lo <= hi {
            Self { lo, hi }
        } else {
            Self { lo: hi, hi: lo }
        }
    }

    /// Create an interval from a midpoint and radius.
    ///
    /// The endpoints are computed with outward (directed) rounding so that
    /// the resulting [lo, hi] is guaranteed to contain {x : |x - mid| <= radius}.
    #[inline]
    pub fn from_midpoint_radius(mid: f64, radius: f64) -> Self {
        debug_assert!(radius >= 0.0 || radius.is_nan());
        if mid.is_nan() || radius.is_nan() {
            return Self {
                lo: f64::NAN,
                hi: f64::NAN,
            };
        }
        if radius == f64::INFINITY {
            return Interval::entire();
        }
        let lo = sub_rd(mid, radius);
        let hi = add_ru(mid, radius);
        Self { lo, hi }
    }

    /// The zero-width interval [0, 0].
    #[inline]
    pub fn zero() -> Self {
        Self { lo: 0.0, hi: 0.0 }
    }

    /// The interval containing all reals [-inf, +inf].
    #[inline]
    pub fn entire() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// NaN interval (empty/invalid).
    #[inline]
    pub fn nan() -> Self {
        Self {
            lo: f64::NAN,
            hi: f64::NAN,
        }
    }

    /// Midpoint of the interval.
    #[inline]
    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) * 0.5
    }

    /// Radius (half-width) of the interval, rounded outward so that
    /// {x : |x - midpoint()| <= radius()} always contains [lo, hi].
    #[inline]
    pub fn radius(&self) -> f64 {
        if self.lo == self.hi {
            return 0.0;
        }
        if !self.lo.is_finite() || !self.hi.is_finite() {
            return f64::INFINITY;
        }
        let mid = self.midpoint();
        if !mid.is_finite() {
            // lo + hi overflowed to +inf or -inf; the interval is finite but
            // wider than any representable radius, so return +inf.
            return f64::INFINITY;
        }
        // hi - mid and mid - lo with exact residuals (TwoSum), then take the
        // larger value with its residual and round the radius up.
        let s1 = self.hi - mid;
        let e1 = two_sum_err(self.hi, -mid, s1);
        let s2 = mid - self.lo;
        let e2 = two_sum_err(mid, -self.lo, s2);
        if s1 >= s2 {
            ru(s1, e1)
        } else {
            ru(s2, e2)
        }
    }

    /// Width of the interval (hi - lo).
    #[inline]
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Relative error: radius / |midpoint|, or 0.0 for exact zero.
    #[inline]
    pub fn relative_error(&self) -> f64 {
        let mid = self.midpoint();
        let abs_mid = mid.abs();
        if abs_mid == 0.0 {
            if self.radius() == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.radius() / abs_mid
        }
    }

    /// Whether this is an exact (zero-width) interval.
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.lo == self.hi
    }

    /// Whether the interval contains a specific value.
    #[inline]
    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    /// The intersection of two intervals (may be empty).
    #[inline]
    pub fn intersect(&self, other: &Interval) -> Option<Interval> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi {
            Some(Interval { lo, hi })
        } else {
            None
        }
    }

    /// The convex hull (smallest enclosing interval) of two intervals.
    #[inline]
    pub fn hull(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Reciprocal with directed rounding.
    #[inline]
    pub fn recip(self) -> Self {
        if self.lo <= 0.0 && self.hi >= 0.0 {
            return Interval::entire();
        }
        // 1/[a,b] = [1/b, 1/a] with directed rounding
        let new_lo = div_rd(1.0, self.hi);
        let new_hi = div_ru(1.0, self.lo);
        Interval {
            lo: new_lo,
            hi: new_hi,
        }
    }
}

// ── Arithmetic with directed rounding ──────────────────────────────────

impl std::ops::Add for Interval {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: add_rd(self.lo, rhs.lo),
            hi: add_ru(self.hi, rhs.hi),
        }
    }
}

impl std::ops::Sub for Interval {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            lo: sub_rd(self.lo, rhs.hi),
            hi: sub_ru(self.hi, rhs.lo),
        }
    }
}

impl std::ops::Mul for Interval {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // All four products with directed rounding; the min/max combine them.
        let (lo, hi);

        lo = mul_rd(self.lo, rhs.lo)
            .min(mul_rd(self.lo, rhs.hi))
            .min(mul_rd(self.hi, rhs.lo))
            .min(mul_rd(self.hi, rhs.hi));

        hi = mul_ru(self.lo, rhs.lo)
            .max(mul_ru(self.lo, rhs.hi))
            .max(mul_ru(self.hi, rhs.lo))
            .max(mul_ru(self.hi, rhs.hi));

        Self { lo, hi }
    }
}

impl std::ops::Div for Interval {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        if rhs.lo <= 0.0 && rhs.hi >= 0.0 {
            return Interval::entire();
        }
        // Use properly rounded reciprocal
        self * rhs.recip()
    }
}

impl std::ops::Neg for Interval {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

// ── Scalar ops ─────────────────────────────────────────────────────────

impl std::ops::Add<f64> for Interval {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        Self {
            lo: add_rd(self.lo, rhs),
            hi: add_ru(self.hi, rhs),
        }
    }
}

impl std::ops::Sub<f64> for Interval {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Self {
            lo: sub_rd(self.lo, rhs),
            hi: sub_ru(self.hi, rhs),
        }
    }
}

impl std::ops::Mul<f64> for Interval {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        if rhs >= 0.0 {
            Self {
                lo: mul_rd(self.lo, rhs),
                hi: mul_ru(self.hi, rhs),
            }
        } else {
            Self {
                lo: mul_rd(self.hi, rhs),
                hi: mul_ru(self.lo, rhs),
            }
        }
    }
}

impl std::ops::Div<f64> for Interval {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        // Use directed rounding for the reciprocal
        if rhs > 0.0 {
            Self {
                lo: div_rd(self.lo, rhs),
                hi: div_ru(self.hi, rhs),
            }
        } else if rhs < 0.0 {
            Self {
                lo: div_rd(self.hi, rhs),
                hi: div_ru(self.lo, rhs),
            }
        } else {
            // Division by zero
            Interval::entire()
        }
    }
}

// ── Directed rounding helpers ──────────────────────────────────────────

/// Exact TwoSum error of fl(a + b): returns e such that a + b = fl(a + b) + e.
#[inline]
pub(crate) fn two_sum_err(a: f64, b: f64, s: f64) -> f64 {
    let bv = s - a;
    let av = s - bv;
    let br = b - bv;
    let ar = a - av;
    ar + br
}

/// Round a value upward given an exact residual: if v = s + e with s the
/// round-to-nearest evaluation, returns fl_ru(v).
#[inline]
fn ru(s: f64, e: f64) -> f64 {
    let t = s + e;
    if !t.is_finite() {
        return t;
    }
    let e2 = two_sum_err(s, e, t);
    if e2 > 0.0 {
        next_up(t)
    } else {
        t
    }
}

/// Round a value downward given an exact residual: if v = s + e with s the
/// round-to-nearest evaluation, returns fl_rd(v).
#[inline]
fn rd(s: f64, e: f64) -> f64 {
    let t = s + e;
    if !t.is_finite() {
        return t;
    }
    let e2 = two_sum_err(s, e, t);
    if e2 < 0.0 {
        next_down(t)
    } else {
        t
    }
}

/// Add with round-down (toward -infinity). Exact; no rounding-mode changes.
///
/// e = (a + b) - s is the exact rounding residual: if e < 0 the true sum
/// lies below s (including exact-tie cases where RTN picked the upper
/// neighbor), so the directed result moves one ulp down.
#[inline]
pub(crate) fn add_rd(a: f64, b: f64) -> f64 {
    let s = a + b;
    if s.is_finite() {
        let e = two_sum_err(a, b, s);
        if e < 0.0 {
            next_down(s)
        } else {
            s
        }
    } else if s == f64::INFINITY {
        f64::MAX
    } else {
        s
    }
}

/// Add with round-up (toward +infinity). Exact; no rounding-mode changes.
#[inline]
pub(crate) fn add_ru(a: f64, b: f64) -> f64 {
    let s = a + b;
    if s.is_finite() {
        let e = two_sum_err(a, b, s);
        if e > 0.0 {
            next_up(s)
        } else {
            s
        }
    } else if s == f64::NEG_INFINITY {
        f64::MIN
    } else {
        s
    }
}

/// Subtract with round-down.
#[inline]
pub(crate) fn sub_rd(a: f64, b: f64) -> f64 {
    add_rd(a, -b)
}

/// Subtract with round-up.
#[inline]
pub(crate) fn sub_ru(a: f64, b: f64) -> f64 {
    add_ru(a, -b)
}

/// Multiply with round-down. Exact; uses the FMA residual (mul_add is
/// exact on all platforms: it returns the correctly rounded result of
/// a*b + c, and the residual a*b - fl(a*b) is exactly representable).
#[inline]
pub(crate) fn mul_rd(a: f64, b: f64) -> f64 {
    let s = a * b;
    if s.is_finite() {
        let e = -a.mul_add(b, -s);
        if e < 0.0 {
            next_down(s)
        } else {
            s
        }
    } else if s == f64::INFINITY {
        f64::MAX
    } else {
        s
    }
}

/// Multiply with round-up.
#[inline]
pub(crate) fn mul_ru(a: f64, b: f64) -> f64 {
    let s = a * b;
    if s.is_finite() {
        let e = -a.mul_add(b, -s);
        if e > 0.0 {
            next_up(s)
        } else {
            s
        }
    } else if s == f64::NEG_INFINITY {
        f64::MIN
    } else {
        s
    }
}

/// Divide with round-down. The sign of the exact error b*s - a tells
/// whether s = fl(a/b) lies above or below the exact quotient; when
/// the quotient is exact (e == 0) the evaluation is returned as-is.
#[inline]
fn div_rd(a: f64, b: f64) -> f64 {
    let s = a / b;
    if s.is_finite() {
        let e = b.mul_add(s, -a);
        if e != 0.0 && (e > 0.0) == (b > 0.0) {
            next_down(s)
        } else {
            s
        }
    } else if s == f64::INFINITY {
        f64::MAX
    } else {
        s
    }
}

/// Divide with round-up. The sign of the exact error b*s - a tells
/// whether s = fl(a/b) lies above or below the exact quotient; when
/// the quotient is exact (e == 0) the evaluation is returned as-is.
#[inline]
pub(crate) fn div_ru(a: f64, b: f64) -> f64 {
    let s = a / b;
    if s.is_finite() {
        let e = b.mul_add(s, -a);
        if e != 0.0 && (e > 0.0) != (b > 0.0) {
            next_up(s)
        } else {
            s
        }
    } else if s == f64::NEG_INFINITY {
        f64::MIN
    } else {
        s
    }
}

/// Directed (round-up) chained addition: fl_ru(x + y) given both values.
#[inline]
pub(crate) fn add_ru_chain(x: f64, y: f64) -> f64 {
    let s = x + y;
    if !s.is_finite() {
        return s;
    }
    let e = two_sum_err(x, y, s);
    if e > 0.0 {
        next_up(s)
    } else {
        s
    }
}

/// The successor of x (toward +infinity). Exact, works for ±0, ±inf, NaN.
#[inline]
pub(crate) fn next_up(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// The predecessor of x (toward -infinity). Exact, works for ±0, ±inf, NaN.
#[inline]
pub(crate) fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// The result of applying `next_up` n times.
#[inline]
pub(crate) fn next_up_n(x: f64, n: u32) -> f64 {
    let mut r = x;
    for _ in 0..n {
        r = next_up(r);
    }
    r
}

/// The result of applying `next_down` n times.
#[inline]
pub(crate) fn next_down_n(x: f64, n: u32) -> f64 {
    let mut r = x;
    for _ in 0..n {
        r = next_down(r);
    }
    r
}

/// Half of the spacing between adjacent floats at x (0.5 ulp), exact.
/// Returns 0.0 for x == 0, and inf for x == ±inf.
#[inline]
pub(crate) fn half_ulp(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() {
        return 0.0;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    (next_up(x.abs()) - x.abs()) * 0.5
}

/// Number of ulps reserved for libm (CRT) function error on platforms
/// without correctly-rounded transcendentals (notably MSVC's pow/exp/ln).
/// IEEE-754 mandates correct rounding only for the basic operations and
/// sqrt, so enclosures built from libm evaluations are expanded by this
/// many ulps to stay rigorous in practice.
pub(crate) const LIBSM_ULP_ALLOWANCE: u32 = 4;

// ── Display ────────────────────────────────────────────────────────────

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_exact() {
            write!(f, "{}", self.lo)
        } else {
            write!(f, "[{}, {}]", self.lo, self.hi)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact() {
        let pi = std::f64::consts::PI;
        let i = Interval::exact(pi);
        assert_eq!(i.lo, pi);
        assert_eq!(i.hi, pi);
        assert!(i.is_exact());
    }

    #[test]
    fn test_from_midpoint_radius() {
        let i = Interval::from_midpoint_radius(1.0, 0.5);
        assert!((i.lo - 0.5).abs() < 1e-15);
        assert!((i.hi - 1.5).abs() < 1e-15);
    }

    #[test]
    fn test_addition() {
        let a = Interval::exact(1.0);
        let b = Interval::exact(2.0);
        let c = a + b;
        assert!(c.lo <= 3.0);
        assert!(c.hi >= 3.0);
    }

    #[test]
    fn test_subtraction() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(0.5, 1.5);
        let c = a - b;
        assert!(c.lo <= -0.5);
        assert!(c.hi >= 2.5);
    }

    #[test]
    fn test_multiplication_positive() {
        let a = Interval::exact(2.0);
        let b = Interval::exact(3.0);
        let c = a * b;
        assert!(c.lo <= 6.0);
        assert!(c.hi >= 6.0);
    }

    #[test]
    fn test_division_by_zero_produces_entire() {
        let a = Interval::exact(1.0);
        let b = Interval::new(-1.0, 1.0);
        let c = a / b;
        assert_eq!(c.lo, f64::NEG_INFINITY);
        assert_eq!(c.hi, f64::INFINITY);
    }

    #[test]
    fn test_division_positive() {
        let a = Interval::exact(6.0);
        let b = Interval::exact(3.0);
        let c = a / b;
        assert!(c.lo <= 2.0);
        assert!(c.hi >= 2.0);
    }

    #[test]
    fn test_scalar_div_rounding() {
        let a = Interval::new(1.0, 2.0);
        let c = a / 3.0;
        // Result should contain the true interval [1/3, 2/3]
        assert!(c.lo <= 1.0 / 3.0);
        assert!(c.hi >= 2.0 / 3.0);
    }

    #[test]
    fn test_scalar_div_by_zero() {
        let a = Interval::exact(1.0);
        let c = a / 0.0;
        assert_eq!(c.lo, f64::NEG_INFINITY);
        assert_eq!(c.hi, f64::INFINITY);
    }

    #[test]
    fn test_relative_error() {
        let i = Interval::from_midpoint_radius(10.0, 0.01);
        assert!((i.relative_error() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_radius_no_nan_on_midpoint_overflow() {
        // lo + hi overflows to +inf, so the midpoint is inf; the radius
        // must be +inf (the interval is finite but wider than any
        // representable radius), never NaN.
        let lo = f64::MAX * 0.75;
        let hi = f64::MAX * 0.9;
        assert!((lo + hi).is_infinite());
        let i = Interval::new(lo, hi);
        let r = i.radius();
        assert_eq!(r, f64::INFINITY);
        assert!(!r.is_nan());
    }

    #[test]
    fn test_relative_error_zero_midpoint() {
        let i = Interval::from_midpoint_radius(0.0, 0.1);
        assert_eq!(i.relative_error(), f64::INFINITY);
    }

    #[test]
    fn test_recip() {
        let a = Interval::new(2.0, 4.0);
        let r = a.recip();
        assert!(r.lo <= 0.25);
        assert!(r.hi >= 0.5);
    }
}
