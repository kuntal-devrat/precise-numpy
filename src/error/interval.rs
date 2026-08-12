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
    #[inline]
    pub fn from_midpoint_radius(mid: f64, radius: f64) -> Self {
        debug_assert!(radius >= 0.0);
        Self {
            lo: mid - radius,
            hi: mid + radius,
        }
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

    /// Radius (half-width) of the interval.
    #[inline]
    pub fn radius(&self) -> f64 {
        (self.hi - self.lo) * 0.5
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
        Interval { lo: new_lo, hi: new_hi }
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
        // Optimized: batch rounding mode switches instead of 8 separate toggles.
        // We compute all round-down products together, then all round-up products.
        let (lo, hi);

        set_round_down();
        let p1_lo = self.lo * rhs.lo;
        let p2_lo = self.lo * rhs.hi;
        let p3_lo = self.hi * rhs.lo;
        let p4_lo = self.hi * rhs.hi;
        lo = p1_lo.min(p2_lo).min(p3_lo).min(p4_lo);

        set_round_up();
        let p1_hi = self.lo * rhs.lo;
        let p2_hi = self.lo * rhs.hi;
        let p3_hi = self.hi * rhs.lo;
        let p4_hi = self.hi * rhs.hi;
        hi = p1_hi.max(p2_hi).max(p3_hi).max(p4_hi);

        restore_rounding();

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

/// Add with round-down (toward -infinity).
#[inline]
fn add_rd(a: f64, b: f64) -> f64 {
    set_round_down();
    let r = a + b;
    restore_rounding();
    r
}

/// Add with round-up (toward +infinity).
#[inline]
fn add_ru(a: f64, b: f64) -> f64 {
    set_round_up();
    let r = a + b;
    restore_rounding();
    r
}

/// Subtract with round-down.
#[inline]
fn sub_rd(a: f64, b: f64) -> f64 {
    set_round_down();
    let r = a - b;
    restore_rounding();
    r
}

/// Subtract with round-up.
#[inline]
fn sub_ru(a: f64, b: f64) -> f64 {
    set_round_up();
    let r = a - b;
    restore_rounding();
    r
}

/// Multiply with round-down.
#[inline]
fn mul_rd(a: f64, b: f64) -> f64 {
    set_round_down();
    let r = a * b;
    restore_rounding();
    r
}

/// Multiply with round-up.
#[inline]
fn mul_ru(a: f64, b: f64) -> f64 {
    set_round_up();
    let r = a * b;
    restore_rounding();
    r
}

/// Divide with round-down.
#[inline]
fn div_rd(a: f64, b: f64) -> f64 {
    set_round_down();
    let r = a / b;
    restore_rounding();
    r
}

/// Divide with round-up.
#[inline]
fn div_ru(a: f64, b: f64) -> f64 {
    set_round_up();
    let r = a / b;
    restore_rounding();
    r
}

// ── FPU rounding mode control ──────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
fn set_round_down() {
    unsafe {
        let mut mxcsr: u32 = 0;
        std::arch::asm!(
            "stmxcsr [{0}]",
            in(reg) &mut mxcsr,
            options(nostack),
        );
        mxcsr = (mxcsr & !0x6000) | 0x2000;
        std::arch::asm!(
            "ldmxcsr [{0}]",
            in(reg) &mxcsr,
            options(nostack),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn set_round_up() {
    unsafe {
        let mut mxcsr: u32 = 0;
        std::arch::asm!(
            "stmxcsr [{0}]",
            in(reg) &mut mxcsr,
            options(nostack),
        );
        mxcsr = (mxcsr & !0x6000) | 0x4000;
        std::arch::asm!(
            "ldmxcsr [{0}]",
            in(reg) &mxcsr,
            options(nostack),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn restore_rounding() {
    unsafe {
        let mut mxcsr: u32 = 0;
        std::arch::asm!(
            "stmxcsr [{0}]",
            in(reg) &mut mxcsr,
            options(nostack),
        );
        mxcsr = mxcsr & !0x6000;
        std::arch::asm!(
            "ldmxcsr [{0}]",
            in(reg) &mxcsr,
            options(nostack),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn set_round_down() {
    unsafe {
        let mut fpcr: u64 = 0;
        std::arch::asm!("mrs {0}, FPCR", out(reg) fpcr);
        fpcr = (fpcr & !0x0C00_0000) | 0x0400_0000;
        std::arch::asm!("msr FPCR, {0}", in(reg) fpcr);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn set_round_up() {
    unsafe {
        let mut fpcr: u64 = 0;
        std::arch::asm!("mrs {0}, FPCR", out(reg) fpcr);
        fpcr = (fpcr & !0x0C00_0000) | 0x0800_0000;
        std::arch::asm!("msr FPCR, {0}", in(reg) fpcr);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn restore_rounding() {
    unsafe {
        let mut fpcr: u64 = 0;
        std::arch::asm!("mrs {0}, FPCR", out(reg) fpcr);
        fpcr = fpcr & !0x0C00_0000;
        std::arch::asm!("msr FPCR, {0}", in(reg) fpcr);
    }
}

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
        let i = Interval::exact(3.14);
        assert_eq!(i.lo, 3.14);
        assert_eq!(i.hi, 3.14);
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
