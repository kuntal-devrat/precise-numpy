#![allow(dead_code)]

/// Runtime CPU feature detection and SIMD kernel dispatch.
/// Supports AVX-512 (8-wide f64), AVX2/FMA (4-wide f64), ARM NEON, and fallback.

pub mod vec_ops {

    /// Parallel dispatch threshold (elements).
    pub const PAR_THRESHOLD: usize = 32_768;

    // ── Element-wise binary ops ────────────────────────────────────────

    /// Add two slices element-wise into output.
    #[inline]
    pub fn add_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return add_f64_avx512(a, b, out);
                }
            }
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return add_f64_avx2(a, b, out);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                return add_f64_neon(a, b, out);
            }
        }

        for i in 0..n {
            out[i] = a[i] + b[i];
        }
    }

    /// Subtract b from a element-wise.
    #[inline]
    pub fn sub_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return sub_f64_avx512(a, b, out);
                }
            }
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return sub_f64_avx2(a, b, out);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                return sub_f64_neon(a, b, out);
            }
        }

        for i in 0..n {
            out[i] = a[i] - b[i];
        }
    }

    /// Multiply a * b element-wise.
    #[inline]
    pub fn mul_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return mul_f64_avx512(a, b, out);
                }
            }
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return mul_f64_avx2(a, b, out);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                return mul_f64_neon(a, b, out);
            }
        }

        for i in 0..n {
            out[i] = a[i] * b[i];
        }
    }

    /// Divide a / b element-wise.
    #[inline]
    pub fn div_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return div_f64_avx512(a, b, out);
                }
            }
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return div_f64_avx2(a, b, out);
                }
            }
        }

        for i in 0..n {
            out[i] = a[i] / b[i];
        }
    }

    // ── Super-Instruction: Single-Pass Fused Streaming Kernels ─────────

    /// Single-pass streaming kernel for interval addition.
    /// Computes midpoints AND radii in vector registers simultaneously.
    #[inline]
    pub fn add_intervals_stream(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        let n = a_mids.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return add_intervals_avx512(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
                }
            } else if is_x86_feature_detected!("avx2") {
                unsafe {
                    return add_intervals_avx2(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                return add_intervals_neon(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
            }
        }

        for i in 0..n {
            r_mids[i] = a_mids[i] + b_mids[i];
            r_rads[i] = a_rads[i] + b_rads[i];
        }
    }

    /// Single-pass streaming kernel for interval subtraction.
    #[inline]
    pub fn sub_intervals_stream(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        let n = a_mids.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return sub_intervals_avx512(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
                }
            } else if is_x86_feature_detected!("avx2") {
                unsafe {
                    return sub_intervals_avx2(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                return sub_intervals_neon(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
            }
        }

        for i in 0..n {
            r_mids[i] = a_mids[i] - b_mids[i];
            r_rads[i] = a_rads[i] + b_rads[i];
        }
    }

    /// Single-pass streaming kernel for interval multiplication.
    /// Computes midpoints AND radii in vector registers simultaneously.
    /// ZERO temporary vector allocations.
    #[inline]
    pub fn mul_intervals_stream(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        let n = a_mids.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
                unsafe {
                    return mul_intervals_avx512_fma(
                        a_mids, a_rads, b_mids, b_rads, r_mids, r_rads,
                    );
                }
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    return mul_intervals_avx2_fma(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
                }
            } else if is_x86_feature_detected!("avx2") {
                unsafe {
                    return mul_intervals_avx2(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                return mul_intervals_neon(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
            }
        }

        for i in 0..n {
            let am = a_mids[i];
            let ar = a_rads[i];
            let bm = b_mids[i];
            let br = b_rads[i];
            r_mids[i] = am * bm;
            r_rads[i] = am.abs() * br + bm.abs() * ar + ar * br;
        }
    }

    /// Single-pass streaming kernel for scalar interval multiplication.
    #[inline]
    pub fn mul_scalar_stream(
        a_mids: &[f64],
        a_rads: &[f64],
        s_mid: f64,
        s_rad: f64,
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        let n = a_mids.len();
        let abs_s_mid = s_mid.abs();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    return mul_scalar_avx2_fma(
                        a_mids, a_rads, s_mid, s_rad, abs_s_mid, r_mids, r_rads,
                    );
                }
            }
        }

        for i in 0..n {
            r_mids[i] = a_mids[i] * s_mid;
            r_rads[i] = a_mids[i].abs() * s_rad + a_rads[i] * abs_s_mid + a_rads[i] * s_rad;
        }
    }

    // ── Rigorous directed-rounding kernels ──────────────────────────────
    //
    // These kernels compute the midpoint in round-to-nearest and a radius
    // that includes the rounding error of the midpoint operation itself.
    // Each kernel runs a round-to-nearest phase followed by a round-up
    // phase; MXCSR/FPCR state is per-thread, so parallel chunked callers
    // are safe as long as each chunk invokes the kernel on its own thread.

    /// Rigorous interval addition: mid = fl(a_mid + b_mid), radius includes
    /// the exact TwoSum rounding error of the midpoint sum.
    pub fn add_intervals_rigorous(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        let n = a_mids.len();
        for i in 0..n {
            let am = a_mids[i];
            let bm = b_mids[i];
            let s = am + bm;
            let bv = s - am;
            let av = s - bv;
            let br = bm - bv;
            let ar = am - av;
            let err = ar + br;
            r_mids[i] = s;
            r_rads[i] = crate::error::interval::add_ru_chain(
                crate::error::interval::add_ru_chain(a_rads[i], b_rads[i]),
                err.abs(),
            );
        }
    }

    /// Rigorous interval subtraction: mid = fl(a_mid - b_mid), radius
    /// includes the exact TwoSum rounding error of the midpoint difference.
    pub fn sub_intervals_rigorous(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        let n = a_mids.len();
        for i in 0..n {
            let am = a_mids[i];
            let bm = b_mids[i];
            let s = am - bm;
            let bv = s - am;
            let av = s - bv;
            let br = bm - bv;
            let ar = am - av;
            let err = ar + br;
            r_mids[i] = s;
            r_rads[i] = crate::error::interval::add_ru_chain(
                crate::error::interval::add_ru_chain(a_rads[i], b_rads[i]),
                err.abs(),
            );
        }
    }

    /// Rigorous interval multiplication: mid = fl(a_mid * b_mid), radius
    /// includes the exact FMA residual of the midpoint product (0 when the
    /// product is exact), plus the input-radius contributions rounded up.
    pub fn mul_intervals_rigorous(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        let n = a_mids.len();
        for i in 0..n {
            let am = a_mids[i];
            let bm = b_mids[i];
            let s = am * bm;
            let err = if s.is_finite() {
                am.mul_add(bm, -s).abs()
            } else {
                f64::INFINITY
            };
            r_mids[i] = s;
            r_rads[i] = crate::error::interval::add_ru_chain(
                crate::error::interval::add_ru_chain(
                    crate::error::interval::mul_ru(am.abs(), b_rads[i]),
                    crate::error::interval::mul_ru(bm.abs(), a_rads[i]),
                ),
                crate::error::interval::add_ru_chain(
                    crate::error::interval::mul_ru(a_rads[i], b_rads[i]),
                    err,
                ),
            );
        }
    }

    /// Rigorous interval division for a divisor interval that does not
    /// contain zero. Radius = (a_rad + |s|*b_rad + half_ulp(s)*|b_mid|)
    /// divided by (|b_mid| - b_rad), with outward rounding; if the
    /// denominator is non-positive the result radius is +inf.
    pub fn div_intervals_rigorous(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        let n = a_mids.len();
        for i in 0..n {
            r_mids[i] = a_mids[i] / b_mids[i];
        }
        for i in 0..n {
            let bm = b_mids[i];
            let s = r_mids[i];
            let exact_err = if s.is_finite() {
                bm.mul_add(s, -a_mids[i]).abs()
            } else {
                f64::INFINITY
            };
            let nums = crate::error::interval::add_ru_chain(
                crate::error::interval::add_ru_chain(a_rads[i], exact_err),
                crate::error::interval::mul_ru(s.abs(), b_rads[i]),
            );
            let dens = crate::error::interval::sub_rd(bm.abs(), b_rads[i]);
            if dens <= 0.0 {
                r_rads[i] = f64::INFINITY;
            } else {
                r_rads[i] = crate::error::interval::div_ru(nums, dens);
            }
        }
    }

    /// Hardware SIMD vector square root.
    #[inline]
    pub fn sqrt_f64(a: &[f64], out: &mut [f64]) {
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return sqrt_f64_avx512(a, out);
                }
            }
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return sqrt_f64_avx2(a, out);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                return sqrt_f64_neon(a, out);
            }
        }

        for i in 0..n {
            out[i] = a[i].sqrt();
        }
    }

    // ── Fused operations (super-instructions) ──────────────────────────

    /// Fused multiply-add: out[i] = a[i] * b[i] + c[i]
    #[inline]
    pub fn fma_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                unsafe {
                    return fma_f64_avx2(a, b, c, out);
                }
            }
        }

        for i in 0..n {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }

    /// Fused |a| * b: out[i] = |a[i]| * b[i]
    #[inline]
    pub fn abs_mul_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return abs_mul_f64_avx2(a, b, out);
                }
            }
        }

        for i in 0..n {
            out[i] = a[i].abs() * b[i];
        }
    }

    /// Fused triple add: out[i] = a[i] + b[i] + c[i]
    #[inline]
    pub fn add3_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return add3_f64_avx2(a, b, c, out);
                }
            }
        }

        for i in 0..n {
            out[i] = a[i] + b[i] + c[i];
        }
    }

    // ── Unary ops ──────────────────────────────────────────────────────

    /// Compute abs(a) element-wise.
    #[inline]
    pub fn abs_f64(a: &[f64], out: &mut [f64]) {
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return abs_f64_avx2(a, out);
                }
            }
        }

        for i in 0..n {
            out[i] = a[i].abs();
        }
    }

    /// Negate element-wise: out[i] = -a[i]
    #[inline]
    pub fn neg_f64(a: &[f64], out: &mut [f64]) {
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return neg_f64_avx2(a, out);
                }
            }
        }

        for i in 0..n {
            out[i] = -a[i];
        }
    }

    /// Scale by scalar: out[i] = a[i] * scalar
    #[inline]
    pub fn scale_f64(a: &[f64], scalar: f64, out: &mut [f64]) {
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return scale_f64_avx2(a, scalar, out);
                }
            }
        }

        for i in 0..n {
            out[i] = a[i] * scalar;
        }
    }

    /// Add scalar: out[i] = a[i] + scalar
    #[inline]
    pub fn add_scalar_f64(a: &[f64], scalar: f64, out: &mut [f64]) {
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return add_scalar_f64_avx2(a, scalar, out);
                }
            }
        }

        for i in 0..n {
            out[i] = a[i] + scalar;
        }
    }

    // ── Reductions ─────────────────────────────────────────────────────

    /// Compute sum with SIMD accumulation.
    #[inline]
    pub fn sum_f64(a: &[f64]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return sum_f64_avx512(a);
                }
            }
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return sum_f64_avx2(a);
                }
            }
        }

        a.iter().copied().sum()
    }

    /// Compute max element with SIMD.
    #[inline]
    pub fn max_f64(a: &[f64]) -> f64 {
        if a.is_empty() {
            return f64::NEG_INFINITY;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return max_f64_avx2(a);
                }
            }
        }

        a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Compute min element with SIMD.
    #[inline]
    pub fn min_f64(a: &[f64]) -> f64 {
        if a.is_empty() {
            return f64::INFINITY;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return min_f64_avx2(a);
                }
            }
        }

        a.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Compute dot product with FMA: sum(a[i] * b[i])
    #[inline]
    pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    return dot_f64_avx512(a, b);
                }
            }
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                unsafe {
                    return dot_f64_fma(a, b);
                }
            }
        }

        let mut sum = 0.0f64;
        for i in 0..n {
            sum = a[i].mul_add(b[i], sum);
        }
        sum
    }

    // ══════════════════════════════════════════════════════════════════
    // AVX-512 Kernels (8 floats per register, x86_64 only)
    // ══════════════════════════════════════════════════════════════════

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_intervals_avx512(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        let n = a_mids.len();
        let mut i = 0;
        while i + 8 <= n {
            let va_mid = _mm512_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm512_loadu_pd(a_rads.as_ptr().add(i));
            let vb_mid = _mm512_loadu_pd(b_mids.as_ptr().add(i));
            let vb_rad = _mm512_loadu_pd(b_rads.as_ptr().add(i));

            let vr_mid = _mm512_add_pd(va_mid, vb_mid);
            let vr_rad = _mm512_add_pd(va_rad, vb_rad);

            _mm512_storeu_pd(r_mids.as_mut_ptr().add(i), vr_mid);
            _mm512_storeu_pd(r_rads.as_mut_ptr().add(i), vr_rad);
            i += 8;
        }
        while i < n {
            *r_mids.get_unchecked_mut(i) = *a_mids.get_unchecked(i) + *b_mids.get_unchecked(i);
            *r_rads.get_unchecked_mut(i) = *a_rads.get_unchecked(i) + *b_rads.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub_intervals_avx512(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        let n = a_mids.len();
        let mut i = 0;
        while i + 8 <= n {
            let va_mid = _mm512_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm512_loadu_pd(a_rads.as_ptr().add(i));
            let vb_mid = _mm512_loadu_pd(b_mids.as_ptr().add(i));
            let vb_rad = _mm512_loadu_pd(b_rads.as_ptr().add(i));

            let vr_mid = _mm512_sub_pd(va_mid, vb_mid);
            let vr_rad = _mm512_add_pd(va_rad, vb_rad);

            _mm512_storeu_pd(r_mids.as_mut_ptr().add(i), vr_mid);
            _mm512_storeu_pd(r_rads.as_mut_ptr().add(i), vr_rad);
            i += 8;
        }
        while i < n {
            *r_mids.get_unchecked_mut(i) = *a_mids.get_unchecked(i) - *b_mids.get_unchecked(i);
            *r_rads.get_unchecked_mut(i) = *a_rads.get_unchecked(i) + *b_rads.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn sqrt_f64_avx512(a: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 8 <= n {
            let va = _mm512_loadu_pd(a.as_ptr().add(i));
            let vr = _mm512_sqrt_pd(va);
            _mm512_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 8;
        }
        while i < n {
            *out.get_unchecked_mut(i) = a.get_unchecked(i).sqrt();
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_f64_avx512(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 8 <= n {
            let va = _mm512_loadu_pd(a.as_ptr().add(i));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i));
            let vr = _mm512_add_pd(va, vb);
            _mm512_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 8;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub_f64_avx512(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 8 <= n {
            let va = _mm512_loadu_pd(a.as_ptr().add(i));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i));
            let vr = _mm512_sub_pd(va, vb);
            _mm512_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 8;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul_f64_avx512(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 8 <= n {
            let va = _mm512_loadu_pd(a.as_ptr().add(i));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i));
            let vr = _mm512_mul_pd(va, vb);
            _mm512_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 8;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn div_f64_avx512(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 8 <= n {
            let va = _mm512_loadu_pd(a.as_ptr().add(i));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i));
            let vr = _mm512_div_pd(va, vb);
            _mm512_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 8;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) / *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512dq")]
    unsafe fn mul_intervals_avx512_fma(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        let n = a_mids.len();
        let abs_mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFF));
        let mut i = 0;

        while i + 8 <= n {
            let va_mid = _mm512_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm512_loadu_pd(a_rads.as_ptr().add(i));
            let vb_mid = _mm512_loadu_pd(b_mids.as_ptr().add(i));
            let vb_rad = _mm512_loadu_pd(b_rads.as_ptr().add(i));

            let vr_mid = _mm512_mul_pd(va_mid, vb_mid);

            let vabs_a = _mm512_and_pd(va_mid, abs_mask);
            let vabs_b = _mm512_and_pd(vb_mid, abs_mask);

            let vterm1 = _mm512_mul_pd(vabs_a, vb_rad);
            let vterm1_2 = _mm512_fmadd_pd(vabs_b, va_rad, vterm1);
            let vr_rad = _mm512_fmadd_pd(va_rad, vb_rad, vterm1_2);

            _mm512_storeu_pd(r_mids.as_mut_ptr().add(i), vr_mid);
            _mm512_storeu_pd(r_rads.as_mut_ptr().add(i), vr_rad);

            i += 8;
        }

        while i < n {
            let am = *a_mids.get_unchecked(i);
            let ar = *a_rads.get_unchecked(i);
            let bm = *b_mids.get_unchecked(i);
            let br = *b_rads.get_unchecked(i);
            *r_mids.get_unchecked_mut(i) = am * bm;
            *r_rads.get_unchecked_mut(i) = am.abs() * br + bm.abs() * ar + ar * br;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn sum_f64_avx512(a: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm512_setzero_pd();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm512_loadu_pd(a.as_ptr().add(i));
            acc = _mm512_add_pd(acc, v);
            i += 8;
        }
        let mut total = _mm512_reduce_add_pd(acc);
        while i < n {
            total += *a.get_unchecked(i);
            i += 1;
        }
        total
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm512_setzero_pd();
        let mut i = 0;
        while i + 8 <= n {
            let va = _mm512_loadu_pd(a.as_ptr().add(i));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i));
            acc = _mm512_fmadd_pd(va, vb, acc);
            i += 8;
        }
        let mut total = _mm512_reduce_add_pd(acc);
        while i < n {
            total = a.get_unchecked(i).mul_add(*b.get_unchecked(i), total);
            i += 1;
        }
        total
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_intervals_avx2(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        let n = a_mids.len();
        let mut i = 0;
        while i + 4 <= n {
            let va_mid = _mm256_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm256_loadu_pd(a_rads.as_ptr().add(i));
            let vb_mid = _mm256_loadu_pd(b_mids.as_ptr().add(i));
            let vb_rad = _mm256_loadu_pd(b_rads.as_ptr().add(i));

            let vr_mid = _mm256_add_pd(va_mid, vb_mid);
            let vr_rad = _mm256_add_pd(va_rad, vb_rad);

            _mm256_storeu_pd(r_mids.as_mut_ptr().add(i), vr_mid);
            _mm256_storeu_pd(r_rads.as_mut_ptr().add(i), vr_rad);
            i += 4;
        }
        while i < n {
            *r_mids.get_unchecked_mut(i) = *a_mids.get_unchecked(i) + *b_mids.get_unchecked(i);
            *r_rads.get_unchecked_mut(i) = *a_rads.get_unchecked(i) + *b_rads.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_intervals_avx2(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        let n = a_mids.len();
        let mut i = 0;
        while i + 4 <= n {
            let va_mid = _mm256_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm256_loadu_pd(a_rads.as_ptr().add(i));
            let vb_mid = _mm256_loadu_pd(b_mids.as_ptr().add(i));
            let vb_rad = _mm256_loadu_pd(b_rads.as_ptr().add(i));

            let vr_mid = _mm256_sub_pd(va_mid, vb_mid);
            let vr_rad = _mm256_add_pd(va_rad, vb_rad);

            _mm256_storeu_pd(r_mids.as_mut_ptr().add(i), vr_mid);
            _mm256_storeu_pd(r_rads.as_mut_ptr().add(i), vr_rad);
            i += 4;
        }
        while i < n {
            *r_mids.get_unchecked_mut(i) = *a_mids.get_unchecked(i) - *b_mids.get_unchecked(i);
            *r_rads.get_unchecked_mut(i) = *a_rads.get_unchecked(i) + *b_rads.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn mul_scalar_avx2_fma(
        a_mids: &[f64],
        a_rads: &[f64],
        s_mid: f64,
        s_rad: f64,
        abs_s_mid: f64,
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        let n = a_mids.len();
        let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF));

        let vs_mid = _mm256_set1_pd(s_mid);
        let vs_rad = _mm256_set1_pd(s_rad);
        let vabs_s_mid = _mm256_set1_pd(abs_s_mid);

        let mut i = 0;
        while i + 4 <= n {
            let va_mid = _mm256_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm256_loadu_pd(a_rads.as_ptr().add(i));

            let vr_mid = _mm256_mul_pd(va_mid, vs_mid);

            let vabs_a = _mm256_and_pd(va_mid, abs_mask);
            let vt1 = _mm256_mul_pd(vabs_a, vs_rad);
            let vt2 = _mm256_fmadd_pd(va_rad, vabs_s_mid, vt1);
            let vr_rad = _mm256_fmadd_pd(va_rad, vs_rad, vt2);

            _mm256_storeu_pd(r_mids.as_mut_ptr().add(i), vr_mid);
            _mm256_storeu_pd(r_rads.as_mut_ptr().add(i), vr_rad);

            i += 4;
        }
        while i < n {
            let am = *a_mids.get_unchecked(i);
            let ar = *a_rads.get_unchecked(i);
            *r_mids.get_unchecked_mut(i) = am * s_mid;
            *r_rads.get_unchecked_mut(i) = am.abs() * s_rad + ar * abs_s_mid + ar * s_rad;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sqrt_f64_avx2(a: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vr = _mm256_sqrt_pd(va);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = a.get_unchecked(i).sqrt();
            i += 1;
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // AVX2 Single-Pass Interval Multiplication Kernel
    // ══════════════════════════════════════════════════════════════════

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn mul_intervals_avx2_fma(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        let n = a_mids.len();
        let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF));
        let mut i = 0;

        while i + 4 <= n {
            let va_mid = _mm256_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm256_loadu_pd(a_rads.as_ptr().add(i));
            let vb_mid = _mm256_loadu_pd(b_mids.as_ptr().add(i));
            let vb_rad = _mm256_loadu_pd(b_rads.as_ptr().add(i));

            let vr_mid = _mm256_mul_pd(va_mid, vb_mid);

            let vabs_a = _mm256_and_pd(va_mid, abs_mask);
            let vabs_b = _mm256_and_pd(vb_mid, abs_mask);

            let vterm1 = _mm256_mul_pd(vabs_a, vb_rad);
            let vterm1_2 = _mm256_fmadd_pd(vabs_b, va_rad, vterm1);
            let vr_rad = _mm256_fmadd_pd(va_rad, vb_rad, vterm1_2);

            _mm256_storeu_pd(r_mids.as_mut_ptr().add(i), vr_mid);
            _mm256_storeu_pd(r_rads.as_mut_ptr().add(i), vr_rad);

            i += 4;
        }

        while i < n {
            let am = *a_mids.get_unchecked(i);
            let ar = *a_rads.get_unchecked(i);
            let bm = *b_mids.get_unchecked(i);
            let br = *b_rads.get_unchecked(i);
            *r_mids.get_unchecked_mut(i) = am * bm;
            *r_rads.get_unchecked_mut(i) = am.abs() * br + bm.abs() * ar + ar * br;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_intervals_avx2(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::x86_64::*;
        let n = a_mids.len();
        let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF));
        let mut i = 0;

        while i + 4 <= n {
            let va_mid = _mm256_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm256_loadu_pd(a_rads.as_ptr().add(i));
            let vb_mid = _mm256_loadu_pd(b_mids.as_ptr().add(i));
            let vb_rad = _mm256_loadu_pd(b_rads.as_ptr().add(i));

            let vr_mid = _mm256_mul_pd(va_mid, vb_mid);

            let vabs_a = _mm256_and_pd(va_mid, abs_mask);
            let vabs_b = _mm256_and_pd(vb_mid, abs_mask);

            let vt1 = _mm256_mul_pd(vabs_a, vb_rad);
            let vt2 = _mm256_mul_pd(vabs_b, va_rad);
            let vt3 = _mm256_mul_pd(va_rad, vb_rad);

            let vr_rad = _mm256_add_pd(_mm256_add_pd(vt1, vt2), vt3);

            _mm256_storeu_pd(r_mids.as_mut_ptr().add(i), vr_mid);
            _mm256_storeu_pd(r_rads.as_mut_ptr().add(i), vr_rad);

            i += 4;
        }

        while i < n {
            let am = *a_mids.get_unchecked(i);
            let ar = *a_rads.get_unchecked(i);
            let bm = *b_mids.get_unchecked(i);
            let br = *b_rads.get_unchecked(i);
            *r_mids.get_unchecked_mut(i) = am * bm;
            *r_rads.get_unchecked_mut(i) = am.abs() * br + bm.abs() * ar + ar * br;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_f64_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vr = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_f64_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vr = _mm256_sub_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_f64_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vr = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn div_f64_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vr = _mm256_div_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) / *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fma_f64_avx2(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vc = _mm256_loadu_pd(c.as_ptr().add(i));
            let vr = _mm256_fmadd_pd(va, vb, vc);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = a
                .get_unchecked(i)
                .mul_add(*b.get_unchecked(i), *c.get_unchecked(i));
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn abs_mul_f64_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF));
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let va_abs = _mm256_and_pd(va, abs_mask);
            let vr = _mm256_mul_pd(va_abs, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = a.get_unchecked(i).abs() * *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add3_f64_avx2(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vc = _mm256_loadu_pd(c.as_ptr().add(i));
            let vr = _mm256_add_pd(_mm256_add_pd(va, vb), vc);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) =
                *a.get_unchecked(i) + *b.get_unchecked(i) + *c.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn abs_f64_avx2(a: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF));
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vr = _mm256_and_pd(va, abs_mask);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = a.get_unchecked(i).abs();
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn neg_f64_avx2(a: &[f64], out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(i64::MIN));
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vr = _mm256_xor_pd(va, sign_mask);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = -*a.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_f64_avx2(a: &[f64], scalar: f64, out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let vs = _mm256_set1_pd(scalar);
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vr = _mm256_mul_pd(va, vs);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) * scalar;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_scalar_f64_avx2(a: &[f64], scalar: f64, out: &mut [f64]) {
        use std::arch::x86_64::*;
        let n = a.len();
        let vs = _mm256_set1_pd(scalar);
        let mut i = 0;
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vr = _mm256_add_pd(va, vs);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
            i += 4;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) + scalar;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_f64_avx2(a: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm256_setzero_pd();
        let mut i = 0;
        while i + 8 <= n {
            let v0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let v1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
            acc = _mm256_add_pd(acc, _mm256_add_pd(v0, v1));
            i += 8;
        }
        while i + 4 <= n {
            let v = _mm256_loadu_pd(a.as_ptr().add(i));
            acc = _mm256_add_pd(acc, v);
            i += 4;
        }
        let hi128 = _mm256_extractf128_pd(acc, 1);
        let lo128 = _mm256_castpd256_pd128(acc);
        let sum128 = _mm_add_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(sum128, sum128);
        let mut total = _mm_cvtsd_f64(_mm_add_sd(sum128, hi64));
        while i < n {
            total += *a.get_unchecked(i);
            i += 1;
        }
        total
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn max_f64_avx2(a: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm256_set1_pd(f64::NEG_INFINITY);
        let mut i = 0;
        while i + 4 <= n {
            let v = _mm256_loadu_pd(a.as_ptr().add(i));
            acc = _mm256_max_pd(acc, v);
            i += 4;
        }
        let hi128 = _mm256_extractf128_pd(acc, 1);
        let lo128 = _mm256_castpd256_pd128(acc);
        let m128 = _mm_max_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(m128, m128);
        let max_sd = _mm_max_sd(m128, hi64);
        let mut result = _mm_cvtsd_f64(max_sd);
        while i < n {
            let v = *a.get_unchecked(i);
            if v > result {
                result = v;
            }
            i += 1;
        }
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn min_f64_avx2(a: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc = _mm256_set1_pd(f64::INFINITY);
        let mut i = 0;
        while i + 4 <= n {
            let v = _mm256_loadu_pd(a.as_ptr().add(i));
            acc = _mm256_min_pd(acc, v);
            i += 4;
        }
        let hi128 = _mm256_extractf128_pd(acc, 1);
        let lo128 = _mm256_castpd256_pd128(acc);
        let m128 = _mm_min_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(m128, m128);
        let min_sd = _mm_min_sd(m128, hi64);
        let mut result = _mm_cvtsd_f64(min_sd);
        while i < n {
            let v = *a.get_unchecked(i);
            if v < result {
                result = v;
            }
            i += 1;
        }
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_f64_fma(a: &[f64], b: &[f64]) -> f64 {
        use std::arch::x86_64::*;
        let n = a.len();
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 8 <= n {
            let va0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_pd(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_pd(va0, vb0, acc0);
            let va1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
            let vb1 = _mm256_loadu_pd(b.as_ptr().add(i + 4));
            acc1 = _mm256_fmadd_pd(va1, vb1, acc1);
            i += 8;
        }
        let mut acc = _mm256_add_pd(acc0, acc1);
        while i + 4 <= n {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            acc = _mm256_fmadd_pd(va, vb, acc);
            i += 4;
        }
        let hi128 = _mm256_extractf128_pd(acc, 1);
        let lo128 = _mm256_castpd256_pd128(acc);
        let sum128 = _mm_add_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(sum128, sum128);
        let mut total = _mm_cvtsd_f64(_mm_add_sd(sum128, hi64));
        while i < n {
            total = a.get_unchecked(i).mul_add(*b.get_unchecked(i), total);
            i += 1;
        }
        total
    }

    // ══════════════════════════════════════════════════════════════════
    // ARM NEON Kernels (aarch64 only)
    // ══════════════════════════════════════════════════════════════════

    #[cfg(target_arch = "aarch64")]
    unsafe fn add_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::aarch64::*;
        let n = a.len();
        let mut i = 0;
        while i + 2 <= n {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vr = vaddq_f64(va, vb);
            vst1q_f64(out.as_mut_ptr().add(i), vr);
            i += 2;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn sub_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::aarch64::*;
        let n = a.len();
        let mut i = 0;
        while i + 2 <= n {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vr = vsubq_f64(va, vb);
            vst1q_f64(out.as_mut_ptr().add(i), vr);
            i += 2;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn mul_f64_neon(a: &[f64], b: &[f64], out: &mut [f64]) {
        use std::arch::aarch64::*;
        let n = a.len();
        let mut i = 0;
        while i + 2 <= n {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vr = vmulq_f64(va, vb);
            vst1q_f64(out.as_mut_ptr().add(i), vr);
            i += 2;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn add_intervals_neon(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::aarch64::*;
        let n = a_mids.len();
        let mut i = 0;
        while i + 2 <= n {
            let va_mid = vld1q_f64(a_mids.as_ptr().add(i));
            let va_rad = vld1q_f64(a_rads.as_ptr().add(i));
            let vb_mid = vld1q_f64(b_mids.as_ptr().add(i));
            let vb_rad = vld1q_f64(b_rads.as_ptr().add(i));

            let vr_mid = vaddq_f64(va_mid, vb_mid);
            let vr_rad = vaddq_f64(va_rad, vb_rad);

            vst1q_f64(r_mids.as_mut_ptr().add(i), vr_mid);
            vst1q_f64(r_rads.as_mut_ptr().add(i), vr_rad);
            i += 2;
        }
        while i < n {
            *r_mids.get_unchecked_mut(i) = *a_mids.get_unchecked(i) + *b_mids.get_unchecked(i);
            *r_rads.get_unchecked_mut(i) = *a_rads.get_unchecked(i) + *b_rads.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn sub_intervals_neon(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::aarch64::*;
        let n = a_mids.len();
        let mut i = 0;
        while i + 2 <= n {
            let va_mid = vld1q_f64(a_mids.as_ptr().add(i));
            let va_rad = vld1q_f64(a_rads.as_ptr().add(i));
            let vb_mid = vld1q_f64(b_mids.as_ptr().add(i));
            let vb_rad = vld1q_f64(b_rads.as_ptr().add(i));

            let vr_mid = vsubq_f64(va_mid, vb_mid);
            let vr_rad = vaddq_f64(va_rad, vb_rad);

            vst1q_f64(r_mids.as_mut_ptr().add(i), vr_mid);
            vst1q_f64(r_rads.as_mut_ptr().add(i), vr_rad);
            i += 2;
        }
        while i < n {
            *r_mids.get_unchecked_mut(i) = *a_mids.get_unchecked(i) - *b_mids.get_unchecked(i);
            *r_rads.get_unchecked_mut(i) = *a_rads.get_unchecked(i) + *b_rads.get_unchecked(i);
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn sqrt_f64_neon(a: &[f64], out: &mut [f64]) {
        use std::arch::aarch64::*;
        let n = a.len();
        let mut i = 0;
        while i + 2 <= n {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vr = vsqrtq_f64(va);
            vst1q_f64(out.as_mut_ptr().add(i), vr);
            i += 2;
        }
        while i < n {
            *out.get_unchecked_mut(i) = a.get_unchecked(i).sqrt();
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn mul_intervals_neon(
        a_mids: &[f64],
        a_rads: &[f64],
        b_mids: &[f64],
        b_rads: &[f64],
        r_mids: &mut [f64],
        r_rads: &mut [f64],
    ) {
        use std::arch::aarch64::*;
        let n = a_mids.len();
        let mut i = 0;
        while i + 2 <= n {
            let va_mid = vld1q_f64(a_mids.as_ptr().add(i));
            let va_rad = vld1q_f64(a_rads.as_ptr().add(i));
            let vb_mid = vld1q_f64(b_mids.as_ptr().add(i));
            let vb_rad = vld1q_f64(b_rads.as_ptr().add(i));

            let vr_mid = vmulq_f64(va_mid, vb_mid);
            let vabs_a = vabsq_f64(va_mid);
            let vabs_b = vabsq_f64(vb_mid);

            let vt1 = vmulq_f64(vabs_a, vb_rad);
            let vt2 = vfmaq_f64(vt1, vabs_b, va_rad);
            let vr_rad = vfmaq_f64(vt2, va_rad, vb_rad);

            vst1q_f64(r_mids.as_mut_ptr().add(i), vr_mid);
            vst1q_f64(r_rads.as_mut_ptr().add(i), vr_rad);
            i += 2;
        }
        while i < n {
            let am = *a_mids.get_unchecked(i);
            let ar = *a_rads.get_unchecked(i);
            let bm = *b_mids.get_unchecked(i);
            let br = *b_rads.get_unchecked(i);
            *r_mids.get_unchecked_mut(i) = am * bm;
            *r_rads.get_unchecked_mut(i) = am.abs() * br + bm.abs() * ar + ar * br;
            i += 1;
        }
    }
}
