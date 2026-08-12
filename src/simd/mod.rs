/// Runtime CPU feature detection and SIMD kernel dispatch.
/// Provides fused super-instructions that eliminate intermediate allocations.

/// Generic SIMD-capable vector type for f64 interval operations.
pub mod vec_ops {

    // ── Parallel dispatch threshold ────────────────────────────────────
    /// Arrays smaller than this use sequential SIMD; larger use Rayon.
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
            if is_x86_feature_detected!("avx2") {
                unsafe { return add_f64_avx2(a, b, out); }
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
            if is_x86_feature_detected!("avx2") {
                unsafe { return sub_f64_avx2(a, b, out); }
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
            if is_x86_feature_detected!("avx2") {
                unsafe { return mul_f64_avx2(a, b, out); }
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
            if is_x86_feature_detected!("avx2") {
                unsafe { return div_f64_avx2(a, b, out); }
            }
        }

        for i in 0..n {
            out[i] = a[i] / b[i];
        }
    }

    // ── Super-Instruction: Fused Streaming Interval Multiplication ─────
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
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    return mul_intervals_avx2_fma(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
                }
            } else if is_x86_feature_detected!("avx2") {
                unsafe {
                    return mul_intervals_avx2(a_mids, a_rads, b_mids, b_rads, r_mids, r_rads);
                }
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

    // ── Fused operations (super-instructions) ──────────────────────────

    /// Fused multiply-add: out[i] = a[i] * b[i] + c[i]
    #[inline]
    pub fn fma_f64(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
        let n = a.len();
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(c.len(), n);
        debug_assert_eq!(out.len(), n);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                unsafe { return fma_f64_avx2(a, b, c, out); }
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
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(out.len(), n);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { return abs_mul_f64_avx2(a, b, out); }
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
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(c.len(), n);
        debug_assert_eq!(out.len(), n);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { return add3_f64_avx2(a, b, c, out); }
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
        debug_assert_eq!(out.len(), n);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { return abs_f64_avx2(a, out); }
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
        debug_assert_eq!(out.len(), n);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { return neg_f64_avx2(a, out); }
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
        debug_assert_eq!(out.len(), n);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { return scale_f64_avx2(a, scalar, out); }
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
        debug_assert_eq!(out.len(), n);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { return add_scalar_f64_avx2(a, scalar, out); }
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
            if is_x86_feature_detected!("avx2") {
                unsafe { return sum_f64_avx2(a); }
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
                unsafe { return max_f64_avx2(a); }
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
                unsafe { return min_f64_avx2(a); }
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
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                unsafe { return dot_f64_fma(a, b); }
            }
        }

        let mut sum = 0.0f64;
        for i in 0..n {
            sum = a[i].mul_add(b[i], sum);
        }
        sum
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

        // Process 4 floats (256-bit) per iteration
        while i + 4 <= n {
            let va_mid = _mm256_loadu_pd(a_mids.as_ptr().add(i));
            let va_rad = _mm256_loadu_pd(a_rads.as_ptr().add(i));
            let vb_mid = _mm256_loadu_pd(b_mids.as_ptr().add(i));
            let vb_rad = _mm256_loadu_pd(b_rads.as_ptr().add(i));

            // r_mid = a_mid * b_mid
            let vr_mid = _mm256_mul_pd(va_mid, vb_mid);

            // abs_a_mid = |a_mid|, abs_b_mid = |b_mid|
            let vabs_a = _mm256_and_pd(va_mid, abs_mask);
            let vabs_b = _mm256_and_pd(vb_mid, abs_mask);

            // term1 = |a_mid| * b_rad
            // term2 = |b_mid| * a_rad + term1 (using FMA: vabs_b * va_rad + term1)
            let vterm1 = _mm256_mul_pd(vabs_a, vb_rad);
            let vterm1_2 = _mm256_fmadd_pd(vabs_b, va_rad, vterm1);

            // r_rad = a_rad * b_rad + term1_2 (using FMA)
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

    // ══════════════════════════════════════════════════════════════════
    // Standard AVX2 kernels
    // ══════════════════════════════════════════════════════════════════

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
            *out.get_unchecked_mut(i) = a.get_unchecked(i).mul_add(*b.get_unchecked(i), *c.get_unchecked(i));
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
            *out.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i) + *c.get_unchecked(i);
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
            if v > result { result = v; }
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
            if v < result { result = v; }
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
}
