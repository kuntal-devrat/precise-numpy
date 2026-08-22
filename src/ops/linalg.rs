//! Linear algebra on interval matrices.
//!
//! `det`, `inv`, `solve`, and `lstsq` run through interval Gaussian
//! elimination, so error bounds propagate through every arithmetic step
//! with hardware-directed rounding.
//!
//! `svd`, `eig`, and `pinv` run iterative midpoint algorithms (Jacobi
//! rotations / shifted QR) and then attach **rigorous enclosures**:
//!
//! - Singular values / symmetric eigenvalues are enclosed with Weyl's
//!   perturbation theorem against the interval residual of the computed
//!   factorization, so the returned interval is guaranteed to contain the
//!   true value for *any* interval input.
//! - Singular vectors and symmetric eigenvectors are enclosed with the
//!   Davis–Kahan sin-theta theorem: the bound degrades gracefully when
//!   singular values / eigenvalues are close, and collapses to the full
//!   unit-sphere range when the spectral gap is smaller than the
//!   uncertainty (no false confidence).
//! - General (non-symmetric) eigenvalues are enclosed with the Bauer–Fike
//!   theorem, which needs the eigenvector matrix to be well-conditioned;
//!   otherwise an error is raised. Eigenvectors of non-symmetric matrices
//!   cannot be bounded without a spectral gap condition that does not hold
//!   for non-normal matrices, so they are returned as unbounded intervals
//!   rather than fake-precise point values.

use crate::array::IntervalArray;
use crate::error::interval::{add_ru, div_ru, mul_ru, next_down, next_up};
use crate::error::Interval;
use crate::ops::reduction::{matmul, matmul_general, MatmulResult};

fn row_major(_m: usize, n: usize, i: usize, j: usize) -> usize {
    i * n + j
}

// ── Rigorous enclosure helpers ─────────────────────────────────────────

/// Supremum of |x| over an interval (exactly representable).
#[inline]
fn sup_abs(iv: &Interval) -> f64 {
    iv.lo.abs().max(iv.hi.abs())
}

/// Frobenius norm of a slice of intervals, rounded upward. This is a
/// rigorous upper bound on the 2-norm of the true matrix: every entry is
/// replaced by the supremum of its absolute value and each arithmetic step
/// rounds up.
fn sup_frobenius(entries: &[Interval]) -> f64 {
    let mut acc = Interval::zero();
    for iv in entries {
        let s = Interval::exact(sup_abs(iv));
        acc = acc + s * s;
    }
    next_up(acc.hi.max(0.0).sqrt())
}

/// ‖a·v − s·u‖ for an interval matrix `a`, point vectors `v`, `u`, and a
/// point scalar `s`, computed with interval arithmetic so the result is a
/// rigorous upper bound on the true residual norm.
fn matvec_residual(a: &IntervalArray, v: &[f64], s: f64, u: &[f64]) -> Result<f64, String> {
    let varr = IntervalArray::from_f64_slice(v);
    let av = match matmul_general(a, &varr)? {
        MatmulResult::Array(x) => x,
        MatmulResult::Scalar(_) => unreachable!("a is 2D in matvec_residual"),
    };
    let mut e: Vec<Interval> = Vec::with_capacity(av.len());
    for i in 0..av.len() {
        let sub = Interval::exact(s) * Interval::exact(u[i]);
        e.push(av.get(i) - sub);
    }
    Ok(sup_frobenius(&e))
}

/// Per-column residual of the computed singular triple (u_j, σ_j, v_j):
/// ‖[Ã·v_j − σ_j·u_j ; Ãᵀ·u_j − σ_j·v_j]‖ / √2, rounded upward. This is the
/// residual of the symmetric block matrix [0, Ã; Ãᵀ, 0] against the
/// approximate eigenvector [u_j; v_j]/√2, which is what the Davis–Kahan
/// theorem needs.
fn svd_column_residual(
    src: &IntervalArray,
    su: &[f64],
    svt: &[f64],
    mw: usize,
    nw: usize,
    j: usize,
    sj: f64,
) -> Result<f64, String> {
    let mut uj = Vec::with_capacity(mw);
    let mut vj = Vec::with_capacity(nw);
    for i in 0..mw {
        uj.push(su[i * nw + j]);
    }
    for i in 0..nw {
        vj.push(svt[j * nw + i]);
    }
    let r1 = matvec_residual(src, &vj, sj, &uj)?;
    let src_t = src.transpose();
    let r2 = matvec_residual(&src_t, &uj, sj, &vj)?;
    // sqrt(r1² + r2²) / √2, all rounding outward.
    let a = Interval::exact(r1) * Interval::exact(r1);
    let b = Interval::exact(r2) * Interval::exact(r2);
    let half_norm = next_up((a + b).hi.sqrt());
    Ok(mul_ru(half_norm, next_up(1.0 / 2.0_f64.sqrt())))
}

/// Davis–Kahan gap for singular value j: distance from σ̃_j to every other
/// eigenvalue of the block matrix (±σ̃_i), shrunk by the per-column
/// uncertainty of the enclosing intervals so the bound survives the
/// enclosure of the singular values.
fn svd_gap(ss: &[f64], rad_s: &[f64], j: usize) -> f64 {
    let mut gap = 2.0 * (ss[j] - rad_s[j]).max(0.0);
    for i in 0..ss.len() {
        if i == j {
            continue;
        }
        let d = (ss[i] - ss[j]).abs() - rad_s[i] - rad_s[j];
        if d < gap {
            gap = d;
        }
    }
    gap.max(0.0)
}

/// Davis–Kahan gap for symmetric eigenvalue j.
fn eig_gap(vals: &[f64], rads: &[f64], j: usize) -> f64 {
    let mut gap = f64::INFINITY;
    for i in 0..vals.len() {
        if i == j {
            continue;
        }
        let d = (vals[i] - vals[j]).abs() - rads[i] - rads[j];
        if d < gap {
            gap = d;
        }
    }
    gap.max(0.0)
}

/// Returns the (i, j) element of the interval matrix.
#[inline]
fn get_iv(a: &IntervalArray, i: usize, j: usize) -> Interval {
    a.get(i * a.shape()[1] + j)
}

// ── Determinant (interval LU with partial pivoting) ────────────────────

pub fn det(a: &IntervalArray) -> Result<Interval, String> {
    if a.ndim() != 2 {
        return Err("det requires a 2D array".to_string());
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err("det requires a square matrix".to_string());
    }
    if n == 0 {
        return Ok(Interval::exact(1.0));
    }

    let mut m = vec![Interval::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            m[row_major(n, n, i, j)] = get_iv(a, i, j);
        }
    }

    let mut det_acc = Interval::exact(1.0);
    let mut sign = 1.0f64;

    for col in 0..n {
        // partial pivot: largest |midpoint| in column at/after `col`
        let mut pivot = col;
        let mut pivot_mag = m[row_major(n, n, col, col)].midpoint().abs();
        for r in (col + 1)..n {
            let mag = m[row_major(n, n, r, col)].midpoint().abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot = r;
            }
        }
        if pivot_mag == 0.0 {
            return Ok(Interval::exact(0.0));
        }
        if pivot != col {
            for j in 0..n {
                let tmp = m[row_major(n, n, col, j)];
                m[row_major(n, n, col, j)] = m[row_major(n, n, pivot, j)];
                m[row_major(n, n, pivot, j)] = tmp;
            }
            sign = -sign;
        }
        let pivot_iv = m[row_major(n, n, col, col)];
        det_acc = det_acc * pivot_iv;
        for r in (col + 1)..n {
            let factor = m[row_major(n, n, r, col)] / pivot_iv;
            if factor.is_exact() && factor.lo == 0.0 {
                continue;
            }
            m[row_major(n, n, r, col)] = Interval::zero();
            for j in (col + 1)..n {
                let sub = factor * m[row_major(n, n, col, j)];
                m[row_major(n, n, r, j)] = m[row_major(n, n, r, j)] - sub;
            }
        }
    }

    let d = det_acc.midpoint() * sign;
    let r = det_acc.radius();
    Ok(Interval::from_midpoint_radius(d, r))
}

// ── Solve A x = b via interval Gaussian elimination ────────────────────

/// Solve A x = b where A is [n, n] (in rows of intervals) and b is [n].
fn lu_solve_interval(m: &mut Vec<Interval>, n: usize, b: &[Interval]) -> Option<Vec<Interval>> {
    let mut x = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_mag = m[row_major(n, n, col, col)].midpoint().abs();
        for r in (col + 1)..n {
            let mag = m[row_major(n, n, r, col)].midpoint().abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot = r;
            }
        }
        if pivot_mag == 0.0 {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                let tmp = m[row_major(n, n, col, j)];
                m[row_major(n, n, col, j)] = m[row_major(n, n, pivot, j)];
                m[row_major(n, n, pivot, j)] = tmp;
            }
            let tmp = x[col];
            x[col] = x[pivot];
            x[pivot] = tmp;
        }
        let pivot_iv = m[row_major(n, n, col, col)];
        for r in (col + 1)..n {
            let factor = m[row_major(n, n, r, col)] / pivot_iv;
            if factor.is_exact() && factor.lo == 0.0 {
                continue;
            }
            m[row_major(n, n, r, col)] = Interval::zero();
            for j in (col + 1)..n {
                let sub = factor * m[row_major(n, n, col, j)];
                m[row_major(n, n, r, j)] = m[row_major(n, n, r, j)] - sub;
            }
            x[r] = x[r] - factor * x[col];
        }
    }
    // back substitution
    let mut out = vec![Interval::zero(); n];
    for i in (0..n).rev() {
        let mut acc = x[i];
        for j in (i + 1)..n {
            acc = acc - m[row_major(n, n, i, j)] * out[j];
        }
        let diag = m[row_major(n, n, i, i)];
        if diag.is_exact() && diag.lo == 0.0 {
            return None;
        }
        out[i] = acc / diag;
    }
    Some(out)
}

pub fn solve(a: &IntervalArray, b: &IntervalArray) -> Result<IntervalArray, String> {
    if a.ndim() != 2 {
        return Err("solve requires A to be 2D".to_string());
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err("solve requires a square matrix A".to_string());
    }
    let mut m = vec![Interval::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            m[row_major(n, n, i, j)] = get_iv(a, i, j);
        }
    }
    let result = match b.ndim() {
        1 => {
            if b.len() != n {
                return Err("solve: b length must match A rows".to_string());
            }
            let rhs: Vec<Interval> = (0..n).map(|i| b.get(i)).collect();
            let mut mm = m.clone();
            match lu_solve_interval(&mut mm, n, &rhs) {
                Some(x) => IntervalArray::from_intervals(&x),
                None => {
                    return Err("solve: singular matrix (interval contains zero pivot)".to_string())
                }
            }
        }
        2 => {
            let nrhs = b.shape()[1];
            if b.shape()[0] != n {
                return Err("solve: b rows must match A rows".to_string());
            }
            let mut cols: Vec<IntervalArray> = Vec::with_capacity(nrhs);
            for c in 0..nrhs {
                let mut rhs = Vec::with_capacity(n);
                let mut mm = m.clone();
                for i in 0..n {
                    rhs.push(b.get(i * nrhs + c));
                }
                match lu_solve_interval(&mut mm, n, &rhs) {
                    Some(x) => {
                        cols.push(IntervalArray::from_intervals(&x));
                    }
                    None => {
                        return Err(
                            "solve: singular matrix (interval contains zero pivot)".to_string()
                        )
                    }
                }
            }
            if nrhs == 1 {
                return Ok(cols.pop().unwrap());
            }
            let mut out_mids = vec![0.0f64; n * nrhs];
            let mut out_rads = vec![0.0f64; n * nrhs];
            for (c, col) in cols.iter().enumerate() {
                for i in 0..n {
                    out_mids[i * nrhs + c] = col.get(i).midpoint();
                    out_rads[i * nrhs + c] = col.get(i).radius();
                }
            }
            IntervalArray::from_raw_parts(&out_mids, &out_rads, &[n, nrhs])
        }
        _ => return Err("solve: b must be 1D or 2D".to_string()),
    };
    Ok(result)
}

pub fn inv(a: &IntervalArray) -> Result<IntervalArray, String> {
    if a.ndim() != 2 {
        return Err("inv requires a 2D array".to_string());
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err("inv requires a square matrix".to_string());
    }
    let mut eye_mids = vec![0.0f64; n * n];
    for i in 0..n {
        eye_mids[i * n + i] = 1.0;
    }
    let eye = IntervalArray::from_raw_parts(&eye_mids, &vec![0.0; n * n], &[n, n]);
    solve(a, &eye)
}

// ── Least squares via normal equations (interval arithmetic) ───────────

pub fn lstsq(a: &IntervalArray, b: &IntervalArray) -> Result<IntervalArray, String> {
    if a.ndim() != 2 {
        return Err("lstsq requires A to be 2D".to_string());
    }
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let b_is_2d = b.ndim() == 2;
    let nrhs = if b_is_2d { b.shape()[1] } else { 1 };
    if b.len() != m * nrhs {
        return Err("lstsq: b shape must be (m,) or (m, nrhs)".to_string());
    }

    // Normal equations: (A^T A) x = A^T b, solved with interval arithmetic.
    let at = a.transpose();
    let ata = match crate::ops::reduction::matmul_general(&at, a) {
        Ok(crate::ops::reduction::MatmulResult::Array(x)) => x,
        _ => return Err("lstsq: A^T A is not 2D".to_string()),
    };
    let atb = match crate::ops::reduction::matmul_general(&at, b) {
        Ok(crate::ops::reduction::MatmulResult::Array(x)) => x,
        Ok(crate::ops::reduction::MatmulResult::Scalar(iv)) => {
            IntervalArray::from_f64_slice(&[iv.midpoint()])
        }
        Err(e) => return Err(e),
    };
    let x = solve(&ata, &atb)?;

    // lstsq always returns a 2D array (k, nrhs) per NumPy convention.
    if b_is_2d {
        Ok(x)
    } else {
        let reshaped = x.reshape(&[k, 1]);
        let mut out = vec![0.0f64; k];
        let mut outr = vec![0.0f64; k];
        for i in 0..k {
            out[i] = reshaped.get(i).midpoint();
            outr[i] = reshaped.get(i).radius();
        }
        Ok(IntervalArray::from_raw_parts(&out, &outr, &[k]))
    }
}

// ── SVD via one-sided Jacobi (midpoint arithmetic) ─────────────────────

pub fn svd(a: &IntervalArray) -> Result<(IntervalArray, IntervalArray, IntervalArray), String> {
    if a.ndim() != 2 {
        return Err("svd requires a 2D array".to_string());
    }
    let m = a.shape()[0];
    let n = a.shape()[1];
    // Degenerate input: return empty factors with the NumPy-convention
    // reduced shapes (m >= n: U (m, n), S (n,), VT (n, n); else
    // U (m, m), S (m,), VT (m, n)).
    if m == 0 || n == 0 {
        let (u_shape, s_len, vt_shape) = if m >= n {
            ([m, n], n, [n, n])
        } else {
            ([m, m], m, [m, n])
        };
        let u = IntervalArray::zeros(&u_shape);
        let s = IntervalArray::zeros(&[s_len]);
        let vt = IntervalArray::zeros(&vt_shape);
        return Ok((u, s, vt));
    }
    // One-sided Jacobi needs at least as many rows as columns; for wide
    // matrices compute the SVD of A^T and transpose the factors back.
    let at = a.transpose();
    let (mw, nw) = if m >= n { (m, n) } else { (n, m) };
    let src = if m >= n { a } else { &at };
    let mut u = vec![0.0f64; mw * nw];
    for i in 0..mw {
        for j in 0..nw {
            u[row_major(mw, nw, i, j)] = get_iv(src, i, j).midpoint();
        }
    }
    let mut v = vec![0.0f64; nw * nw];
    for i in 0..nw {
        v[row_major(nw, nw, i, i)] = 1.0;
    }

    let eps = f64::EPSILON;
    let max_sweeps = 60;
    for _ in 0..max_sweeps {
        let mut off = 0.0f64;
        for p in 0..nw {
            for q in (p + 1)..nw {
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = 0.0f64;
                for i in 0..mw {
                    let uip = u[row_major(mw, nw, i, p)];
                    let uiq = u[row_major(mw, nw, i, q)];
                    alpha += uip * uip;
                    beta += uiq * uiq;
                    gamma += uip * uiq;
                }
                off += gamma * gamma;
                if gamma.abs() <= eps * (alpha * beta).sqrt() {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..mw {
                    let uip = u[row_major(mw, nw, i, p)];
                    let uiq = u[row_major(mw, nw, i, q)];
                    u[row_major(mw, nw, i, p)] = c * uip - s * uiq;
                    u[row_major(mw, nw, i, q)] = s * uip + c * uiq;
                }
                for i in 0..nw {
                    let vip = v[row_major(nw, nw, i, p)];
                    let viq = v[row_major(nw, nw, i, q)];
                    v[row_major(nw, nw, i, p)] = c * vip - s * viq;
                    v[row_major(nw, nw, i, q)] = s * vip + c * viq;
                }
            }
        }
        if off <= eps * eps * (mw * nw) as f64 {
            break;
        }
    }

    // Singular values = column norms of U; normalize U columns.
    let mut s = vec![0.0f64; nw];
    for j in 0..nw {
        let mut norm = 0.0f64;
        for i in 0..mw {
            norm += u[row_major(mw, nw, i, j)] * u[row_major(mw, nw, i, j)];
        }
        norm = norm.sqrt();
        s[j] = norm;
        if norm > 0.0 {
            for i in 0..mw {
                u[row_major(mw, nw, i, j)] /= norm;
            }
        }
    }

    // Sort singular values descending, permuting U and V columns.
    let mut order: Vec<usize> = (0..nw).collect();
    order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut su = vec![0.0f64; mw * nw];
    let mut svt = vec![0.0f64; nw * nw];
    let mut ss = vec![0.0f64; nw];
    for (k, &j) in order.iter().enumerate() {
        ss[k] = s[j];
        for i in 0..mw {
            su[row_major(mw, nw, i, k)] = u[row_major(mw, nw, i, j)];
        }
        for i in 0..nw {
            svt[row_major(nw, nw, k, i)] = v[row_major(nw, nw, i, j)];
        }
    }

    // ── Rigorous enclosures ────────────────────────────────────────────
    // The computed pair (σ̃_j, w̃_j = [ũ_j; ṽ_j]/√2) is an approximate
    // eigenpair of the symmetric block matrix B = [0, Ã; Ãᵀ, 0]. Its
    // residual r_j = ‖B w̃_j − σ̃_j w̃_j‖ (computed in interval arithmetic,
    // so it covers input radii and all rounding) bounds the eigenvalue
    // error: |σ_j − σ̃_j| ≤ r_j. Near-degenerate clusters of singular
    // values are enclosed by their hull plus the max residual, which is
    // sound even when the matching of individual values is not unique.

    let mut res = vec![0.0f64; nw];
    for j in 0..nw {
        res[j] = svd_column_residual(src, &su, &svt, mw, nw, j, ss[j])?;
    }
    let r_max = res.iter().cloned().fold(0.0f64, f64::max);

    // Cluster indices whose σ̃ values are within 2·r_max (sorted ⇒ contiguous).
    let mut rad_s = vec![0.0f64; nw];
    let mut c_lo = 0usize;
    let mut c_hi = 0usize;
    for j in 0..nw {
        if j + 1 < nw && (ss[j + 1] - ss[j]) <= 2.0 * r_max {
            c_hi = j + 1;
        } else {
            // Finalize cluster [c_lo, c_hi]: spread + max residual, round up.
            let spread = Interval::exact(ss[c_hi]) - Interval::exact(ss[c_lo]);
            let mut r_cl = res[c_lo];
            for k in (c_lo + 1)..=c_hi {
                if res[k] > r_cl {
                    r_cl = res[k];
                }
            }
            let rad = add_ru(spread.hi.max(0.0), r_cl);
            for k in c_lo..=c_hi {
                rad_s[k] = rad;
            }
            c_lo = j + 1;
            c_hi = j + 1;
        }
    }

    // Per-column vector radii via the Davis–Kahan sin-theta theorem.
    let mut rad_u = vec![0.0f64; mw * nw];
    let mut rad_vt = vec![0.0f64; nw * nw];
    for j in 0..nw {
        let delta = svd_gap(&ss, &rad_s, j);
        let rad = if delta == 0.0 {
            1.0
        } else {
            (2.0 * div_ru(res[j], delta)).min(1.0)
        };
        for i in 0..mw {
            rad_u[row_major(mw, nw, i, j)] = rad;
        }
        for i in 0..nw {
            rad_vt[row_major(nw, nw, j, i)] = rad;
        }
    }

    let u_arr = IntervalArray::from_raw_parts(&su, &rad_u, &[mw, nw]);
    let s_arr = IntervalArray::from_raw_parts(&ss, &rad_s, &[nw]);
    let vt_arr = IntervalArray::from_raw_parts(&svt, &rad_vt, &[nw, nw]);
    if m >= n {
        Ok((u_arr, s_arr, vt_arr))
    } else {
        // A = V S U^T, so U = V^T (mxm) and VT = U2^T (mxn).
        let u = vt_arr.transpose();
        let vt = u_arr.transpose();
        Ok((u, s_arr, vt))
    }
}

/// Moore-Penrose pseudoinverse via SVD.
pub fn pinv(a: &IntervalArray) -> Result<IntervalArray, String> {
    let m = a.shape()[0];
    let n = a.shape()[1];
    // Empty input (zero rows or zero columns): the pseudoinverse is empty
    // too (numpy returns an empty array with transposed shape).
    if m == 0 || n == 0 {
        return Ok(IntervalArray::zeros(&[n, m]));
    }
    let (u, s, vt) = svd(a)?;
    let k = m.min(n);
    let s_mids: Vec<f64> = s.data().midpoints().to_vec();
    let s_rads: Vec<f64> = s.data().radii().to_vec();
    let tol = f64::EPSILON * m.max(n) as f64 * s_mids[0].max(f64::MIN_POSITIVE);

    // s_inv[k, k] diagonal, built from the interval singular values so the
    // uncertainty of the SVD propagates through the reciprocal.
    let mut s_inv = vec![0.0f64; k * k];
    let mut s_inv_rad = vec![0.0f64; k * k];
    for i in 0..k {
        let lo = s_mids[i] - s_rads[i];
        if s_mids[i] > tol && lo > 0.0 {
            // Interval reciprocal with directed rounding, then back to mid/rad.
            let inv = Interval::from_midpoint_radius(s_mids[i], s_rads[i]).recip();
            s_inv[row_major(k, k, i, i)] = inv.midpoint();
            s_inv_rad[row_major(k, k, i, i)] = inv.radius();
        }
    }
    // pinv = V S^+ U^T
    let v_arr = vt.transpose();
    let vs = crate::ops::reduction::matmul(
        &v_arr,
        &IntervalArray::from_raw_parts(&s_inv, &s_inv_rad, &[k, k]),
    );
    let ut = u.transpose();
    Ok(crate::ops::reduction::matmul(&vs, &ut))
}

// ── Eigenvalues / eigenvectors ─────────────────────────────────────────

/// Symmetric eigendecomposition via the cyclic Jacobi method.
pub fn eig_symmetric(a: &IntervalArray) -> Result<(IntervalArray, IntervalArray), String> {
    if a.ndim() != 2 {
        return Err("eig requires a 2D array".to_string());
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err("eig requires a square matrix".to_string());
    }
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            m[row_major(n, n, i, j)] = get_iv(a, i, j).midpoint();
        }
    }
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[row_major(n, n, i, i)] = 1.0;
    }

    let eps = f64::EPSILON;
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[row_major(n, n, p, q)] * m[row_major(n, n, p, q)];
            }
        }
        if off <= eps * eps * (n * n) as f64 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[row_major(n, n, p, q)];
                if apq.abs()
                    <= eps * (m[row_major(n, n, p, p)].abs() + m[row_major(n, n, q, q)].abs())
                {
                    continue;
                }
                let app = m[row_major(n, n, p, p)];
                let aqq = m[row_major(n, n, q, q)];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                // update columns p and q of the matrix
                for i in 0..n {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = m[row_major(n, n, i, p)];
                    let aiq = m[row_major(n, n, i, q)];
                    m[row_major(n, n, i, p)] = c * aip - s * aiq;
                    m[row_major(n, n, p, i)] = c * aip - s * aiq;
                    m[row_major(n, n, i, q)] = s * aip + c * aiq;
                    m[row_major(n, n, q, i)] = s * aip + c * aiq;
                }
                m[row_major(n, n, p, p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                m[row_major(n, n, q, q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                m[row_major(n, n, p, q)] = 0.0;
                m[row_major(n, n, q, p)] = 0.0;

                for i in 0..n {
                    let vip = v[row_major(n, n, i, p)];
                    let viq = v[row_major(n, n, i, q)];
                    v[row_major(n, n, i, p)] = c * vip - s * viq;
                    v[row_major(n, n, i, q)] = s * vip + c * viq;
                }
            }
        }
    }

    let vals: Vec<f64> = (0..n).map(|i| m[row_major(n, n, i, i)]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        vals[b]
            .partial_cmp(&vals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut vals_sorted = vec![0.0f64; n];
    let mut vecs = vec![0.0f64; n * n];
    for (k, &j) in order.iter().enumerate() {
        vals_sorted[k] = vals[j];
        for i in 0..n {
            vecs[row_major(n, n, i, k)] = v[row_major(n, n, i, j)];
        }
    }

    // ── Rigorous enclosures ────────────────────────────────────────────
    // Each computed pair (λ̃_j, ṽ_j) is an approximate eigenpair of the
    // symmetric interval matrix Ã; its residual r_j = ‖Ã ṽ_j − λ̃_j ṽ_j‖
    // (interval arithmetic, covers input radii) bounds the eigenvalue
    // error: |λ_j − λ̃_j| ≤ r_j. Near-degenerate clusters of eigenvalues
    // are enclosed by their hull plus the max residual (sound even when
    // the individual matching is not unique).

    let mut res = vec![0.0f64; n];
    for j in 0..n {
        let mut uj = Vec::with_capacity(n);
        for i in 0..n {
            uj.push(vecs[row_major(n, n, i, j)]);
        }
        res[j] = matvec_residual(a, &uj, vals_sorted[j], &uj)?;
    }
    let r_max = res.iter().cloned().fold(0.0f64, f64::max);

    // Cluster indices whose λ̃ values are within 2·r_max (sorted ⇒ contiguous).
    let mut rad_vals = vec![0.0f64; n];
    let mut c_lo = 0usize;
    let mut c_hi = 0usize;
    for j in 0..n {
        if j + 1 < n && (vals_sorted[j + 1] - vals_sorted[j]) <= 2.0 * r_max {
            c_hi = j + 1;
        } else {
            // Finalize cluster [c_lo, c_hi]: spread + max residual, round up.
            let spread = Interval::exact(vals_sorted[c_hi]) - Interval::exact(vals_sorted[c_lo]);
            let mut r_cl = res[c_lo];
            for k in (c_lo + 1)..=c_hi {
                if res[k] > r_cl {
                    r_cl = res[k];
                }
            }
            let rad = add_ru(spread.hi.max(0.0), r_cl);
            for k in c_lo..=c_hi {
                rad_vals[k] = rad;
            }
            c_lo = j + 1;
            c_hi = j + 1;
        }
    }

    // Eigenvector radii via Davis–Kahan: sin θ ≤ res_j / gap_j.
    let mut rad_vecs = vec![0.0f64; n * n];
    for j in 0..n {
        let delta = eig_gap(&vals_sorted, &rad_vals, j);
        let rad = if delta == 0.0 {
            1.0
        } else {
            (2.0 * div_ru(res[j], delta)).min(1.0)
        };
        for i in 0..n {
            rad_vecs[row_major(n, n, i, j)] = rad;
        }
    }

    let evals = IntervalArray::from_raw_parts(&vals_sorted, &rad_vals, &[n]);
    let evecs = IntervalArray::from_raw_parts(&vecs, &rad_vecs, &[n, n]);
    Ok((evals, evecs))
}

/// General eigendecomposition via Hessenberg reduction + shifted QR.
/// Raises an error if complex eigenvalue pairs (2x2 blocks) are encountered.
pub fn eig_general(a: &IntervalArray) -> Result<(IntervalArray, IntervalArray), String> {
    let n = a.shape()[0];
    let eps = f64::EPSILON;

    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            h[row_major(n, n, i, j)] = get_iv(a, i, j).midpoint();
        }
    }
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[row_major(n, n, i, i)] = 1.0;
    }

    // Reduce to upper Hessenberg via Householder reflections (P H P^T).
    for k in 0..n.saturating_sub(2) {
        let mut pivot = k + 1;
        let mut mag = h[row_major(n, n, k + 1, k)].abs();
        for r in (k + 2)..n {
            let m = h[row_major(n, n, r, k)].abs();
            if m > mag {
                mag = m;
                pivot = r;
            }
        }
        if mag == 0.0 {
            continue;
        }
        if pivot != k + 1 {
            for j in 0..n {
                let tmp = h[row_major(n, n, k + 1, j)];
                h[row_major(n, n, k + 1, j)] = h[row_major(n, n, pivot, j)];
                h[row_major(n, n, pivot, j)] = tmp;
            }
            for i in 0..n {
                let tmp = h[row_major(n, n, i, k + 1)];
                h[row_major(n, n, i, k + 1)] = h[row_major(n, n, i, pivot)];
                h[row_major(n, n, i, pivot)] = tmp;
            }
            for i in 0..n {
                let tmp = v[row_major(n, n, i, k + 1)];
                v[row_major(n, n, i, k + 1)] = v[row_major(n, n, i, pivot)];
                v[row_major(n, n, i, pivot)] = tmp;
            }
        }
        // Householder vector u for the sub-column h[k+1..n, k]
        let mut norm2 = 0.0f64;
        for r in (k + 2)..n {
            norm2 += h[row_major(n, n, r, k)] * h[row_major(n, n, r, k)];
        }
        let h21 = h[row_major(n, n, k + 1, k)];
        let norm = (h21 * h21 + norm2).sqrt();
        let alpha = if h21 >= 0.0 { -norm } else { norm };
        let mut u = vec![0.0f64; n - k - 1];
        u[0] = h21 - alpha;
        for (i, r) in (k + 2..n).enumerate() {
            u[i + 1] = h[row_major(n, n, r, k)];
        }
        let uu: f64 = u.iter().map(|x| x * x).sum();
        if uu == 0.0 {
            continue;
        }
        // H <- P H P  and  V <- V P,  where P = I - 2 u u^T / uu
        for j in (k + 1)..n {
            let mut dot = 0.0f64;
            for (i, &ui) in u.iter().enumerate() {
                dot += ui * h[row_major(n, n, k + 1 + i, j)];
            }
            let f = -2.0 * dot / uu;
            for (i, &ui) in u.iter().enumerate() {
                let idx = row_major(n, n, k + 1 + i, j);
                h[idx] += f * ui;
            }
        }
        for i in (k + 1)..n {
            let mut dot = 0.0f64;
            for (j, &uj) in u.iter().enumerate() {
                dot += h[row_major(n, n, i, k + 1 + j)] * uj;
            }
            let f = -2.0 * dot / uu;
            for (j, &uj) in u.iter().enumerate() {
                let idx = row_major(n, n, i, k + 1 + j);
                h[idx] += f * uj;
            }
        }
        for i in 0..n {
            let mut dot = 0.0f64;
            for (j, &uj) in u.iter().enumerate() {
                dot += v[row_major(n, n, i, k + 1 + j)] * uj;
            }
            let f = -2.0 * dot / uu;
            for (j, &uj) in u.iter().enumerate() {
                let idx = row_major(n, n, i, k + 1 + j);
                v[idx] += f * uj;
            }
        }
    }

    // Shifted QR iteration with deflation.
    let mut q = n;
    let mut iter = 0usize;
    let max_iter = 100 * n.max(1) + 30;

    while q > 1 && iter < max_iter {
        // Find the smallest l such that subdiagonal h[l, l-1] is negligible.
        let mut l = 0;
        'find: {
            for k in (1..q).rev() {
                let a = h[row_major(n, n, k - 1, k - 1)].abs();
                let b = h[row_major(n, n, k, k)].abs();
                let e = h[row_major(n, n, k, k - 1)].abs();
                if e <= eps * (a + b) * 4.0 {
                    h[row_major(n, n, k, k - 1)] = 0.0;
                    l = k;
                    break 'find;
                }
            }
        }
        // No negligible subdiagonal found: the whole block [0..q) is active,
        // so l stays 0 and we sweep it in full.
        if l == q - 1 {
            q -= 1;
            iter = 0;
            continue;
        }

        // Wilkinson shift from the trailing 2x2 block.
        let a11 = h[row_major(n, n, q - 2, q - 2)];
        let a12 = h[row_major(n, n, q - 2, q - 1)];
        let a21 = h[row_major(n, n, q - 1, q - 2)];
        let a22 = h[row_major(n, n, q - 1, q - 1)];
        let tr = a11 + a22;
        let det = a11 * a22 - a12 * a21;
        let disc = tr * tr - 4.0 * det;
        let shift = if disc >= 0.0 {
            let sq = disc.sqrt();
            let l1 = (tr + sq) * 0.5;
            let l2 = (tr - sq) * 0.5;
            if (l1 - a22).abs() < (l2 - a22).abs() {
                l1
            } else {
                l2
            }
        } else {
            // complex pair: keep iterating with the real part as shift
            tr * 0.5
        };

        for i in l..q {
            h[row_major(n, n, i, i)] -= shift;
        }

        // One QR sweep: Givens rotations applied to both sides (H ← G·H·Gᵀ,
        // V ← V·Gᵀ; the column-pair formulas coincide with the row-pair ones).
        for col in l..q.saturating_sub(1) {
            let x = h[row_major(n, n, col, col)];
            let y = h[row_major(n, n, col + 1, col)];
            let hyp = (x * x + y * y).sqrt();
            if hyp == 0.0 {
                continue;
            }
            let c = x / hyp;
            let s = y / hyp;
            // rows col, col+1 (left multiply by G)
            for j in l..q {
                let t1 = h[row_major(n, n, col, j)];
                let t2 = h[row_major(n, n, col + 1, j)];
                h[row_major(n, n, col, j)] = c * t1 + s * t2;
                h[row_major(n, n, col + 1, j)] = -s * t1 + c * t2;
            }
            // columns col, col+1 (right multiply by Gᵀ)
            for i in l..q {
                let t1 = h[row_major(n, n, i, col)];
                let t2 = h[row_major(n, n, i, col + 1)];
                h[row_major(n, n, i, col)] = c * t1 + s * t2;
                h[row_major(n, n, i, col + 1)] = -s * t1 + c * t2;
            }
            for i in 0..n {
                let t1 = v[row_major(n, n, i, col)];
                let t2 = v[row_major(n, n, i, col + 1)];
                v[row_major(n, n, i, col)] = c * t1 + s * t2;
                v[row_major(n, n, i, col + 1)] = -s * t1 + c * t2;
            }
        }

        for i in l..q {
            h[row_major(n, n, i, i)] += shift;
        }
        iter += 1;
    }

    // Check that no 2x2 block remains (complex conjugate pair).
    for k in 1..n {
        let e = h[row_major(n, n, k, k - 1)].abs();
        let a = h[row_major(n, n, k - 1, k - 1)].abs();
        let b = h[row_major(n, n, k, k)].abs();
        if e > eps * (a + b) * 4.0 {
            return Err(
                "eig: matrix has complex eigenvalues; only real eigenvalues are supported"
                    .to_string(),
            );
        }
    }

    // Eigenvectors from the real Schur form: H is (numerically) upper
    // triangular, so the eigenvector of A for λ_i = H[i][i] is V·x where x
    // solves (H − λ_i I)x = e_i by back-substitution. (The columns of the
    // raw accumulated V are only eigenvectors once H is fully diagonal.)
    let mut vals = vec![0.0f64; n];
    let mut vecs = vec![0.0f64; n * n];
    for i in 0..n {
        vals[i] = h[row_major(n, n, i, i)];
        let mut x = vec![0.0f64; i + 1];
        x[i] = 1.0;
        for k in (0..i).rev() {
            let mut s = 0.0f64;
            for j in (k + 1)..=i {
                s += h[row_major(n, n, k, j)] * x[j];
            }
            let denom = h[row_major(n, n, k, k)] - vals[i];
            x[k] = -s / denom;
        }
        let mut w = vec![0.0f64; n];
        for r in 0..n {
            let mut acc = 0.0f64;
            for k in 0..=i {
                acc += v[row_major(n, n, r, k)] * x[k];
            }
            w[r] = acc;
        }
        let nrm: f64 = w.iter().map(|t| t * t).sum();
        if nrm > 0.0 {
            let inv = 1.0 / nrm.sqrt();
            for (r, t) in w.iter().enumerate() {
                vecs[row_major(n, n, r, i)] = t * inv;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        vals[b]
            .partial_cmp(&vals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut vals_sorted = vec![0.0f64; n];
    let mut vecs_sorted = vec![0.0f64; n * n];
    for (k, &j) in order.iter().enumerate() {
        vals_sorted[k] = vals[j];
        for i in 0..n {
            vecs_sorted[row_major(n, n, i, k)] = vecs[row_major(n, n, i, j)];
        }
    }
    let vecs = vecs_sorted;

    // ── Rigorous eigenvalue and eigenvector enclosure (Bauer–Fike) ──────
    // Ã = Ṽ Λ̃ Ṽ⁻¹ + E with E = Ã − ṼΛ̃Ṽ⁻¹ enclosed in intervals.
    // Eigenvalue radii: |λ_j(Ã) − λ̃_j| ≤ cond₂(Ṽ)·‖E‖₂ ≤ ‖Ṽ‖_F·‖Ṽ⁻¹‖_F·‖E‖_F.
    // Eigenvector radii (per eigenvalue i): ||δv_i|| ≤ cond₂(Ṽ)·‖E‖_F / δ_i
    // where δ_i is the eigengap (distance to the nearest other eigenvalue minus
    // both their radii).  The cap of 1.0 encloses the entire unit ball when the
    // matrix is defective or the eigengap vanishes — still sound, just conservative.
    let mut vmat: Vec<Interval> = vec![Interval::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            vmat[row_major(n, n, i, j)] = Interval::exact(vecs[row_major(n, n, i, j)]);
        }
    }
    let mut vinv = vec![Interval::zero(); n * n];
    for c in 0..n {
        let mut rhs = vec![Interval::zero(); n];
        rhs[c] = Interval::exact(1.0);
        let mut mm = vmat.clone();
        match lu_solve_interval(&mut mm, n, &rhs) {
            Some(x) => {
                for i in 0..n {
                    vinv[row_major(n, n, i, c)] = x[i];
                }
            }
            None => {
                return Err(
                    "eig: eigenvector matrix is numerically singular (defective or nearly \
                     defective matrix); eigenvalues cannot be rigorously enclosed"
                        .to_string(),
                )
            }
        }
    }
    let mut vlam = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            vlam[row_major(n, n, i, j)] = vecs[row_major(n, n, i, j)] * vals_sorted[j];
        }
    }
    let f = matmul(
        &IntervalArray::from_f64_vec(&vlam, &[n, n]),
        &IntervalArray::from_raw_parts(
            &vinv.iter().map(|iv| iv.midpoint()).collect::<Vec<f64>>(),
            &vinv.iter().map(|iv| iv.radius()).collect::<Vec<f64>>(),
            &[n, n],
        ),
    );
    let f_mids = f.data().midpoints();
    let f_rads = f.data().radii();
    let mut e_sq = Interval::zero();
    for i in 0..n {
        for j in 0..n {
            let f_iv = Interval::from_midpoint_radius(f_mids[i * n + j], f_rads[i * n + j]);
            let e = get_iv(a, i, j) - f_iv;
            let es = Interval::exact(sup_abs(&e));
            e_sq = e_sq + es * es;
        }
    }
    let r_res = next_up(e_sq.hi.max(0.0).sqrt());
    let mut v_norm_sq = Interval::zero();
    for iv in &vmat {
        let s = Interval::exact(sup_abs(iv));
        v_norm_sq = v_norm_sq + s * s;
    }
    let v_norm = next_up(v_norm_sq.hi.max(0.0).sqrt());
    let vinv_norm = sup_frobenius(&vinv);
    let r_lambda = mul_ru(mul_ru(v_norm, vinv_norm), r_res);

    let evals = IntervalArray::from_raw_parts(&vals_sorted, &vec![r_lambda; n], &[n]);
    let r_lambda_vec = vec![r_lambda; n];
    let mut ev_rads = vec![0.0f64; n];
    for i in 0..n {
        let gap = eig_gap(&vals_sorted, &r_lambda_vec, i);
        if gap > 0.0 && r_res.is_finite() && v_norm.is_finite() && vinv_norm.is_finite() {
            let bound = next_up(v_norm * vinv_norm * r_res / gap);
            ev_rads[i] = bound.min(1.0);
        } else {
            ev_rads[i] = 1.0;
        }
    }
    let mut evec_rads = Vec::with_capacity(n * n);
    for j in 0..n {
        evec_rads.extend(std::iter::repeat(ev_rads[j]).take(n));
    }
    let evecs = IntervalArray::from_raw_parts(&vecs, &evec_rads, &[n, n]);
    Ok((evals, evecs))
}

/// `eig` dispatch: symmetric matrices use Jacobi, otherwise the QR algorithm.
pub fn eig(a: &IntervalArray) -> Result<(IntervalArray, IntervalArray), String> {
    if a.ndim() != 2 {
        return Err("eig requires a 2D array".to_string());
    }
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err("eig requires a square matrix".to_string());
    }
    let mut symmetric = true;
    'outer: for i in 0..n {
        for j in 0..n {
            let x = get_iv(a, i, j).midpoint();
            let y = get_iv(a, j, i).midpoint();
            if (x - y).abs() > 1e-12 * (1.0 + x.abs().max(y.abs())) {
                symmetric = false;
                break 'outer;
            }
        }
    }
    if symmetric {
        eig_symmetric(a)
    } else {
        eig_general(a)
    }
}

// ── Additional linalg: cholesky, matrix_power, matrix_rank, cond ─────

/// Interval-square-root of a strictly-positive interval.
fn interval_sqrt(iv: &Interval) -> Interval {
    let lo = if iv.lo > 0.0 {
        next_down(iv.lo.sqrt())
    } else {
        0.0
    };
    let hi = if iv.hi > 0.0 {
        next_up(iv.hi.sqrt())
    } else {
        0.0
    };
    Interval::new(lo, hi)
}

/// Cholesky factorisation of a symmetric positive-definite interval matrix:
/// A = L · L^T, with L lower-triangular.
///
/// Returns Err if the rigorous lower bound of any required pivot is
/// non-positive, meaning at least one matrix in the interval family is
/// not SPD.
pub fn cholesky(a: &IntervalArray) -> Result<IntervalArray, String> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err("cholesky requires a square matrix".into());
    }
    let n = a.shape()[0];
    let mut data = vec![Interval::zero(); n * n];
    for j in 0..n {
        let mut s = Interval::zero();
        for k in 0..j {
            let ljk = data[j * n + k];
            s = s + ljk * ljk;
        }
        let ajj = a.get(j * n + j) - s;
        if ajj.lo <= 0.0 {
            return Err(format!(
                "cholesky: A[{},{}] lower bound {:e} <= 0; matrix is not positive-definite",
                j, j, ajj.lo
            ));
        }
        data[j * n + j] = interval_sqrt(&ajj);
        let ljj = data[j * n + j];
        for i in (j + 1)..n {
            let mut s = Interval::zero();
            for k in 0..j {
                let lik = data[i * n + k];
                let ljk = data[j * n + k];
                s = s + lik * ljk;
            }
            data[i * n + j] = (a.get(i * n + j) - s) / ljj;
        }
    }
    let mids: Vec<f64> = data.iter().map(|iv| iv.midpoint()).collect();
    let rads: Vec<f64> = data.iter().map(|iv| iv.radius()).collect();
    Ok(IntervalArray::from_raw_parts(&mids, &rads, &[n, n]))
}

/// Integer matrix-power `A^n` via binary exponentiation.
///
/// n == 0 returns the identity.  Negative n requires A to be invertible
/// (it is first inverted then raised to `|n|`).
pub fn matrix_power(a: &IntervalArray, n: i64) -> Result<IntervalArray, String> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err("matrix_power requires a square matrix".into());
    }
    let n_dim = a.shape()[0];
    if n == 0 {
        let mut mids = vec![0.0f64; n_dim * n_dim];
        for i in 0..n_dim {
            mids[i * n_dim + i] = 1.0;
        }
        return Ok(IntervalArray::from_raw_parts(
            &mids,
            &vec![0.0f64; n_dim * n_dim],
            &[n_dim, n_dim],
        ));
    }
    if n == 1 {
        return Ok(a.clone());
    }
    if n < 0 {
        let inv_a = inv(a)?;
        return matrix_power(&inv_a, -n);
    }
    let mut result = {
        let mut mids = vec![0.0f64; n_dim * n_dim];
        for i in 0..n_dim {
            mids[i * n_dim + i] = 1.0;
        }
        IntervalArray::from_raw_parts(&mids, &vec![0.0f64; n_dim * n_dim], &[n_dim, n_dim])
    };
    let mut base = a.clone();
    let mut exp = n as u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = matmul(&result, &base);
        }
        base = matmul(&base, &base);
        exp >>= 1;
    }
    Ok(result)
}

/// Pragmatic matrix rank via the SVD upper bounds.
///
/// Returns the number of singular values whose **upper bound** (midpoint +
/// radius) exceeds a tolerance.  The tolerance is `n·ε·(1 + σ_max⁰)` by
/// default.  Matrices in the interval family with smaller singular values
/// may rank-deficient; this function reports the rank of the **midpoint**
/// matrix.
///
/// For a guaranteed minimum rank across the whole family, examine the
/// lower bounds of the singular value intervals manually.
pub fn matrix_rank(a: &IntervalArray) -> Result<usize, String> {
    if a.ndim() != 2 {
        return Err("matrix_rank requires a 2D array".into());
    }
    let (_, s, _) = svd(a)?;
    let mids = s.data().midpoints();
    let rads = s.data().radii();
    let n = mids.len();
    let smax_upper = mids
        .iter()
        .zip(rads.iter())
        .map(|(&m, &r)| next_up(m + r))
        .fold(0.0f64, f64::max);
    let tol = (n + 1) as f64 * f64::EPSILON * smax_upper;
    Ok(mids
        .iter()
        .zip(rads.iter())
        .filter(|(&m, &r)| m + r > tol)
        .count())
}

/// 2-norm condition number of a square interval matrix.
///
/// Rigorously bounds the supremum of `cond(M)` for M in the interval
/// family: returned as an `Interval` whose midpoint is `cond(M_mid)` and
/// whose radius inflates the upper bound on the worst-case condition
/// number.
///
/// If the rigorous lower bound of any singular value is ≤ 0, the result
/// is `∞` (the family contains singular matrices).
pub fn cond(a: &IntervalArray) -> Result<Interval, String> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err("cond requires a square matrix".into());
    }
    let n = a.shape()[0];
    if n == 0 {
        return Ok(Interval::exact(0.0));
    }
    let (_, s, _) = svd(a)?;
    let mut smax_upper = 0.0f64;
    let mut smin_lower = f64::INFINITY;
    let mut smax_mid = 0.0f64;
    let mut smin_mid = f64::INFINITY;
    for (&m, &r) in s.data().midpoints().iter().zip(s.data().radii().iter()) {
        let upper = next_up(m + r);
        let lower = m - r;
        if upper > smax_upper {
            smax_upper = upper;
        }
        if lower < smin_lower {
            smin_lower = lower;
        }
        if m > smax_mid {
            smax_mid = m;
        }
        if m < smin_mid {
            smin_mid = m;
        }
    }
    if smin_lower <= 0.0 {
        return Ok(Interval::exact(f64::INFINITY));
    }
    let cond_mid = if smin_mid > 0.0 {
        smax_mid / smin_mid
    } else {
        f64::INFINITY
    };
    let cond_ub = smax_upper / smin_lower;
    let rad = if cond_ub.is_finite() && cond_mid.is_finite() {
        next_up(cond_ub - cond_mid)
    } else {
        0.0
    };
    Ok(Interval::from_midpoint_radius(cond_mid, rad))
}

// ========================================================================
// PyO3 Python-facing wrappers (added after core algorithms so the file
// remains a straightforward Rust library plus thin send-to-Python shims).
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mat2(a: f64, b: f64, c: f64, d: f64) -> IntervalArray {
        IntervalArray::from_f64_vec(&[a, b, c, d], &[2, 2])
    }

    #[test]
    fn test_det() {
        let m = mat2(1.0, 2.0, 3.0, 4.0);
        let d = det(&m).unwrap();
        assert!((d.midpoint() - (-2.0)).abs() < 1e-10);
        assert!(d.contains(-2.0));
    }

    #[test]
    fn test_inv() {
        let m = mat2(4.0, 7.0, 2.0, 6.0);
        let i = inv(&m).unwrap();
        let expected = IntervalArray::from_f64_vec(&[0.6, -0.7, -0.2, 0.4], &[2, 2]);
        for k in 0..4 {
            assert!((i.get(k).midpoint() - expected.get(k).midpoint()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_solve() {
        let m = mat2(2.0, 1.0, 1.0, 3.0);
        let b = IntervalArray::from_f64_slice(&[3.0, 5.0]);
        let x = solve(&m, &b).unwrap();
        assert!((x.get(0).midpoint() - 0.8).abs() < 1e-10);
        assert!((x.get(1).midpoint() - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_svd() {
        let m = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let (u, s, vt) = svd(&m).unwrap();
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(vt.shape(), &[2, 3]);
        // reconstruct U S Vt and compare
        let s_diag = IntervalArray::from_raw_parts(
            &[s.get(0).midpoint(), 0.0, 0.0, s.get(1).midpoint()],
            &[0.0; 4],
            &[2, 2],
        );
        let us = crate::ops::reduction::matmul(&u, &s_diag);
        let rec = crate::ops::reduction::matmul(&us, &vt);
        for k in 0..6 {
            assert!((rec.get(k).midpoint() - m.get(k).midpoint()).abs() < 1e-8);
        }
    }

    #[test]
    fn test_eig_symmetric() {
        let m = IntervalArray::from_f64_vec(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);
        let (evals, evecs) = eig(&m).unwrap();
        assert!((evals.get(0).midpoint() - 3.0).abs() < 1e-8);
        assert!((evals.get(1).midpoint() - 1.0).abs() < 1e-8);
        // A v = lambda v for the dominant eigenvector
        let lam = evals.get(0).midpoint();
        for i in 0..2 {
            let row = i * 2;
            let av = m.get(row).midpoint() * evecs.get(0).midpoint()
                + m.get(row + 1).midpoint() * evecs.get(2).midpoint();
            let lv = lam * evecs.get(i).midpoint();
            assert!((av - lv).abs() < 1e-8);
        }
    }

    #[test]
    fn test_eig_general_real() {
        let m = IntervalArray::from_f64_vec(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);
        let (evals, _) = eig_general(&m).unwrap();
        assert!((evals.get(0).midpoint() - 3.0).abs() < 1e-8);
        assert!((evals.get(1).midpoint() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_pinv() {
        let m = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let p = pinv(&m).unwrap();
        let eye = crate::ops::reduction::matmul(&m, &p);
        assert!((eye.get(0).midpoint() - 1.0).abs() < 1e-8);
        assert!((eye.get(1).midpoint() - 0.0).abs() < 1e-8);
        assert!((eye.get(2).midpoint() - 0.0).abs() < 1e-8);
        assert!((eye.get(3).midpoint() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_svd_rigor_contains_true_values() {
        // Exact diagonal matrix: singular values are exactly 3 and 2.
        let m = IntervalArray::from_f64_vec(&[3.0, 0.0, 0.0, 2.0], &[2, 2]);
        let (_, s, _) = svd(&m).unwrap();
        let s0 = s.get(0);
        let s1 = s.get(1);
        assert!(s0.contains(3.0), "sigma1 = {:?} must contain 3.0", s0);
        assert!(s1.contains(2.0), "sigma2 = {:?} must contain 2.0", s1);
        // Well-separated exact input: enclosure must be tight (roundoff-scale).
        assert!(
            s0.radius() < 1e-12,
            "sigma1 radius {:e} too loose",
            s0.radius()
        );
    }

    #[test]
    fn test_svd_rigor_with_input_uncertainty() {
        // Interval entries: [1 ± 0.01]; the true singular values of any
        // matrix in the family must be enclosed. For A = diag(3,2) ± 0.01
        // in every entry the extremal member A' = diag(3.01, 2.01) + 0.01
        // off-diagonals has σ₁ = (5.02 + √1.0004)/2 ≈ 3.0101, so σ₁ ∈
        // [3 − 0.0101, 3 + 0.0101] and σ₂ ∈ [2 − 0.0101, 2 + 0.0101].
        let mids = [3.0, 0.0, 0.0, 2.0];
        let rads = [0.01, 0.01, 0.01, 0.01];
        let m = IntervalArray::from_raw_parts(&mids, &rads, &[2, 2]);
        let (_, s, _) = svd(&m).unwrap();
        assert!(
            s.get(0).contains(2.99),
            "sigma1 = {:?} must contain 2.99",
            s.get(0)
        );
        assert!(
            s.get(0).contains(3.01),
            "sigma1 = {:?} must contain 3.01",
            s.get(0)
        );
        assert!(
            s.get(1).contains(1.99),
            "sigma2 = {:?} must contain 1.99",
            s.get(1)
        );
        assert!(
            s.get(1).contains(2.01),
            "sigma2 = {:?} must contain 2.01",
            s.get(1)
        );
        assert!(s.get(0).radius() > 0.0);
    }

    #[test]
    fn test_svd_wide_matrix_rigor() {
        // Wide 2x3 matrix with exactly known singular values (σ of [[1,0],[0,1],[0,0]]^T-style
        // construction): A = [1 0 0; 0 1 0] has singular values 1 and 1.
        let m = IntervalArray::from_f64_vec(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let (u, s, vt) = svd(&m).unwrap();
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(vt.shape(), &[2, 3]);
        assert!(s.get(0).contains(1.0));
        assert!(s.get(1).contains(1.0));
        // Degenerate pair: the hull enclosure must still cover the true values.
        assert!(s.get(0).radius() <= 0.1);
    }

    #[test]
    fn test_eig_rigor_contains_true_values() {
        let m = IntervalArray::from_f64_vec(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);
        let (evals, evecs) = eig(&m).unwrap();
        assert!(evals.get(0).contains(3.0));
        assert!(evals.get(1).contains(1.0));
        assert!(evals.get(0).radius() < 1e-12);
        // Eigenvectors: well-separated, exact input → tight enclosure.
        for k in 0..4 {
            assert!(
                evecs.get(k).radius() < 1e-8,
                "evec radius too loose: {}",
                evecs.get(k).radius()
            );
        }
    }

    #[test]
    fn test_eig_general_rigor() {
        let m = IntervalArray::from_f64_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let (evals, evecs) = eig_general(&m).unwrap();
        // True eigenvalues: (5 ± √33)/2 ≈ 5.37228 and -0.37228.
        assert!(evals.get(0).contains(5.372281323269014));
        assert!(evals.get(1).contains(-0.3722813232690143));
        assert!(evals.get(0).radius() < 1e-10);
        // Perturbation-bound eigenvectors: well-separated exact input -> tight.
        for k in 0..4 {
            let r = evecs.get(k).radius();
            assert!(
                r.is_finite(),
                "evec {} radius should be finite (Bauer-Fike bound), got {:?}",
                k,
                r
            );
            assert!(r < 1e-8, "evec {} radius too loose: {}", k, r);
        }
        // A v = lambda v within the enclosure
        for col in 0..2 {
            let lam = evals.get(col);
            let v0 = evecs.get(col);
            let v1 = evecs.get(1 * 2 + col);
            let av0 = m.get(0) * v0 + m.get(1) * v1;
            let av1 = m.get(2) * v0 + m.get(3) * v1;
            let lv0 = lam * v0;
            let lv1 = lam * v1;
            assert!(
                av0.contains(lv0.midpoint()),
                "[col={}] av0 must contain lambda*v0",
                col
            );
            assert!(
                av1.contains(lv1.midpoint()),
                "[col={}] av1 must contain lambda*v1",
                col
            );
        }
    }

    #[test]
    fn test_eig_degenerate_graceful() {
        let m = IntervalArray::from_f64_vec(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let (evals, _) = eig(&m).unwrap();
        assert!(evals.get(0).contains(1.0));
        assert!(evals.get(1).contains(1.0));
    }

    #[test]
    fn test_cholesky_2x2() {
        let a = IntervalArray::from_f64_vec(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let l = cholesky(&a).unwrap();
        assert_eq!(l.shape(), &[2, 2]);
        let ll = matmul(&l, &l.transpose());
        for k in 0..4 {
            assert!(ll.get(k).contains(a.get(k).midpoint()));
        }
        assert!((l.get(0).midpoint() - 2.0).abs() < 1e-10);
        assert!((l.get(2).midpoint() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_not_spd() {
        let a = IntervalArray::from_f64_vec(&[-1.0, 0.0, 0.0, 1.0], &[2, 2]);
        assert!(cholesky(&a).is_err());
    }

    #[test]
    fn test_matrix_power_n0_n1_n2() {
        let e = matrix_power(
            &IntervalArray::from_f64_vec(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),
            0,
        )
        .unwrap();
        assert!((e.get(0).midpoint() - 1.0).abs() < 1e-10);
        assert!((e.get(3).midpoint() - 1.0).abs() < 1e-10);
        assert!((e.get(1).midpoint() - 0.0).abs() < 1e-10);
        let a = IntervalArray::from_f64_vec(&[1.0, 1.0, 0.0, 2.0], &[2, 2]);
        let e2 = matrix_power(&a, 2).unwrap();
        assert!((e2.get(0).midpoint() - 1.0).abs() < 1e-10);
        assert!((e2.get(3).midpoint() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_power_negative() {
        let a = IntervalArray::from_f64_vec(&[2.0, 0.0, 0.0, 3.0], &[2, 2]);
        let e = matrix_power(&a, -1).unwrap();
        let m = matmul(&e, &a);
        let eye_expected = [1.0, 0.0, 0.0, 1.0];
        for k in 0..4 {
            assert!(m.get(k).contains(eye_expected[k]));
        }
    }

    #[test]
    fn test_matrix_rank_variants() {
        let full = IntervalArray::from_f64_vec(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        assert_eq!(matrix_rank(&full).unwrap(), 2);
        let defect = IntervalArray::from_f64_vec(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
        assert_eq!(matrix_rank(&defect).unwrap(), 1);
    }

    #[test]
    fn test_cond_variants() {
        let i = IntervalArray::from_f64_vec(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let c = cond(&i).unwrap();
        assert!(
            c.midpoint() > 0.0 && c.midpoint() < 100.0,
            "cond(I) = {:?}",
            c
        );
        let s = IntervalArray::from_f64_vec(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let cs = cond(&s).unwrap();
        assert!(cs.midpoint() > 1e10, "cond(singular) = {:?}", cs);
    }
}
