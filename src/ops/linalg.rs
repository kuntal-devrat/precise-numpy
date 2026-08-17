//! Linear algebra on interval matrices.
//!
//! `det`, `inv`, `solve`, and `lstsq` run through interval Gaussian
//! elimination, so error bounds propagate through every arithmetic step
//! with hardware-directed rounding. `eig` and `svd` use iterative
//! algorithms (Jacobi rotations / QR) on the midpoints and return exact
//! intervals, since rigorous interval versions are not practical.

use crate::array::IntervalArray;
use crate::error::Interval;

fn row_major(_m: usize, n: usize, i: usize, j: usize) -> usize {
    i * n + j
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
                    return Err(
                        "solve: singular matrix (interval contains zero pivot)".to_string()
                    )
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
    let mut eye_rads = vec![0.0f64; n * n];
    for i in 0..n {
        eye_mids[i * n + i] = 1.0;
    }
    let eye = IntervalArray::from_raw_parts(&eye_mids, &eye_rads, &[n, n]);
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

    let u_arr = IntervalArray::from_raw_parts(&su, &vec![0.0; mw * nw], &[mw, nw]);
    let s_arr = IntervalArray::from_raw_parts(&ss, &vec![0.0; nw], &[nw]);
    let vt_arr = IntervalArray::from_raw_parts(&svt, &vec![0.0; nw * nw], &[nw, nw]);
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
    let (u, s, vt) = svd(a)?;
    let m = a.shape()[0];
    let n = a.shape()[1];
    let k = m.min(n);
    let s_vals: Vec<f64> = s.data().midpoints().to_vec();
    let tol = f64::EPSILON * m.max(n) as f64 * s_vals[0].max(f64::MIN_POSITIVE);

    // s_inv[k, k] diagonal
    let mut s_inv = vec![0.0f64; k * k];
    for i in 0..k {
        if s_vals[i] > tol {
            s_inv[row_major(k, k, i, i)] = 1.0 / s_vals[i];
        }
    }
    // pinv = V S^+ U^T
    let v_arr = vt.transpose();
    let vs = crate::ops::reduction::matmul(
        &v_arr,
        &IntervalArray::from_raw_parts(&s_inv, &vec![0.0; k * k], &[k, k]),
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
                if apq.abs() <= eps * (m[row_major(n, n, p, p)].abs() + m[row_major(n, n, q, q)].abs()) {
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
    order.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut vals_sorted = vec![0.0f64; n];
    let mut vecs = vec![0.0f64; n * n];
    for (k, &j) in order.iter().enumerate() {
        vals_sorted[k] = vals[j];
        for i in 0..n {
            vecs[row_major(n, n, i, k)] = v[row_major(n, n, i, j)];
        }
    }

    let evals = IntervalArray::from_raw_parts(&vals_sorted, &vec![0.0; n], &[n]);
    let evecs = IntervalArray::from_raw_parts(&vecs, &vec![0.0; n * n], &[n, n]);
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

        // One QR sweep: Givens rotations applied to both sides.
        for col in l..q.saturating_sub(1) {
            let x = h[row_major(n, n, col, col)];
            let y = h[row_major(n, n, col + 1, col)];
            let hyp = (x * x + y * y).sqrt();
            if hyp == 0.0 {
                continue;
            }
            let c = x / hyp;
            let s = y / hyp;
            // rows col, col+1 (left multiply)
            for j in l..q {
                let t1 = h[row_major(n, n, col, j)];
                let t2 = h[row_major(n, n, col + 1, j)];
                h[row_major(n, n, col, j)] = c * t1 + s * t2;
                h[row_major(n, n, col + 1, j)] = -s * t1 + c * t2;
            }
            // columns col, col+1 (right multiply)
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

    let vals: Vec<f64> = (0..n).map(|i| h[row_major(n, n, i, i)]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut vals_sorted = vec![0.0f64; n];
    let mut vecs = vec![0.0f64; n * n];
    for (k, &j) in order.iter().enumerate() {
        vals_sorted[k] = vals[j];
        for i in 0..n {
            vecs[row_major(n, n, i, k)] = v[row_major(n, n, i, j)];
        }
    }

    let evals = IntervalArray::from_raw_parts(&vals_sorted, &vec![0.0; n], &[n]);
    let evecs = IntervalArray::from_raw_parts(&vecs, &vec![0.0; n * n], &[n, n]);
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
}
