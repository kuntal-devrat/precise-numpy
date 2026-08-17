//! Deterministic, seedable pseudo-random number generation.
//!
//! Uses the xoshiro256** generator (public domain, Blackman & Vigna).
//! No external dependencies are required.

use std::sync::Mutex;

use crate::array::IntervalArray;

struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        // SplitMix64 to spread the seed into the state.
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform double in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

static RNG: Mutex<Option<Xoshiro256>> = Mutex::new(None);

fn with_rng<F: FnOnce(&mut Xoshiro256) -> R, R>(f: F) -> R {
    let mut guard = RNG.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(Xoshiro256::new(0x9E37_79B9_7F4A_7C15));
    }
    f(guard.as_mut().unwrap())
}

/// Seed the global random number generator.
pub fn seed(seed: u64) {
    let mut guard = RNG.lock().unwrap_or_else(|p| p.into_inner());
    *guard = Some(Xoshiro256::new(seed));
}

/// One uniform double in [0, 1).
pub fn random_f64() -> f64 {
    with_rng(|r| r.next_f64())
}

/// Standard normal double.
pub fn random_normal() -> f64 {
    with_rng(|r| r.next_normal())
}

/// Uniform double in [low, high).
pub fn random_uniform(low: f64, high: f64) -> f64 {
    with_rng(|r| low + (high - low) * r.next_f64())
}

/// Integer in [low, high).
pub fn random_int(low: i64, high: i64) -> i64 {
    let span = (high - low).max(1) as u64;
    with_rng(|r| low + (r.next_u64() % span) as i64)
}

fn shape_of(size: Option<Vec<usize>>) -> Vec<usize> {
    size.unwrap_or_else(|| vec![1])
}

fn to_array(vals: Vec<f64>, shape: &[usize]) -> IntervalArray {
    let rads = vec![0.0f64; vals.len()];
    IntervalArray::from_raw_parts(&vals, &rads, shape)
}

/// Array of uniform doubles in [0, 1).
pub fn rand_array(size: &[usize]) -> IntervalArray {
    let n: usize = size.iter().product();
    let vals: Vec<f64> = (0..n).map(|_| random_f64()).collect();
    to_array(vals, size)
}

/// Array of standard normal doubles.
pub fn randn_array(size: &[usize]) -> IntervalArray {
    let n: usize = size.iter().product();
    let vals: Vec<f64> = (0..n).map(|_| random_normal()).collect();
    to_array(vals, size)
}

/// Array of uniform doubles in [low, high).
pub fn uniform_array(low: f64, high: f64, size: &[usize]) -> IntervalArray {
    let n: usize = size.iter().product();
    let vals: Vec<f64> = (0..n).map(|_| random_uniform(low, high)).collect();
    to_array(vals, size)
}

/// Array of normal doubles with loc/scale.
pub fn normal_array(loc: f64, scale: f64, size: &[usize]) -> IntervalArray {
    let n: usize = size.iter().product();
    let vals: Vec<f64> = (0..n).map(|_| loc + scale * random_normal()).collect();
    to_array(vals, size)
}

/// Array of integers in [low, high).
pub fn randint_array(low: i64, high: i64, size: &[usize]) -> IntervalArray {
    let n: usize = size.iter().product();
    let vals: Vec<f64> = (0..n).map(|_| random_int(low, high) as f64).collect();
    to_array(vals, size)
}

pub fn shape_of_scalar() -> Vec<usize> {
    vec![1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_reproducibility() {
        let mut guard = RNG.lock().unwrap();
        *guard = Some(Xoshiro256::new(42));
        let a = guard.as_mut().unwrap().next_f64();
        *guard = Some(Xoshiro256::new(42));
        let b = guard.as_mut().unwrap().next_f64();
        assert_eq!(a, b);
    }

    #[test]
    fn test_range() {
        seed(1);
        for _ in 0..1000 {
            let x = random_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn test_int_range() {
        seed(1);
        for _ in 0..1000 {
            let x = random_int(0, 10);
            assert!((0..10).contains(&x));
        }
    }

    #[test]
    fn test_array() {
        seed(7);
        let a = rand_array(&[3, 2]);
        assert_eq!(a.shape(), &[3, 2]);
        assert!(a.is_exact());
    }
}
