use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use pyo3::Bound;

mod array;
mod error;
mod ops;
mod parallel;
mod simd;

use array::IntervalArray;
use error::Interval;
use ops::{arithmetic, math, reduction};

/// A NumPy-compatible interval array with guaranteed numerical error bounds.
#[pyclass(name = "IntervalArray", frozen)]
struct PyIntervalArray {
    inner: IntervalArray,
}

#[pymethods]
impl PyIntervalArray {
    #[new]
    #[pyo3(signature = (values, error=0.0))]
    fn new(values: &Bound<'_, PyList>, error: f64) -> PyResult<Self> {
        let n = values.len();
        if n == 0 {
            return Ok(Self {
                inner: IntervalArray::zeros(&[0]),
            });
        }
        let mut mids = Vec::with_capacity(n);
        for i in 0..n {
            let v: f64 = values.get_item(i)?.extract()?;
            mids.push(v);
        }
        if error == 0.0 {
            Ok(Self {
                inner: IntervalArray::from_f64_slice(&mids),
            })
        } else {
            let rads = vec![error; n];
            Ok(Self {
                inner: IntervalArray::from_raw_parts(&mids, &rads, &[n]),
            })
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn max_relative_error(&self) -> f64 {
        self.inner.max_relative_error()
    }

    fn max_radius(&self) -> f64 {
        self.inner.max_radius()
    }

    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn get(&self, idx: usize) -> PyResult<(f64, f64)> {
        if idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("index {} out of range for array of length {}", idx, self.inner.len())
            ));
        }
        let iv = self.inner.get(idx);
        Ok((iv.midpoint(), iv.radius()))
    }

    fn midpoint(&self, idx: usize) -> PyResult<f64> {
        if idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
        }
        Ok(self.inner.get(idx).midpoint())
    }

    fn radius(&self, idx: usize) -> PyResult<f64> {
        if idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
        }
        Ok(self.inner.get(idx).radius())
    }

    fn __repr__(&self) -> String {
        let n = self.inner.len();
        if n <= 8 {
            let items: Vec<String> = (0..n)
                .map(|i| {
                    let iv = self.inner.get(i);
                    if iv.is_exact() {
                        format!("{}", iv.lo)
                    } else {
                        format!("{}+/-{}", iv.midpoint(), iv.radius())
                    }
                })
                .collect();
            format!("IntervalArray([{}])", items.join(", "))
        } else {
            format!(
                "IntervalArray(shape={}, max_err={:.6e})",
                format!("{:?}", self.inner.shape()),
                self.inner.max_relative_error()
            )
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    // ── Indexing ──

    fn __getitem__(&self, idx: isize) -> PyResult<(f64, f64)> {
        let n = self.inner.len() as isize;
        let actual = if idx < 0 { n + idx } else { idx };
        if actual < 0 || actual >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
        }
        let iv = self.inner.get(actual as usize);
        Ok((iv.midpoint(), iv.radius()))
    }

    // ── Arithmetic operators ──

    fn __add__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyIntervalArray>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || arithmetic::add_arrays(&a, &b));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || {
                arithmetic::add_scalar(&a, Interval::exact(scalar))
            });
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported operand type for +",
            ))
        }
    }

    fn __radd__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyIntervalArray>> {
        Self::__add__(slf, other)
    }

    fn __sub__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyIntervalArray>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || arithmetic::sub_arrays(&a, &b));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || {
                arithmetic::add_scalar(&a, Interval::exact(-scalar))
            });
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported operand type for -",
            ))
        }
    }

    fn __rsub__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyIntervalArray>> {
        let py = slf.py();
        if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || {
                let neg = arithmetic::neg_array(&a);
                arithmetic::add_scalar(&neg, Interval::exact(scalar))
            });
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported operand type for -",
            ))
        }
    }

    fn __mul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyIntervalArray>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || arithmetic::mul_arrays(&a, &b));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || arithmetic::scale_array(&a, scalar));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported operand type for *",
            ))
        }
    }

    fn __rmul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyIntervalArray>> {
        Self::__mul__(slf, other)
    }

    fn __truediv__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyIntervalArray>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || arithmetic::div_arrays(&a, &b));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || arithmetic::scale_array(&a, 1.0 / scalar));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported operand type for /",
            ))
        }
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: arithmetic::neg_array(&self.inner),
        }
    }

    fn __abs__(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::abs_interval),
        }
    }

    // ── Math functions ──

    fn sin(&self) -> Self {
        if self.inner.is_exact() {
            Self {
                inner: math::sin_batch_exact(&self.inner),
            }
        } else {
            Self {
                inner: math::apply_unary(&self.inner, math::sin_interval),
            }
        }
    }

    fn cos(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::cos_interval),
        }
    }

    fn tan(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::tan_interval),
        }
    }

    fn exp(&self) -> Self {
        if self.inner.is_exact() {
            Self {
                inner: math::exp_batch_exact(&self.inner),
            }
        } else {
            Self {
                inner: math::apply_unary(&self.inner, math::exp_interval),
            }
        }
    }

    fn ln(&self) -> Self {
        if self.inner.is_exact() {
            Self {
                inner: math::ln_batch_exact(&self.inner),
            }
        } else {
            Self {
                inner: math::apply_unary(&self.inner, math::ln_interval),
            }
        }
    }

    fn log2(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::log2_interval),
        }
    }

    fn log10(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::log10_interval),
        }
    }

    fn sqrt(&self) -> Self {
        if self.inner.is_exact() {
            Self {
                inner: math::sqrt_batch_exact(&self.inner),
            }
        } else {
            Self {
                inner: math::apply_unary(&self.inner, math::sqrt_interval),
            }
        }
    }

    fn abs(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::abs_interval),
        }
    }

    // ── Reductions ──

    fn sum(&self) -> (f64, f64) {
        let iv = reduction::sum(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    fn mean(&self) -> (f64, f64) {
        let iv = reduction::mean(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    fn var(&self) -> (f64, f64) {
        let iv = reduction::var(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    fn std(&self) -> (f64, f64) {
        let iv = reduction::std_dev(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    fn min_val(&self) -> (f64, f64) {
        let iv = reduction::min(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    fn max_val(&self) -> (f64, f64) {
        let iv = reduction::max(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    fn dot(&self, other: &Self) -> (f64, f64) {
        let iv = reduction::dot(&self.inner, &other.inner);
        (iv.midpoint(), iv.radius())
    }

    fn matmul(&self, other: &Self) -> Self {
        Self {
            inner: reduction::matmul(&self.inner, &other.inner),
        }
    }

    fn cumsum(&self) -> Self {
        Self {
            inner: reduction::cumsum(&self.inner),
        }
    }

    fn prod(&self) -> (f64, f64) {
        let iv = reduction::prod(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    fn norm(&self) -> (f64, f64) {
        let iv = reduction::norm_l2(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    // ── Conversion helpers ──

    fn tolist(&self) -> Vec<(f64, f64)> {
        let n = self.inner.len();
        let mids = self.inner.data().midpoints();
        let rads = self.inner.data().radii();
        (0..n).map(|i| (mids[i], rads[i])).collect()
    }

    fn values(&self) -> Vec<f64> {
        self.inner.data().midpoints().to_vec()
    }

    fn radii(&self) -> Vec<f64> {
        self.inner.data().radii().to_vec()
    }

    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.reshape(&shape),
        })
    }

    fn transpose(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.transpose(),
        })
    }

    /// Scale the array by an exact scalar (fast path).
    fn scale(&self, scalar: f64) -> Self {
        Self {
            inner: arithmetic::scale_array(&self.inner, scalar),
        }
    }
}

// ── Module-level functions ──

#[pyfunction(signature = (values, error=0.0))]
#[pyo3(name = "array")]
fn parray(values: &Bound<'_, PyList>, error: f64) -> PyResult<PyIntervalArray> {
    PyIntervalArray::new(values, error)
}

#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyResult<PyIntervalArray> {
    Ok(PyIntervalArray {
        inner: IntervalArray::zeros(&shape),
    })
}

#[pyfunction]
fn ones(shape: Vec<usize>) -> PyResult<PyIntervalArray> {
    Ok(PyIntervalArray {
        inner: IntervalArray::ones(&shape),
    })
}

#[pyfunction]
#[pyo3(signature = (shape, value, error=0.0))]
fn full(shape: Vec<usize>, value: f64, error: f64) -> PyResult<PyIntervalArray> {
    Ok(PyIntervalArray {
        inner: IntervalArray::full(&shape, Interval::from_midpoint_radius(value, error)),
    })
}

#[pyfunction]
#[pyo3(signature = (start, stop, num=50))]
fn linspace(start: f64, stop: f64, num: usize) -> PyResult<PyIntervalArray> {
    Ok(PyIntervalArray {
        inner: IntervalArray::linspace(start, stop, num),
    })
}

#[pyfunction]
#[pyo3(signature = (start, stop, step=1.0))]
fn arange(start: f64, stop: f64, step: f64) -> PyResult<PyIntervalArray> {
    Ok(PyIntervalArray {
        inner: IntervalArray::arange(start, stop, step),
    })
}

/// Get the number of Rayon worker threads.
#[pyfunction]
fn num_threads() -> usize {
    parallel::num_threads()
}

/// Python module definition.
#[pymodule]
fn _precise_numpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIntervalArray>()?;
    m.add_function(wrap_pyfunction!(parray, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
