//! Boolean array used as the result of interval comparisons.

use std::sync::Arc;

use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3::Bound;

/// A NumPy-style boolean array (mask).
#[pyclass(name = "BoolArray", frozen)]
pub struct PyBoolArray {
    pub(crate) inner: Arc<Vec<bool>>,
}

impl PyBoolArray {
    pub fn new(values: Vec<bool>) -> Self {
        Self {
            inner: Arc::new(values),
        }
    }

    pub fn to_vec(&self) -> Vec<bool> {
        self.inner.as_ref().clone()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    fn index_range(&self, idx: isize) -> Option<usize> {
        let n = self.inner.len() as isize;
        let i = if idx < 0 { n + idx } else { idx };
        if i < 0 || i >= n {
            None
        } else {
            Some(i as usize)
        }
    }
}

fn slice_to_range(slice: &Bound<'_, PySlice>, len: usize) -> Option<Vec<usize>> {
    let indices = slice.indices(len as isize).ok()?;
    let (mut start, stop, step) = (indices.start, indices.stop, indices.step);
    let mut out = Vec::new();
    if step > 0 {
        while start < stop {
            out.push(start as usize);
            start += step;
        }
    } else {
        while start > stop {
            out.push(start as usize);
            start += step;
        }
    }
    Some(out)
}

#[pymethods]
impl PyBoolArray {
    #[new]
    fn new_py(values: Vec<bool>) -> Self {
        Self::new(values)
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __bool__(&self) -> PyResult<bool> {
        if self.inner.len() == 1 {
            Ok(self.inner[0])
        } else {
            Err(PyValueError::new_err(
                "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()",
            ))
        }
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = index.py();
        if let Ok(i) = index.extract::<isize>() {
            match self.index_range(i) {
                Some(k) => Ok(self.inner[k].to_object(py).into_any()),
                None => Err(PyIndexError::new_err("boolean index out of range")),
            }
        } else if let Ok(sl) = index.downcast::<PySlice>() {
            let idxs = slice_to_range(sl, self.inner.len())
                .ok_or_else(|| PyValueError::new_err("invalid slice"))?;
            let vals: Vec<bool> = idxs.iter().map(|&k| self.inner[k]).collect();
            Ok(Py::new(py, PyBoolArray::new(vals))?.into_any())
        } else {
            Err(PyTypeError::new_err(
                "only integers and slices are valid indices for a BoolArray",
            ))
        }
    }

    fn tolist(&self) -> Vec<bool> {
        self.inner.as_ref().clone()
    }

    fn any(&self) -> bool {
        self.inner.iter().any(|&b| b)
    }

    fn all(&self) -> bool {
        self.inner.iter().all(|&b| b)
    }

    /// Number of True entries.
    fn sum(&self) -> usize {
        self.inner.iter().filter(|&&b| b).count()
    }

    fn count_nonzero(&self) -> usize {
        self.inner.iter().filter(|&&b| b).count()
    }

    fn __repr__(&self) -> String {
        let items: Vec<String> = self.inner.iter().map(|&b| b.to_string()).collect();
        format!("BoolArray([{}])", items.join(", "))
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    // ── Logical operators ──

    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBoolArray> {
        if let Ok(o) = other.downcast::<PyBoolArray>() {
            if o.borrow().len() != self.len() {
                return Err(PyValueError::new_err(
                    "operands could not be broadcast together (boolean arrays must match)",
                ));
            }
            let vals: Vec<bool> = self
                .inner
                .iter()
                .zip(o.borrow().inner.iter())
                .map(|(&a, &b)| a && b)
                .collect();
            return Ok(PyBoolArray::new(vals));
        }
        if let Ok(b) = other.extract::<bool>() {
            let vals: Vec<bool> = self.inner.iter().map(|&a| a && b).collect();
            return Ok(PyBoolArray::new(vals));
        }
        Err(PyTypeError::new_err("unsupported operand type for &"))
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBoolArray> {
        if let Ok(o) = other.downcast::<PyBoolArray>() {
            if o.borrow().len() != self.len() {
                return Err(PyValueError::new_err(
                    "operands could not be broadcast together (boolean arrays must match)",
                ));
            }
            let vals: Vec<bool> = self
                .inner
                .iter()
                .zip(o.borrow().inner.iter())
                .map(|(&a, &b)| a || b)
                .collect();
            return Ok(PyBoolArray::new(vals));
        }
        if let Ok(b) = other.extract::<bool>() {
            let vals: Vec<bool> = self.inner.iter().map(|&a| a || b).collect();
            return Ok(PyBoolArray::new(vals));
        }
        Err(PyTypeError::new_err("unsupported operand type for |"))
    }

    fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBoolArray> {
        if let Ok(o) = other.downcast::<PyBoolArray>() {
            if o.borrow().len() != self.len() {
                return Err(PyValueError::new_err(
                    "operands could not be broadcast together (boolean arrays must match)",
                ));
            }
            let vals: Vec<bool> = self
                .inner
                .iter()
                .zip(o.borrow().inner.iter())
                .map(|(&a, &b)| a ^ b)
                .collect();
            return Ok(PyBoolArray::new(vals));
        }
        if let Ok(b) = other.extract::<bool>() {
            let vals: Vec<bool> = self.inner.iter().map(|&a| a ^ b).collect();
            return Ok(PyBoolArray::new(vals));
        }
        Err(PyTypeError::new_err("unsupported operand type for ^"))
    }

    fn __invert__(&self) -> PyBoolArray {
        let vals: Vec<bool> = self.inner.iter().map(|&a| !a).collect();
        PyBoolArray::new(vals)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_none() {
            return Ok(false);
        }
        if let Ok(o) = other.downcast::<PyBoolArray>() {
            if o.borrow().len() != self.len() {
                return Ok(false);
            }
            return Ok(self.inner.iter().zip(o.borrow().inner.iter()).all(|(&a, &b)| a == b));
        }
        if let Ok(b) = other.extract::<bool>() {
            if self.len() != 1 {
                return Ok(false);
            }
            return Ok(self.inner[0] == b);
        }
        Ok(false)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(!self.__eq__(other)?)
    }
}
