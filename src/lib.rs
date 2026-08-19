#![allow(dead_code, unused_variables)]

use pyo3::exceptions::{
    PyIndexError, PyNotImplementedError, PyRuntimeWarning, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyIterator, PyList, PyModule, PySlice, PyTuple};
use pyo3::Bound;

mod array;
mod bool_array;
mod error;
mod ops;
mod parallel;
mod simd;

use array::IntervalArray;
use bool_array::PyBoolArray;
use error::Interval;
use ops::{arithmetic, broadcast, compare, extra, linalg, math, random, reduction};

// ══════════════════════════════════════════════════════════════════════
// IntervalArray pyclass
// ══════════════════════════════════════════════════════════════════════

/// A NumPy-compatible interval array with guaranteed numerical error bounds.
#[pyclass(name = "IntervalArray", module = "precise_numpy._precise_numpy")]
struct PyIntervalArray {
    inner: IntervalArray,
}

// ── Shared helpers ─────────────────────────────────────────────────────

/// Build a nested Python list matching `shape` from a flat slice of values
/// (numpy `tolist()` semantics).
fn build_nested_lists<'py>(
    py: Python<'py>,
    flat: &[(f64, f64)],
    shape: &[usize],
    offset: &mut usize,
) -> PyResult<Py<PyAny>> {
    if shape.len() <= 1 {
        let items: Vec<Py<PyAny>> = (0..shape.first().copied().unwrap_or(0))
            .map(|_| {
                let (m, r) = flat[*offset];
                *offset += 1;
                PyTuple::new_bound(py, [m, r]).unbind().into_any()
            })
            .collect();
        return Ok(PyList::new_bound(py, items).into_any().unbind());
    }
    let mut items: Vec<Py<PyAny>> = Vec::with_capacity(shape[0]);
    for _ in 0..shape[0] {
        items.push(build_nested_lists(py, flat, &shape[1..], offset)?);
    }
    Ok(PyList::new_bound(py, items).into_any().unbind())
}

fn warn_div_zero(py: Python<'_>) -> PyResult<()> {
    let category = py.get_type_bound::<PyRuntimeWarning>();
    PyErr::warn_bound(
        py,
        category.as_any(),
        "divide by zero encountered in interval division; returning the entire real line",
        2,
    )
}

/// `s / 0` returns the entire real line and signals the caller to warn.
fn scalar_div_array(a: &IntervalArray, s: f64) -> (IntervalArray, bool) {
    if s == 0.0 {
        let n = a.len();
        let mids = vec![0.0f64; n];
        let rads = vec![f64::INFINITY; n];
        return (IntervalArray::from_raw_parts(&mids, &rads, a.shape()), true);
    }
    (arithmetic::scale_array(a, 1.0 / s), false)
}

/// Broadcast two arrays to a common shape, raising on incompatible shapes.
fn broadcast_for_op(
    a: &IntervalArray,
    b: &IntervalArray,
) -> PyResult<(IntervalArray, IntervalArray)> {
    let shape = broadcast::broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "operands could not be broadcast together with shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ))
    })?;
    Ok((
        broadcast::broadcast_to(a, &shape),
        broadcast::broadcast_to(b, &shape),
    ))
}

fn add_dispatch(a: &IntervalArray, b: &IntervalArray) -> PyResult<IntervalArray> {
    let (ba, bb) = broadcast_for_op(a, b)?;
    Ok(arithmetic::add_arrays(&ba, &bb))
}

fn sub_dispatch(a: &IntervalArray, b: &IntervalArray) -> PyResult<IntervalArray> {
    let (ba, bb) = broadcast_for_op(a, b)?;
    Ok(arithmetic::sub_arrays(&ba, &bb))
}

fn mul_dispatch(a: &IntervalArray, b: &IntervalArray) -> PyResult<IntervalArray> {
    let (ba, bb) = broadcast_for_op(a, b)?;
    Ok(arithmetic::mul_arrays(&ba, &bb))
}

/// Element-wise interval division; the bool signals whether any divisor
/// interval contains zero (caller is expected to emit a warning).
fn div_dispatch(a: &IntervalArray, b: &IntervalArray) -> PyResult<(IntervalArray, bool)> {
    let (ba, bb) = broadcast_for_op(a, b)?;
    Ok(arithmetic::div_arrays(&ba, &bb))
}

fn pow_dispatch(a: &IntervalArray, b: &IntervalArray) -> PyResult<IntervalArray> {
    let (ba, bb) = broadcast_for_op(a, b)?;
    Ok(extra::pow_arrays(&ba, &bb))
}

fn compare_dispatch(
    a: &IntervalArray,
    b: &IntervalArray,
    op: compare::Cmp,
) -> PyResult<Vec<bool>> {
    let (ba, bb) = broadcast_for_op(a, b)?;
    Ok(compare::compare_arrays(&ba, &bb, op))
}

/// Apply an in-place binary op to `slf`, writing results back into its buffer.
fn apply_inplace(
    py: Python<'_>,
    slf: &mut IntervalArray,
    other: &Bound<'_, PyAny>,
    f_array: impl Fn(Python<'_>, &IntervalArray, &IntervalArray) -> PyResult<IntervalArray>,
    f_scalar: impl Fn(Python<'_>, &IntervalArray, f64) -> PyResult<IntervalArray>,
) -> PyResult<()> {
    let a = slf.clone();
    let result = if let Ok(o) = other.downcast::<PyIntervalArray>() {
        let b = o.borrow().inner.clone();
        let b_shape = broadcast::broadcast_shapes(slf.shape(), b.shape()).ok_or_else(|| {
            PyValueError::new_err(format!(
                "operands could not be broadcast together with shapes {:?} and {:?}",
                slf.shape(),
                b.shape()
            ))
        })?;
        if b_shape != slf.shape() {
            return Err(PyValueError::new_err(format!(
                "output array with shape {:?} is not the result of broadcasting operand shapes {:?} and {:?}",
                slf.shape(),
                slf.shape(),
                b.shape()
            )));
        }
        let bb = broadcast::broadcast_to(&b, slf.shape());
        f_array(py, &a, &bb)?
    } else if let Ok(s) = other.extract::<f64>() {
        f_scalar(py, &a, s)?
    } else {
        return Err(PyTypeError::new_err(
            "unsupported operand type for in-place operation",
        ));
    };
    let (rm, rr) = slf.data_mut().as_mut_slices();
    rm.copy_from_slice(result.data().midpoints());
    rr.copy_from_slice(result.data().radii());
    Ok(())
}

// ── Indexing helpers ───────────────────────────────────────────────────

fn resolve_slice(
    start: Option<isize>,
    stop: Option<isize>,
    step: Option<isize>,
    dim: usize,
) -> PyResult<Vec<usize>> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(PyValueError::new_err("slice step cannot be zero"));
    }
    let n = dim as isize;
    let (s, e) = if step > 0 {
        let s = match start {
            Some(v) => {
                if v < 0 {
                    (v + n).max(0)
                } else {
                    v.min(n)
                }
            }
            None => 0,
        };
        let e = match stop {
            Some(v) => {
                if v < 0 {
                    (v + n).max(0)
                } else {
                    v.min(n)
                }
            }
            None => n,
        };
        (s, e)
    } else {
        let s = match start {
            Some(v) => {
                if v < 0 {
                    (v + n).max(-1)
                } else {
                    v.min(n - 1)
                }
            }
            None => n - 1,
        };
        let e = match stop {
            Some(v) => {
                if v < 0 {
                    (v + n).max(-1)
                } else {
                    v.min(n - 1)
                }
            }
            None => -1,
        };
        (s, e)
    };
    let mut out = Vec::new();
    let mut cur = s;
    if step > 0 {
        while cur < e {
            out.push(cur as usize);
            cur += step;
        }
    } else {
        while cur > e {
            out.push(cur as usize);
            cur += step;
        }
    }
    Ok(out)
}

enum IndexComp {
    /// Selection along the next original axis (in positional order).
    Axis { idxs: Vec<usize>, is_int: bool },
    /// A new axis of length 1 inserted at this position (`None`).
    NewAxis,
}

struct ParsedIndex {
    /// Index components in positional order.
    comps: Vec<IndexComp>,
}

impl ParsedIndex {
    /// The Axis components in order (their position within the Vec is the
    /// ordinal of the original axis they index).
    fn axis_comps(&self) -> impl Iterator<Item = &IndexComp> {
        self.comps.iter().filter(|c| matches!(c, IndexComp::Axis { .. }))
    }

    fn scalar(&self) -> bool {
        self.comps.iter().all(|c| {
            matches!(c, IndexComp::Axis { is_int: true, .. })
        })
    }
}

fn parse_index(slf: &IntervalArray, index: &Bound<'_, PyAny>) -> PyResult<ParsedIndex> {
    let ndim = slf.ndim();
    let shape = slf.shape().to_vec();

    let mut specs: Vec<Bound<'_, PyAny>> = Vec::new();
    if let Ok(tup) = index.downcast::<PyTuple>() {
        for i in 0..tup.len() {
            specs.push(tup.get_item(i)?);
        }
    } else {
        specs.push(index.clone());
    }

    let mut num_real = 0usize;
    let mut num_ellipsis = 0usize;
    for spec in specs.iter() {
        if spec.is_none() {
            // no axis consumed
        } else if spec.is_ellipsis() {
            num_ellipsis += 1;
        } else {
            num_real += 1;
        }
    }
    if num_ellipsis > 1 {
        return Err(PyIndexError::new_err(
            "an index can only have a single ellipsis ('...')",
        ));
    }
    if num_real > ndim {
        return Err(PyIndexError::new_err(format!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            ndim, num_real
        )));
    }
    let fill = ndim - num_real;

    let mut comps: Vec<IndexComp> = Vec::new();
    let mut k = 0usize;

    for spec in specs.iter() {
        if spec.is_none() {
            comps.push(IndexComp::NewAxis);
        } else if spec.is_ellipsis() {
            for _ in 0..fill {
                comps.push(IndexComp::Axis {
                    idxs: (0..shape[k]).collect(),
                    is_int: false,
                });
                k += 1;
            }
        } else if let Ok(i) = spec.extract::<isize>() {
            let n = shape[k] as isize;
            let a = if i < 0 { n + i } else { i };
            if a < 0 || a >= n {
                return Err(PyIndexError::new_err(format!(
                    "index {} is out of bounds for axis {} with size {}",
                    i, k, shape[k]
                )));
            }
            comps.push(IndexComp::Axis {
                idxs: vec![a as usize],
                is_int: true,
            });
            k += 1;
        } else if let Ok(sl) = spec.downcast::<PySlice>() {
            let start = sl.getattr("start")?.extract::<Option<isize>>()?;
            let stop = sl.getattr("stop")?.extract::<Option<isize>>()?;
            let step = sl.getattr("step")?.extract::<Option<isize>>()?;
            comps.push(IndexComp::Axis {
                idxs: resolve_slice(start, stop, step, shape[k])?,
                is_int: false,
            });
            k += 1;
        } else if spec.is_instance_of::<PyBoolArray>() {
            let mask = spec.downcast::<PyBoolArray>()?;
            if mask.borrow().len() != shape[k] {
                return Err(PyIndexError::new_err(format!(
                    "boolean index did not match indexed array along dimension {}; dimension is {} but corresponding boolean dimension is {}",
                    k,
                    shape[k],
                    mask.borrow().len()
                )));
            }
            let idxs: Vec<usize> = mask
                .borrow()
                .to_vec()
                .iter()
                .enumerate()
                .filter(|(_, &b)| b)
                .map(|(i, _)| i)
                .collect();
            comps.push(IndexComp::Axis {
                idxs,
                is_int: false,
            });
            k += 1;
        } else if let Ok(list) = spec.downcast::<PyList>() {
            let n = list.len();
            let first_is_bool = n > 0 && list.get_item(0)?.is_instance_of::<PyBool>();
            if first_is_bool {
                if n != shape[k] {
                    return Err(PyIndexError::new_err(format!(
                        "boolean index did not match indexed array along dimension {}; dimension is {} but corresponding boolean dimension is {}",
                        k,
                        shape[k],
                        n
                    )));
                }
                let mut idxs = Vec::new();
                for i in 0..n {
                    let b: bool = list.get_item(i)?.extract()?;
                    if b {
                        idxs.push(i);
                    }
                }
                comps.push(IndexComp::Axis {
                    idxs,
                    is_int: false,
                });
            } else {
                let mut idxs = Vec::new();
                for i in 0..n {
                    let v: isize = list.get_item(i)?.extract()?;
                    let a = if v < 0 { shape[k] as isize + v } else { v };
                    if a < 0 || a >= shape[k] as isize {
                        return Err(PyIndexError::new_err(format!(
                            "index {} is out of bounds for axis {} with size {}",
                            v, k, shape[k]
                        )));
                    }
                    idxs.push(a as usize);
                }
                comps.push(IndexComp::Axis {
                    idxs,
                    is_int: false,
                });
            }
            k += 1;
        } else {
            let ty = spec
                .get_type()
                .name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "unknown type".to_string());
            return Err(PyTypeError::new_err(format!(
                "only integers, slices, lists, and boolean masks are valid indices; got {}",
                ty
            )));
        }
    }

    while k < ndim {
        comps.push(IndexComp::Axis {
            idxs: (0..shape[k]).collect(),
            is_int: false,
        });
        k += 1;
    }

    Ok(ParsedIndex { comps })
}

fn apply_index_get(
    py: Python<'_>,
    slf: &IntervalArray,
    parsed: &ParsedIndex,
) -> PyResult<Py<PyAny>> {
    let ndim = slf.ndim();
    let strides = slf.strides();
    let scalar = parsed.scalar();
    if scalar {
        let mut flat = 0usize;
        for (k, comp) in parsed.axis_comps().enumerate() {
            if let IndexComp::Axis { idxs, .. } = comp {
                flat += idxs[0] * strides[k];
            }
        }
        let iv = slf.get(flat);
        return Ok((iv.midpoint(), iv.radius()).to_object(py).into_any());
    }

    // Shape of the result: NewAxis contributes a length-1 dimension, a
    // non-integer Axis contributes its selection length, and an integer
    // Axis is consumed (its dimension is dropped).
    let mut out_shape = Vec::new();
    for comp in &parsed.comps {
        match comp {
            IndexComp::NewAxis => out_shape.push(1),
            IndexComp::Axis { idxs, is_int } => {
                if !is_int {
                    out_shape.push(idxs.len());
                }
            }
        }
    }
    let out_total: usize = out_shape.iter().product();
    let mut out_mids = vec![0.0f64; out_total];
    let mut out_rads = vec![0.0f64; out_total];

    let axis_comps: Vec<(usize, &IndexComp)> = parsed.axis_comps().enumerate().collect();
    let mut cursor = vec![0usize; ndim];
    let mut cur_flat = 0isize;
    for (k, comp) in axis_comps.iter() {
        if let IndexComp::Axis { idxs, .. } = comp {
            cur_flat += idxs[0] as isize * strides[*k] as isize;
        }
    }
    let mids = slf.data().midpoints();
    let rads = slf.data().radii();
    for out_flat in 0..out_total {
        out_mids[out_flat] = mids[cur_flat as usize];
        out_rads[out_flat] = rads[cur_flat as usize];
        // Iterate axes in reverse so the last axis increments fastest
        // (row-major output order).
        for &(k, comp) in axis_comps.iter().rev() {
            let IndexComp::Axis { idxs, is_int } = comp else {
                unreachable!()
            };
            if *is_int {
                continue;
            }
            if cursor[k] + 1 < idxs.len() {
                let old = idxs[cursor[k]] as isize;
                cursor[k] += 1;
                let new = idxs[cursor[k]] as isize;
                cur_flat += (new - old) * strides[k] as isize;
                break;
            } else {
                cur_flat -= idxs[cursor[k]] as isize * strides[k] as isize;
                cursor[k] = 0;
                cur_flat += idxs[0] as isize * strides[k] as isize;
            }
        }
    }

    let arr = IntervalArray::from_raw_parts(&out_mids, &out_rads, &out_shape);
    Ok(Py::new(py, PyIntervalArray { inner: arr })?.into_any())
}

fn apply_index_set(
    slf: &mut IntervalArray,
    parsed: &ParsedIndex,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let ndim = slf.ndim();
    let strides = slf.strides().to_vec();
    let axis_comps: Vec<&IndexComp> = parsed.axis_comps().collect();

    let sizes: Vec<usize> = axis_comps
        .iter()
        .map(|c| match c {
            IndexComp::Axis { idxs, .. } => idxs.len(),
            IndexComp::NewAxis => unreachable!(),
        })
        .collect();
    let total: usize = sizes.iter().product();
    let mut positions: Vec<usize> = Vec::with_capacity(total);
    let mut cursor = vec![0usize; ndim];
    for _ in 0..total {
        let mut flat = 0usize;
        for (k, comp) in axis_comps.iter().enumerate() {
            if let IndexComp::Axis { idxs, .. } = comp {
                flat += idxs[cursor[k]] * strides[k];
            }
        }
        positions.push(flat);
        for k in (0..ndim).rev() {
            if cursor[k] + 1 < sizes[k] {
                cursor[k] += 1;
                break;
            }
            cursor[k] = 0;
        }
    }

    if positions.is_empty() {
        return Ok(());
    }

    if let Ok(v) = value.extract::<f64>() {
        for p in &positions {
            slf.set(*p, Interval::exact(v));
        }
    } else if let Ok((m, r)) = value.extract::<(f64, f64)>() {
        let iv = Interval::from_midpoint_radius(m, r);
        for p in &positions {
            slf.set(*p, iv);
        }
    } else if let Ok(arr) = value.downcast::<PyIntervalArray>() {
        let b = arr.borrow().inner.clone();
        if b.len() == 1 {
            let iv = b.get(0);
            for p in &positions {
                slf.set(*p, iv);
            }
        } else if b.len() == positions.len() {
            for (i, p) in positions.iter().enumerate() {
                slf.set(*p, b.get(i));
            }
        } else {
            return Err(PyValueError::new_err(format!(
                "cannot assign {} element(s) to a slice of {} element(s)",
                b.len(),
                positions.len()
            )));
        }
    } else if let Ok(vals) = value.extract::<Vec<f64>>() {
        if vals.len() == positions.len() {
            for (i, p) in positions.iter().enumerate() {
                slf.set(*p, Interval::exact(vals[i]));
            }
        } else {
            return Err(PyValueError::new_err(format!(
                "cannot assign {} element(s) to a slice of {} element(s)",
                vals.len(),
                positions.len()
            )));
        }
    } else if let Ok(vals) = value.extract::<Vec<(f64, f64)>>() {
        if vals.len() == positions.len() {
            for (i, p) in positions.iter().enumerate() {
                slf.set(*p, Interval::from_midpoint_radius(vals[i].0, vals[i].1));
            }
        } else {
            return Err(PyValueError::new_err(format!(
                "cannot assign {} element(s) to a slice of {} element(s)",
                vals.len(),
                positions.len()
            )));
        }
    } else {
        return Err(PyTypeError::new_err(
            "can only assign numbers, (midpoint, radius) tuples, lists, or IntervalArray values",
        ));
    }
    Ok(())
}

// ── Array creation helpers ─────────────────────────────────────────────

fn parse_scalar_leaf(obj: &Bound<'_, PyAny>, error: f64) -> PyResult<(f64, f64)> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok((v, error));
    }
    if let Ok((m, r)) = obj.extract::<(f64, f64)>() {
        return Ok((m, r));
    }
    Err(PyTypeError::new_err(format!(
        "cannot convert {} to an interval value",
        obj.get_type()
            .name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "unknown type".to_string())
    )))
}

/// Recursively parse nested lists/tuples into flat midpoints, radii, and shape.
fn parse_array_input(
    values: &Bound<'_, PyAny>,
    error: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    if values.is_instance_of::<PyList>() || values.is_instance_of::<PyTuple>() {
        let n = values.len()?;
        if n == 0 {
            return Ok((vec![], vec![], vec![0]));
        }
        let first = values.get_item(0)?;
        if first.is_instance_of::<PyList>() {
            let mut all_mids = Vec::new();
            let mut all_rads = Vec::new();
            let mut sub_shape: Option<Vec<usize>> = None;
            for i in 0..n {
                let item = values.get_item(i)?;
                if !item.is_instance_of::<PyList>() {
                    return Err(PyTypeError::new_err(
                        "setting an array element with a sequence: inhomogeneous shape",
                    ));
                }
                let (m, r, s) = parse_array_input(&item, error)?;
                if let Some(prev) = &sub_shape {
                    if *prev != s {
                        return Err(PyTypeError::new_err(
                            "setting an array element with a sequence: inhomogeneous shape",
                        ));
                    }
                } else {
                    sub_shape = Some(s);
                }
                all_mids.extend(m);
                all_rads.extend(r);
            }
            let mut shape = vec![n];
            shape.extend(sub_shape.unwrap());
            return Ok((all_mids, all_rads, shape));
        }
        if first.is_instance_of::<PyTuple>() {
            let mut mids = Vec::with_capacity(n);
            let mut rads = Vec::with_capacity(n);
            for i in 0..n {
                let item = values.get_item(i)?;
                if !item.is_instance_of::<PyTuple>() {
                    return Err(PyTypeError::new_err(
                        "cannot mix (midpoint, radius) tuples with plain numbers",
                    ));
                }
                let (m, r) = item
                    .extract::<(f64, f64)>()
                    .map_err(|_| PyTypeError::new_err("tuple elements must be numbers"))?;
                mids.push(m);
                rads.push(r);
            }
            return Ok((mids, rads, vec![n]));
        }
        let mut mids = Vec::with_capacity(n);
        let mut rads = Vec::with_capacity(n);
        for i in 0..n {
            let item = values.get_item(i)?;
            let (m, r) = parse_scalar_leaf(&item, error)?;
            mids.push(m);
            rads.push(r);
        }
        return Ok((mids, rads, vec![n]));
    }
    let (m, r) = parse_scalar_leaf(values, error)?;
    Ok((vec![m], vec![r], vec![1]))
}

fn extract_shape(shape: &Bound<'_, PyAny>, extra: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    let mut out = Vec::new();
    if let Ok(dims) = shape.extract::<Vec<usize>>() {
        out.extend(dims);
    } else if let Ok(d) = shape.extract::<usize>() {
        out.push(d);
    } else {
        return Err(PyTypeError::new_err(
            "shape must be an int or a sequence of ints",
        ));
    }
    for item in extra.iter() {
        out.push(item.extract::<usize>()?);
    }
    Ok(out)
}

#[pymethods]
impl PyIntervalArray {
    #[new]
    #[pyo3(signature = (values, error=0.0))]
    fn new(values: &Bound<'_, PyAny>, error: f64) -> PyResult<Self> {
        if error < 0.0 {
            return Err(PyValueError::new_err("error must be non-negative"));
        }
        if let Ok(other) = values.downcast::<PyIntervalArray>() {
            let inner = other.borrow().inner.clone();
            return Ok(Self { inner });
        }
        let (mids, rads, shape) = parse_array_input(values, error)?;
        Ok(Self {
            inner: IntervalArray::from_raw_parts(&mids, &rads, &shape),
        })
    }

    // ── Properties ──

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(py, self.inner.shape().to_vec())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        "interval64"
    }

    #[getter]
    fn itemsize(&self) -> usize {
        16
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.len() * 16
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(py, self.inner.strides().to_vec())
    }

    #[getter]
    fn t(&self) -> PyResult<Self> {
        self.transpose()
    }

    fn __len__(&self) -> usize {
        if self.inner.ndim() == 0 {
            0
        } else {
            self.inner.shape()[0]
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        if self.inner.len() == 1 {
            let iv = self.inner.get(0);
            return Ok(iv.midpoint() != 0.0 || iv.radius() != 0.0);
        }
        Err(PyValueError::new_err(
            "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()",
        ))
    }

    fn __repr__(&self) -> String {
        let shape = self.inner.shape();
        let n = self.inner.len();
        let ndim = self.inner.ndim();
        if ndim <= 2 && n <= 64 {
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
            if ndim == 2 {
                let cols = shape[1];
                let rows: Vec<String> = items
                    .chunks(cols)
                    .map(|row| format!("[{}]", row.join(", ")))
                    .collect();
                format!("IntervalArray([{}])", rows.join(", "))
            } else {
                format!("IntervalArray([{}])", items.join(", "))
            }
        } else {
            format!(
                "IntervalArray(shape={:?}, max_err={:.6e})",
                shape,
                self.inner.max_relative_error()
            )
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    // ── Indexing ──

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let parsed = parse_index(&self.inner, index)?;
        apply_index_get(py, &self.inner, &parsed)
    }

    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed = parse_index(&self.inner, index)?;
        apply_index_set(&mut self.inner, &parsed, value)
}

    fn __iter__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let ndim = slf.inner.ndim();
        let n = if ndim == 0 { 0 } else { slf.inner.shape()[0] };
        let mut items: Vec<Py<PyAny>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut comps = Vec::with_capacity(ndim);
            comps.push(IndexComp::Axis {
                idxs: vec![i],
                is_int: true,
            });
            for k in 1..ndim {
                comps.push(IndexComp::Axis {
                    idxs: (0..slf.inner.shape()[k]).collect(),
                    is_int: false,
                });
            }
            let parsed = ParsedIndex { comps };
            items.push(apply_index_get(py, &slf.inner, &parsed)?);
        }
        let list = PyList::new_bound(py, items);
        PyIterator::from_bound_object(&list)
    }

    fn get(&self, idx: usize) -> PyResult<(f64, f64)> {
        if idx >= self.inner.len() {
            return Err(PyIndexError::new_err(format!(
                "index {} out of range for array of length {}",
                idx,
                self.inner.len()
            )));
        }
        let iv = self.inner.get(idx);
        Ok((iv.midpoint(), iv.radius()))
    }

    fn item(&self, idx: usize) -> PyResult<(f64, f64)> {
        self.get(idx)
    }

    fn midpoint(&self, idx: usize) -> PyResult<f64> {
        if idx >= self.inner.len() {
            return Err(PyIndexError::new_err("index out of range"));
        }
        Ok(self.inner.get(idx).midpoint())
    }

    fn radius(&self, idx: usize) -> PyResult<f64> {
        if idx >= self.inner.len() {
            return Err(PyIndexError::new_err("index out of range"));
        }
        Ok(self.inner.get(idx).radius())
    }

    // ── Arithmetic operators ──

    fn __add__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || add_dispatch(&a, &b))?;
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result =
                py.allow_threads(move || arithmetic::add_scalar(&a, Interval::exact(scalar)));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(PyTypeError::new_err("unsupported operand type for +"))
        }
    }

    fn __radd__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::__add__(slf, other)
    }

    fn __sub__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || sub_dispatch(&a, &b))?;
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result =
                py.allow_threads(move || arithmetic::add_scalar(&a, Interval::exact(-scalar)));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(PyTypeError::new_err("unsupported operand type for -"))
        }
    }

    fn __rsub__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || {
                let neg = arithmetic::neg_array(&a);
                arithmetic::add_scalar(&neg, Interval::exact(scalar))
            });
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(PyTypeError::new_err("unsupported operand type for -"))
        }
    }

    fn __mul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || mul_dispatch(&a, &b))?;
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || arithmetic::scale_array(&a, scalar));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(PyTypeError::new_err("unsupported operand type for *"))
        }
    }

    fn __rmul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::__mul__(slf, other)
    }

    fn __truediv__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let (result, warned) = py.allow_threads(move || div_dispatch(&a, &b))?;
            if warned {
                warn_div_zero(py)?;
            }
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let (result, warned) = py.allow_threads(move || scalar_div_array(&a, scalar));
            if warned {
                warn_div_zero(py)?;
            }
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(PyTypeError::new_err("unsupported operand type for /"))
        }
    }

    fn __rtruediv__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let (result, warned) = py.allow_threads(move || rdiv_scalar(&a, scalar));
            if warned {
                warn_div_zero(py)?;
            }
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(PyTypeError::new_err("unsupported operand type for /"))
        }
    }

    fn __floordiv__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let q = Self::__truediv__(slf, other)?;
        let q = q.borrow();
        let inner = math::apply_unary(&q.inner, extra::floor_interval);
        Ok(Py::new(py, PyIntervalArray { inner })?.into_bound(py))
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: arithmetic::neg_array(&self.inner),
        }
    }

    fn __pos__(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn __abs__(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::abs_interval),
        }
    }

    fn __pow__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err("pow() with a modulo is not supported"));
        }
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || pow_dispatch(&a, &b))?;
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || extra::pow_scalar(&a, scalar));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(PyTypeError::new_err("unsupported operand type for **"))
        }
    }

    fn __rpow__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err("pow() with a modulo is not supported"));
        }
        let py = slf.py();
        if let Ok(scalar) = other.extract::<f64>() {
            let a = slf.borrow().inner.clone();
            let result = py.allow_threads(move || extra::rpow_scalar(&a, scalar));
            Ok(Py::new(py, PyIntervalArray { inner: result })?.into_bound(py))
        } else {
            Err(PyTypeError::new_err("unsupported operand type for **"))
        }
    }

    fn __matmul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if let Ok(other_arr) = other.downcast::<PyIntervalArray>() {
            let a = slf.borrow().inner.clone();
            let b = other_arr.borrow().inner.clone();
            let result = py.allow_threads(move || reduction::matmul_general(&a, &b));
            match result {
                Ok(reduction::MatmulResult::Scalar(iv)) => {
                    Ok((iv.midpoint(), iv.radius()).to_object(py).into_any())
                }
                Ok(reduction::MatmulResult::Array(arr)) => {
                    Ok(Py::new(py, PyIntervalArray { inner: arr })?.into_any())
                }
                Err(msg) => Err(PyValueError::new_err(msg)),
            }
        } else {
            Err(PyTypeError::new_err("unsupported operand type for @"))
        }
    }

    fn __rmatmul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Self::__matmul__(slf, other)
    }

    // ── In-place operators ──

    fn __iadd__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = other.py();
        apply_inplace(
            py,
            &mut self.inner,
            other,
            |_, a, b| add_dispatch(a, b),
            |_, a, s| Ok(arithmetic::add_scalar(a, Interval::exact(s))),
        )
    }

    fn __isub__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = other.py();
        apply_inplace(
            py,
            &mut self.inner,
            other,
            |_, a, b| sub_dispatch(a, b),
            |_, a, s| Ok(arithmetic::add_scalar(a, Interval::exact(-s))),
        )
    }

    fn __imul__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = other.py();
        apply_inplace(
            py,
            &mut self.inner,
            other,
            |_, a, b| mul_dispatch(a, b),
            |_, a, s| Ok(arithmetic::scale_array(a, s)),
        )
    }

    fn __itruediv__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = other.py();
        apply_inplace(
            py,
            &mut self.inner,
            other,
            |py, a, b| {
                let (r, warned) = div_dispatch(a, b)?;
                if warned {
                    warn_div_zero(py)?;
                }
                Ok(r)
            },
            |py, a, s| {
                let (r, warned) = scalar_div_array(a, s);
                if warned {
                    warn_div_zero(py)?;
                }
                Ok(r)
            },
        )
    }

    fn __ipow__(
        &mut self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err("pow() with a modulo is not supported"));
        }
        let py = other.py();
        apply_inplace(
            py,
            &mut self.inner,
            other,
            |_, a, b| pow_dispatch(a, b),
            |_, a, s| Ok(extra::pow_scalar(a, s)),
        )
    }

    // ── Comparisons ──

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if other.is_none() {
            return Ok(false.to_object(py).into_any());
        }
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let v = compare_dispatch(&self.inner, &o.borrow().inner, compare::Cmp::Eq)?;
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        if let Ok(s) = other.extract::<f64>() {
            let v = compare::compare_scalar(&self.inner, s, compare::Cmp::Eq);
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        Ok(py.NotImplemented())
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if other.is_none() {
            return Ok(true.to_object(py).into_any());
        }
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let v = compare_dispatch(&self.inner, &o.borrow().inner, compare::Cmp::Ne)?;
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        if let Ok(s) = other.extract::<f64>() {
            let v = compare::compare_scalar(&self.inner, s, compare::Cmp::Ne);
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        Ok(py.NotImplemented())
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let v = compare_dispatch(&self.inner, &o.borrow().inner, compare::Cmp::Lt)?;
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        if let Ok(s) = other.extract::<f64>() {
            let v = compare::compare_scalar(&self.inner, s, compare::Cmp::Lt);
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        Ok(py.NotImplemented())
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let v = compare_dispatch(&self.inner, &o.borrow().inner, compare::Cmp::Le)?;
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        if let Ok(s) = other.extract::<f64>() {
            let v = compare::compare_scalar(&self.inner, s, compare::Cmp::Le);
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        Ok(py.NotImplemented())
    }

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let v = compare_dispatch(&self.inner, &o.borrow().inner, compare::Cmp::Gt)?;
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        if let Ok(s) = other.extract::<f64>() {
            let v = compare::compare_scalar(&self.inner, s, compare::Cmp::Gt);
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        Ok(py.NotImplemented())
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let v = compare_dispatch(&self.inner, &o.borrow().inner, compare::Cmp::Ge)?;
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        if let Ok(s) = other.extract::<f64>() {
            let v = compare::compare_scalar(&self.inner, s, compare::Cmp::Ge);
            return Ok(Py::new(py, PyBoolArray::new(v))?.into_any());
        }
        Ok(py.NotImplemented())
    }

    // ── Math functions ──

    fn sin(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::sin_interval),
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
        Self {
            inner: math::apply_unary(&self.inner, math::exp_interval),
        }
    }

    fn ln(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::ln_interval),
        }
    }

    /// NumPy-style alias for `ln`.
    fn log(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::ln_interval),
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
        Self {
            inner: math::apply_unary(&self.inner, math::sqrt_interval),
        }
    }

    fn abs(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, math::abs_interval),
        }
    }

    fn floor(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, extra::floor_interval),
        }
    }

    fn ceil(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, extra::ceil_interval),
        }
    }

    fn trunc(&self) -> Self {
        Self {
            inner: math::apply_unary(&self.inner, extra::trunc_interval),
        }
    }

    #[pyo3(signature = (ndigits=None))]
    fn round(&self, ndigits: Option<i32>) -> Self {
        match ndigits {
            None => Self {
                inner: math::apply_unary(&self.inner, extra::round_interval),
            },
            Some(d) => Self {
                inner: math::apply_unary(&self.inner, move |iv, m| {
                    extra::round_ndigits_interval(iv, m, d)
                }),
            },
        }
    }

    #[pyo3(signature = (ndigits=None))]
    fn __round__(&self, ndigits: Option<i32>) -> Self {
        match ndigits {
            None => Self {
                inner: math::apply_unary(&self.inner, extra::round_interval),
            },
            Some(d) => Self {
                inner: math::apply_unary(&self.inner, move |iv, m| {
                    extra::round_ndigits_interval(iv, m, d)
                }),
            },
        }
    }

    fn __getstate__(&self) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        (
            self.inner.shape().to_vec(),
            self.inner.data().midpoints().to_vec(),
            self.inner.data().radii().to_vec(),
        )
    }

    fn __reduce__(
        &self,
        py: Python<'_>,
    ) -> PyResult<(PyObject, (Vec<f64>, Vec<f64>, Vec<usize>))> {
        let module = pyo3::types::PyModule::import_bound(py, "precise_numpy._precise_numpy")?;
        let ctor = module.getattr("from_raw_parts")?;
        let state = (
            self.inner.data().midpoints().to_vec(),
            self.inner.data().radii().to_vec(),
            self.inner.shape().to_vec(),
        );
        Ok((ctor.into_any().unbind(), state))
    }

    fn __setstate__(
        &mut self,
        state: (Vec<usize>, Vec<f64>, Vec<f64>),
    ) -> PyResult<()> {
        let (shape, mids, rads) = state;
        if mids.len() != rads.len() {
            return Err(PyValueError::new_err(
                "midpoints and radii must have the same length",
            ));
        }
        if let Some(&r) = rads.iter().find(|&&r| r < 0.0) {
            return Err(PyValueError::new_err(format!(
                "radii must be non-negative, got {}",
                r
            )));
        }
        let total: usize = shape.iter().product();
        if mids.len() != total {
            return Err(PyValueError::new_err(format!(
                "midpoints length {} != product of shape {:?}",
                mids.len(),
                shape
            )));
        }
        self.inner = IntervalArray::from_raw_parts(&mids, &rads, &shape);
        Ok(())
    }

    #[pyo3(signature = (a_min, a_max))]
    fn clip(&self, a_min: f64, a_max: f64) -> PyResult<Self> {
        if a_min > a_max {
            return Err(PyValueError::new_err(format!(
                "a_min must be <= a_max; got a_min={} a_max={}",
                a_min, a_max
            )));
        }
        Ok(Self {
            inner: extra::clip_array(&self.inner, a_min, a_max),
        })
    }

    fn sign(&self) -> Self {
        Self {
            inner: extra::sign_array(&self.inner),
        }
    }

    fn nan_to_num(&self) -> Self {
        Self {
            inner: extra::nan_to_num_array(&self.inner),
        }
    }

    fn power(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = other.py();
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let a = self.inner.clone();
            let b = o.borrow().inner.clone();
            let result = py.allow_threads(move || pow_dispatch(&a, &b))?;
            Ok(Self { inner: result })
        } else if let Ok(s) = other.extract::<f64>() {
            let a = self.inner.clone();
            let result = py.allow_threads(move || extra::pow_scalar(&a, s));
            Ok(Self { inner: result })
        } else {
            Err(PyTypeError::new_err("unsupported operand type for power"))
        }
    }

    #[pyo3(signature = (other))]
    fn maximum(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let a = self.inner.clone();
        let (ba, bb) = if let Ok(o) = other.downcast::<PyIntervalArray>() {
            broadcast_for_op(&a, &o.borrow().inner)?
        } else if let Ok(s) = other.extract::<f64>() {
            let b = broadcast::broadcast_to(&IntervalArray::from_f64_slice(&[s]), a.shape());
            (a, b)
        } else {
            return Err(PyTypeError::new_err("unsupported operand type for maximum"));
        };
        let n = ba.len();
        let mut out_mids = vec![0.0f64; n];
        let mut out_rads = vec![0.0f64; n];
        for i in 0..n {
            let x = ba.get(i);
            let y = bb.get(i);
            let r = Interval::new(x.lo.max(y.lo), x.hi.max(y.hi));
            out_mids[i] = r.midpoint();
            out_rads[i] = r.radius();
        }
        Ok(Self {
            inner: IntervalArray::from_raw_parts(&out_mids, &out_rads, ba.shape()),
        })
    }

    #[pyo3(signature = (other))]
    fn minimum(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let a = self.inner.clone();
        let (ba, bb) = if let Ok(o) = other.downcast::<PyIntervalArray>() {
            broadcast_for_op(&a, &o.borrow().inner)?
        } else if let Ok(s) = other.extract::<f64>() {
            let b = broadcast::broadcast_to(&IntervalArray::from_f64_slice(&[s]), a.shape());
            (a, b)
        } else {
            return Err(PyTypeError::new_err("unsupported operand type for minimum"));
        };
        let n = ba.len();
        let mut out_mids = vec![0.0f64; n];
        let mut out_rads = vec![0.0f64; n];
        for i in 0..n {
            let x = ba.get(i);
            let y = bb.get(i);
            let r = Interval::new(x.lo.min(y.lo), x.hi.min(y.hi));
            out_mids[i] = r.midpoint();
            out_rads[i] = r.radius();
        }
        Ok(Self {
            inner: IntervalArray::from_raw_parts(&out_mids, &out_rads, ba.shape()),
        })
    }

    // ── Reductions ──

    #[pyo3(signature = (axis=None))]
    fn sum<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        let iv = reduction::sum(&self.inner);
        match axis {
            None => Ok((iv.midpoint(), iv.radius()).to_object(py).into_any()),
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                axis_reduce_return(py, &extra::sum_axis(&self.inner, ax))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn mean<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        let iv = reduction::mean(&self.inner);
        match axis {
            None => Ok((iv.midpoint(), iv.radius()).to_object(py).into_any()),
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                axis_reduce_return(py, &extra::mean_axis(&self.inner, ax))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn prod<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        let iv = reduction::prod(&self.inner);
        match axis {
            None => Ok((iv.midpoint(), iv.radius()).to_object(py).into_any()),
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                axis_reduce_return(py, &extra::prod_axis(&self.inner, ax))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn var<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        let iv = reduction::var(&self.inner);
        match axis {
            None => Ok((iv.midpoint(), iv.radius()).to_object(py).into_any()),
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                axis_reduce_return(py, &extra::var_axis(&self.inner, ax))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn std<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        let iv = reduction::std_dev(&self.inner);
        match axis {
            None => Ok((iv.midpoint(), iv.radius()).to_object(py).into_any()),
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                axis_reduce_return(py, &extra::std_axis(&self.inner, ax))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn max<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        if self.inner.is_empty() {
            return Err(PyValueError::new_err(
                "max of an empty array has no value",
            ));
        }
        match axis {
            None => {
                let iv = reduction::max(&self.inner);
                Ok((iv.midpoint(), iv.radius()).to_object(py).into_any())
            }
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                axis_reduce_return(py, &extra::max_axis(&self.inner, ax))
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn min<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        if self.inner.is_empty() {
            return Err(PyValueError::new_err(
                "min of an empty array has no value",
            ));
        }
        match axis {
            None => {
                let iv = reduction::min(&self.inner);
                Ok((iv.midpoint(), iv.radius()).to_object(py).into_any())
            }
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                axis_reduce_return(py, &extra::min_axis(&self.inner, ax))
            }
        }
    }

    /// Backward-compatible alias for `min()`.
    fn min_val(&self) -> (f64, f64) {
        let iv = reduction::min(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    /// Backward-compatible alias for `max()`.
    fn max_val(&self) -> (f64, f64) {
        let iv = reduction::max(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    #[pyo3(signature = (axis=None))]
    fn argmax<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        if self.inner.is_empty() {
            return Err(PyValueError::new_err("attempt to get argmax of an empty array"));
        }
        match axis {
            None => Ok(extra::arg_extreme_flat(&self.inner, true).to_object(py).into_any()),
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                let v = extra::arg_extreme_axis(&self.inner, ax, true);
                axis_list_return(py, &v)
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn argmin<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        if self.inner.is_empty() {
            return Err(PyValueError::new_err("attempt to get argmin of an empty array"));
        }
        match axis {
            None => Ok(extra::arg_extreme_flat(&self.inner, false).to_object(py).into_any()),
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                let v = extra::arg_extreme_axis(&self.inner, ax, false);
                axis_list_return(py, &v)
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn all<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        match axis {
            None => {
                let ok = !self
                    .inner
                    .data()
                    .midpoints()
                    .iter()
                    .zip(self.inner.data().radii().iter())
                    .any(|(&m, &r)| m == 0.0 && r == 0.0);
                Ok(ok.to_object(py).into_any())
            }
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                let v = extra::all_any_axis(&self.inner, ax, true);
                if v.len() == 1 {
                    Ok(v[0].to_object(py).into_any())
                } else {
                    Ok(Py::new(py, PyBoolArray::new(v))?.into_any())
                }
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn any<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        match axis {
            None => {
                let ok = self
                    .inner
                    .data()
                    .midpoints()
                    .iter()
                    .zip(self.inner.data().radii().iter())
                    .any(|(&m, &r)| m != 0.0 || r != 0.0);
                Ok(ok.to_object(py).into_any())
            }
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                let v = extra::all_any_axis(&self.inner, ax, false);
                if v.len() == 1 {
                    Ok(v[0].to_object(py).into_any())
                } else {
                    Ok(Py::new(py, PyBoolArray::new(v))?.into_any())
                }
            }
        }
    }

    #[pyo3(signature = (axis=None))]
    fn cumsum(&self, axis: Option<usize>) -> PyResult<Self> {
        match axis {
            None => Ok(Self {
                inner: reduction::cumsum(&self.inner),
            }),
            Some(ax) => {
                check_axis(self.inner.ndim(), ax)?;
                Ok(Self {
                    inner: extra::cumsum_axis(&self.inner, ax),
                })
            }
        }
    }

    fn dot<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let a = self.inner.clone();
            let b = o.borrow().inner.clone();
            let result = py.allow_threads(move || reduction::dot_general(&a, &b));
            match result {
                Ok(reduction::MatmulResult::Scalar(iv)) => {
                    Ok((iv.midpoint(), iv.radius()).to_object(py).into_any())
                }
                Ok(reduction::MatmulResult::Array(arr)) => {
                    Ok(Py::new(py, PyIntervalArray { inner: arr })?.into_any())
                }
                Err(msg) => Err(PyValueError::new_err(msg)),
            }
        } else {
            Err(PyTypeError::new_err("unsupported operand type for dot"))
        }
    }

    fn matmul<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(o) = other.downcast::<PyIntervalArray>() {
            let a = self.inner.clone();
            let b = o.borrow().inner.clone();
            let result = py.allow_threads(move || reduction::matmul_general(&a, &b));
            match result {
                Ok(reduction::MatmulResult::Scalar(iv)) => {
                    Ok((iv.midpoint(), iv.radius()).to_object(py).into_any())
                }
                Ok(reduction::MatmulResult::Array(arr)) => {
                    Ok(Py::new(py, PyIntervalArray { inner: arr })?.into_any())
                }
                Err(msg) => Err(PyValueError::new_err(msg)),
            }
        } else {
            Err(PyTypeError::new_err("unsupported operand type for matmul"))
        }
    }

    fn norm(&self) -> (f64, f64) {
        let iv = reduction::norm_l2(&self.inner);
        (iv.midpoint(), iv.radius())
    }

    // ── Predicates and conversions ──

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

    fn isnan(&self) -> PyBoolArray {
        PyBoolArray::new(compare::is_nan_array(&self.inner))
    }

    fn isinf(&self) -> PyBoolArray {
        PyBoolArray::new(compare::is_inf_array(&self.inner))
    }

    fn isfinite(&self) -> PyBoolArray {
        PyBoolArray::new(compare::is_finite_array(&self.inner))
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let shape = self.inner.shape().to_vec();
        let mids = self.inner.data().midpoints();
        let rads = self.inner.data().radii();
        let flat: Vec<(f64, f64)> = (0..self.inner.len()).map(|i| (mids[i], rads[i])).collect();
        let mut offset = 0usize;
        Ok(build_nested_lists(py, &flat, &shape, &mut offset)?)
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

    fn flatten(&self) -> Self {
        Self {
            inner: self.inner.reshape(&[self.inner.len()]),
        }
    }

    fn ravel(&self) -> Self {
        self.flatten()
    }

    fn sort(&self) -> Self {
        Self {
            inner: extra::sorted_array(&self.inner),
        }
    }

    fn argsort(&self) -> Vec<usize> {
        extra::argsort(&self.inner)
    }

    fn nonzero<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        let lists = extra::nonzero(&self.inner);
        PyTuple::new_bound(py, lists)
    }

    #[pyo3(signature = (shape, *extra))]
    fn reshape(&self, shape: &Bound<'_, PyAny>, extra: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let mut dims: Vec<isize> = Vec::new();
        if let Ok(single) = shape.extract::<isize>() {
            dims.push(single);
        } else if let Ok(v) = shape.extract::<Vec<isize>>() {
            dims.extend(v);
        } else {
            return Err(PyTypeError::new_err(
                "reshape: shape must be an int or a sequence of ints",
            ));
        }
        for item in extra.iter() {
            dims.push(item.extract::<isize>()?);
        }
        // Resolve -1 (infer the missing dimension).
        let mut resolved = Vec::with_capacity(dims.len());
        let mut known_product: i128 = 1;
        let mut inferred: Option<usize> = None;
        for &d in &dims {
            if d == -1 {
                if inferred.is_some() {
                    return Err(PyValueError::new_err(
                        "can only specify one unknown dimension",
                    ));
                }
                inferred = Some(resolved.len());
                resolved.push(1);
            } else if d < 0 {
                return Err(PyValueError::new_err(
                    "negative dimensions are not allowed",
                ));
            } else {
                resolved.push(d as usize);
                known_product *= d as i128;
            }
        }
        let total = self.inner.len() as i128;
        if let Some(pos) = inferred {
            if known_product == 0 || total % known_product != 0 {
                return Err(PyValueError::new_err(format!(
                    "cannot reshape array of size {} into shape {:?}",
                    self.inner.len(),
                    dims
                )));
            }
            resolved[pos] = (total / known_product) as usize;
        }
        let new_total: usize = resolved.iter().product();
        if new_total != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {} into shape {:?}",
                self.inner.len(),
                resolved
            )));
        }
        Ok(Self {
            inner: self.inner.reshape(&resolved),
        })
    }

    fn transpose(&self) -> PyResult<Self> {
        let ndim = self.inner.ndim();
        if ndim == 1 {
            return Ok(Self {
                inner: self.inner.clone(),
            });
        }
        if ndim != 2 {
            return Err(PyValueError::new_err(format!(
                "transpose only supports 1D and 2D arrays, got {}D",
                ndim
            )));
        }
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

/// Convert an axis-reduction array result into a scalar tuple (0-d) or an
/// IntervalArray.
fn axis_reduce_return(py: Python<'_>, arr: &IntervalArray) -> PyResult<Py<PyAny>> {
    if arr.shape().is_empty() {
        let iv = arr.get(0);
        return Ok((iv.midpoint(), iv.radius()).to_object(py).into_any());
    }
    Ok(Py::new(py, PyIntervalArray { inner: arr.clone() })?.into_any())
}

fn check_axis(ndim: usize, axis: usize) -> PyResult<()> {
    if axis >= ndim {
        return Err(PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        )));
    }
    Ok(())
}

/// Convert an axis arg-extreme result into an int (0-d) or a list of ints.
fn axis_list_return(py: Python<'_>, v: &[usize]) -> PyResult<Py<PyAny>> {
    if v.len() == 1 {
        return Ok(v[0].to_object(py).into_any());
    }
    Ok(v.to_vec().to_object(py).into_any())
}

/// Element-wise scalar-over-array division: s / x, returning entire for
/// zero-crossing divisors. The bool signals whether a zero was encountered.
fn rdiv_scalar(a: &IntervalArray, s: f64) -> (IntervalArray, bool) {
    let n = a.len();
    let a_mids = a.data().midpoints();
    let a_rads = a.data().radii();
    let mut out_mids = vec![0.0f64; n];
    let mut out_rads = vec![0.0f64; n];
    let mut warn = false;
    for i in 0..n {
        let am = a_mids[i];
        let ar = a_rads[i];
        let denom_lo = am - ar;
        let denom_hi = am + ar;
        if denom_lo <= 0.0 && denom_hi >= 0.0 {
            warn = true;
            out_mids[i] = 0.0;
            out_rads[i] = f64::INFINITY;
            continue;
        }
        let r = s / am;
        out_mids[i] = r;
        let exact_err = if r.is_finite() {
            am.mul_add(r, -s).abs()
        } else {
            f64::INFINITY
        };
        let den = if denom_lo > 0.0 { denom_lo } else { -denom_hi }.abs();
        let nums = if den > 0.0 {
            crate::error::interval::add_ru_chain(
                crate::error::interval::add_ru_chain(
                    crate::error::interval::mul_ru(r.abs(), ar),
                    exact_err,
                ),
                crate::error::interval::mul_ru(ar, ar),
            )
        } else {
            f64::INFINITY
        };
        out_rads[i] = crate::error::interval::div_ru(nums, den);
    }
    (IntervalArray::from_raw_parts(&out_mids, &out_rads, a.shape()), warn)
}

// ══════════════════════════════════════════════════════════════════════
// Module-level functions
// ══════════════════════════════════════════════════════════════════════

#[pyfunction(signature = (values, error=0.0))]
#[pyo3(name = "array")]
fn parray(values: &Bound<'_, PyAny>, error: f64) -> PyResult<PyIntervalArray> {
    PyIntervalArray::new(values, error)
}

#[pyfunction(signature = (shape, *extra))]
fn zeros(shape: &Bound<'_, PyAny>, extra: &Bound<'_, PyTuple>) -> PyResult<PyIntervalArray> {
    let shape = extract_shape(shape, extra)?;
    Ok(PyIntervalArray {
        inner: IntervalArray::zeros(&shape),
    })
}

#[pyfunction(signature = (shape, *extra))]
fn ones(shape: &Bound<'_, PyAny>, extra: &Bound<'_, PyTuple>) -> PyResult<PyIntervalArray> {
    let shape = extract_shape(shape, extra)?;
    Ok(PyIntervalArray {
        inner: IntervalArray::ones(&shape),
    })
}

#[pyfunction(signature = (shape, *extra))]
fn empty(shape: &Bound<'_, PyAny>, extra: &Bound<'_, PyTuple>) -> PyResult<PyIntervalArray> {
    let shape = extract_shape(shape, extra)?;
    Ok(PyIntervalArray {
        inner: IntervalArray::zeros(&shape),
    })
}

#[pyfunction(signature = (shape, value, error=0.0))]
fn full(
    shape: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    error: f64,
) -> PyResult<PyIntervalArray> {
    if error < 0.0 {
        return Err(PyValueError::new_err("error must be non-negative"));
    }
    let shape = extract_shape(shape, &PyTuple::empty_bound(shape.py()))?;
    let (mid, rad) = if let Ok(v) = value.extract::<f64>() {
        (v, error)
    } else if let Ok((m, r)) = value.extract::<(f64, f64)>() {
        (m, r)
    } else {
        return Err(PyTypeError::new_err(
            "full value must be a number or a (midpoint, radius) tuple",
        ));
    };
    Ok(PyIntervalArray {
        inner: IntervalArray::full(&shape, Interval::from_midpoint_radius(mid, rad)),
    })
}

#[pyfunction(signature = (n, m=None, k=0))]
fn eye(n: usize, m: Option<usize>, k: isize) -> PyResult<PyIntervalArray> {
    let cols = m.unwrap_or(n);
    if k.is_negative() {
        // diagonal starts `k` rows down
        let row0 = (-k) as usize;
        let mut mids = vec![0.0f64; n * cols];
        for i in 0..n {
            if i >= row0 {
                let j = i - row0;
                if j < cols {
                    mids[i * cols + j] = 1.0;
                }
            }
        }
        let rads = vec![0.0f64; n * cols];
        return Ok(PyIntervalArray {
            inner: IntervalArray::from_raw_parts(&mids, &rads, &[n, cols]),
        });
    }
    let col0 = k as usize;
    let mut mids = vec![0.0f64; n * cols];
    for i in 0..n {
        let j = i + col0;
        if j < cols {
            mids[i * cols + j] = 1.0;
        }
    }
    let rads = vec![0.0f64; n * cols];
    Ok(PyIntervalArray {
        inner: IntervalArray::from_raw_parts(&mids, &rads, &[n, cols]),
    })
}

#[pyfunction]
fn identity(n: usize) -> PyResult<PyIntervalArray> {
    eye(n, None, 0)
}

#[pyfunction]
fn diag(v: &Bound<'_, PyAny>) -> PyResult<PyIntervalArray> {
    if let Ok(arr) = v.downcast::<PyIntervalArray>() {
        let a = arr.borrow().inner.clone();
        if a.ndim() == 1 {
            let n = a.len();
            let mut mids = vec![0.0f64; n * n];
            let mut rads = vec![0.0f64; n * n];
            for i in 0..n {
                let iv = a.get(i);
                mids[i * n + i] = iv.midpoint();
                rads[i * n + i] = iv.radius();
            }
            Ok(PyIntervalArray {
                inner: IntervalArray::from_raw_parts(&mids, &rads, &[n, n]),
            })
        } else if a.ndim() == 2 {
            let n = a.shape()[0].min(a.shape()[1]);
            let mut mids = Vec::with_capacity(n);
            let mut rads = Vec::with_capacity(n);
            for i in 0..n {
                let iv = a.get(i * a.shape()[1] + i);
                mids.push(iv.midpoint());
                rads.push(iv.radius());
            }
            Ok(PyIntervalArray {
                inner: IntervalArray::from_raw_parts(&mids, &rads, &[n]),
            })
        } else {
            Err(PyValueError::new_err("diag requires a 1D or 2D array"))
        }
    } else if let Ok(vals) = v.extract::<Vec<f64>>() {
        let n = vals.len();
        let mut mids = vec![0.0f64; n * n];
        for i in 0..n {
            mids[i * n + i] = vals[i];
        }
        let rads = vec![0.0f64; n * n];
        Ok(PyIntervalArray {
            inner: IntervalArray::from_raw_parts(&mids, &rads, &[n, n]),
        })
    } else {
        Err(PyTypeError::new_err("diag requires a 1D or 2D array"))
    }
}

#[pyfunction(signature = (start, stop, num=50, endpoint=true))]
fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> PyResult<PyIntervalArray> {
    let arr = if num == 0 {
        IntervalArray::zeros(&[0])
    } else if num == 1 {
        IntervalArray::from_f64_slice(&[start])
    } else if endpoint {
        let step = (stop - start) / (num - 1) as f64;
        let values: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
        IntervalArray::from_f64_slice(&values)
    } else {
        let step = (stop - start) / num as f64;
        let values: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
        IntervalArray::from_f64_slice(&values)
    };
    Ok(PyIntervalArray { inner: arr })
}

#[pyfunction(signature = (start, stop, step=1.0))]
fn arange(start: f64, stop: f64, step: f64) -> PyResult<PyIntervalArray> {
    if step == 0.0 {
        return Err(PyValueError::new_err("arange: step cannot be zero"));
    }
    Ok(PyIntervalArray {
        inner: IntervalArray::arange(start, stop, step),
    })
}

fn extract_arrays<'py>(
    arrays: &Bound<'py, PyAny>,
) -> PyResult<Vec<Py<PyIntervalArray>>> {
    arrays.extract::<Vec<Py<PyIntervalArray>>>().map_err(|_| {
        PyTypeError::new_err("expected a list of IntervalArray objects")
    })
}

#[pyfunction(signature = (arrays, axis=0))]
fn concatenate(
    py: Python<'_>,
    arrays: &Bound<'_, PyAny>,
    axis: usize,
) -> PyResult<PyIntervalArray> {
    let arrs = extract_arrays(arrays)?;
    let refs: Vec<IntervalArray> = arrs
        .iter()
        .map(|a| a.borrow(py).inner.clone())
        .collect();
    let refs2: Vec<&IntervalArray> = refs.iter().collect();
    let result = extra::concatenate(&refs2, axis).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyIntervalArray { inner: result })
}

#[pyfunction(signature = (arrays, axis=0))]
fn stack(py: Python<'_>, arrays: &Bound<'_, PyAny>, axis: usize) -> PyResult<PyIntervalArray> {
    let arrs = extract_arrays(arrays)?;
    let refs: Vec<IntervalArray> = arrs
        .iter()
        .map(|a| a.borrow(py).inner.clone())
        .collect();
    let refs2: Vec<&IntervalArray> = refs.iter().collect();
    let result = extra::stack(&refs2, axis).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyIntervalArray { inner: result })
}

#[pyfunction]
fn vstack(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<PyIntervalArray> {
    stack(py, arrays, 0)
}

#[pyfunction]
fn hstack(py: Python<'_>, arrays: &Bound<'_, PyAny>) -> PyResult<PyIntervalArray> {
    let arrs = extract_arrays(arrays)?;
    let refs: Vec<IntervalArray> = arrs
        .iter()
        .map(|a| a.borrow(py).inner.clone())
        .collect();
    if refs.is_empty() {
        return Err(PyValueError::new_err("need at least one array to hstack"));
    }
    let axis = if refs[0].ndim() == 1 { 0 } else { 1 };
    let refs2: Vec<&IntervalArray> = refs.iter().collect();
    let result = extra::concatenate(&refs2, axis).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyIntervalArray { inner: result })
}

#[pyfunction(signature = (a, indices_or_sections, axis=0))]
fn split(
    a: &Bound<'_, PyIntervalArray>,
    indices_or_sections: &Bound<'_, PyAny>,
    axis: usize,
) -> PyResult<Vec<PyIntervalArray>> {
    let inner = a.borrow().inner.clone();
    if axis >= inner.ndim() {
        return Err(PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis,
            inner.ndim()
        )));
    }
    let indices: Vec<usize> = if let Ok(v) = indices_or_sections.extract::<usize>() {
        if v == 0 {
            return Err(PyValueError::new_err("number of sections must be >= 1"));
        }
        let dim = inner.shape()[axis];
        if dim % v != 0 {
            return Err(PyValueError::new_err(format!(
                "array split does not result in an equal division: array size {} is not divisible by {}",
                dim, v
            )));
        }
        (1..v).map(|i| i * dim / v).collect()
    } else if let Ok(v) = indices_or_sections.extract::<Vec<usize>>() {
        v
    } else {
        return Err(PyTypeError::new_err(
            "indices_or_sections must be an int or a list of ints",
        ));
    };
    let parts =
        extra::split(&inner, &indices, axis).map_err(|e| PyValueError::new_err(e))?;
    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        out.push(PyIntervalArray { inner: p });
    }
    Ok(out)
}

#[pyfunction(signature = (condition, x, y))]
fn where_impl(
    condition: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<PyIntervalArray> {
    // Resolve condition to a Vec<bool>
    let cond_vec: Vec<bool> = if let Ok(c) = condition.downcast::<PyBoolArray>() {
        c.borrow().to_vec()
    } else if let Ok(c) = condition.downcast::<PyIntervalArray>() {
        let inner = c.borrow().inner.clone();
        inner
            .data()
            .midpoints()
            .iter()
            .zip(inner.data().radii().iter())
            .map(|(&m, &r)| m != 0.0 || r != 0.0)
            .collect()
    } else if let Ok(c) = condition.extract::<Vec<bool>>() {
        c
    } else if let Ok(c) = condition.extract::<bool>() {
        vec![c]
    } else {
        return Err(PyTypeError::new_err(
            "condition must be a BoolArray, IntervalArray, list of bools, or bool",
        ));
    };

    // Resolve x and y to IntervalArray (scalars become 1-element arrays)
    let xa: IntervalArray = if let Ok(a) = x.downcast::<PyIntervalArray>() {
        a.borrow().inner.clone()
    } else if let Ok(s) = x.extract::<f64>() {
        IntervalArray::from_f64_slice(&[s])
    } else {
        return Err(PyTypeError::new_err("x must be an IntervalArray or a number"));
    };
    let ya: IntervalArray = if let Ok(a) = y.downcast::<PyIntervalArray>() {
        a.borrow().inner.clone()
    } else if let Ok(s) = y.extract::<f64>() {
        IntervalArray::from_f64_slice(&[s])
    } else {
        return Err(PyTypeError::new_err("y must be an IntervalArray or a number"));
    };

    // Broadcast condition, x, y together.
    let cond_arr = IntervalArray::from_raw_parts(
        &cond_vec.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<f64>>(),
        &vec![0.0; cond_vec.len()],
        &[cond_vec.len()],
    );
    let shapes: Vec<&[usize]> = vec![cond_arr.shape(), xa.shape(), ya.shape()];
    let common = broadcast::broadcast_shapes_many(&shapes)
        .ok_or_else(|| PyValueError::new_err("operands could not be broadcast together"))?;
    let cb = broadcast::broadcast_to(&cond_arr, &common);
    let xb = broadcast::broadcast_to(&xa, &common);
    let yb = broadcast::broadcast_to(&ya, &common);
    let mask: Vec<bool> = cb.data().midpoints().iter().map(|&m| m != 0.0).collect();

    Ok(PyIntervalArray {
        inner: extra::where_select(&mask, &xb, &yb),
    })
}

#[pyfunction]
fn from_raw_parts(
    midpoints: Vec<f64>,
    radii: Vec<f64>,
    shape: Vec<usize>,
) -> PyResult<PyIntervalArray> {
    if midpoints.len() != radii.len() {
        return Err(PyValueError::new_err(
            "midpoints and radii must have the same length",
        ));
    }
    if let Some(&r) = radii.iter().find(|&&r| r < 0.0) {
        return Err(PyValueError::new_err(format!(
            "radii must be non-negative, got {}",
            r
        )));
    }
    let total: usize = shape.iter().product();
    if midpoints.len() != total {
        return Err(PyValueError::new_err(format!(
            "midpoints length {} != product of shape {:?}",
            midpoints.len(),
            shape
        )));
    }
    Ok(PyIntervalArray {
        inner: IntervalArray::from_raw_parts(&midpoints, &radii, &shape),
    })
}

/// Get the number of Rayon worker threads.
#[pyfunction]
fn num_threads() -> usize {
    parallel::num_threads()
}

// ── Random number generation ───────────────────────────────────────────

#[pyfunction]
fn seed(seed_value: u64) {
    random::seed(seed_value);
}

#[pyfunction(signature = (size=None, *extra))]
fn rand(
    py: Python<'_>,
    size: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    random_size_dispatch(py, size, extra, || random::random_f64())
}

#[pyfunction(signature = (size=None, *extra))]
fn random_sample(
    py: Python<'_>,
    size: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    random_size_dispatch(py, size, extra, || random::random_f64())
}

#[pyfunction(signature = (size=None, *extra))]
fn randn(
    py: Python<'_>,
    size: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    random_size_dispatch(py, size, extra, || random::random_normal())
}

#[pyfunction(signature = (low, high, size=None, *extra))]
fn randint(
    py: Python<'_>,
    low: i64,
    high: i64,
    size: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    if high <= low {
        return Err(PyValueError::new_err("high must be greater than low"));
    }
    let shape = random_shape(size, extra)?;
    if shape.is_empty() {
        return Ok(random::random_int(low, high).to_object(py).into_any());
    }
    let arr = random::randint_array(low, high, &shape);
    Ok(Py::new(py, PyIntervalArray { inner: arr })?.into_any())
}

#[pyfunction(signature = (low, high, size=None, *extra))]
fn uniform(
    py: Python<'_>,
    low: f64,
    high: f64,
    size: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    if high <= low {
        return Err(PyValueError::new_err("high must be greater than low"));
    }
    random_size_dispatch(py, size, extra, || random::random_uniform(low, high))
}

#[pyfunction(signature = (loc=0.0, scale=1.0, size=None, *extra))]
fn normal(
    py: Python<'_>,
    loc: f64,
    scale: f64,
    size: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    if scale < 0.0 {
        return Err(PyValueError::new_err("scale must be >= 0"));
    }
    random_size_dispatch(py, size, extra, || loc + scale * random::random_normal())
}

fn random_shape(
    size: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyTuple>,
) -> PyResult<Vec<usize>> {
    match size {
        None if extra.len() == 0 => Ok(vec![]),
        None => {
            let mut out = Vec::new();
            for item in extra.iter() {
                out.push(item.extract::<usize>()?);
            }
            Ok(out)
        }
        Some(s) => extract_shape(s, extra),
    }
}

/// Generate scalars (empty shape) or an array of the requested shape from `gen`.
fn random_size_dispatch(
    py: Python<'_>,
    size: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyTuple>,
    gen: impl Fn() -> f64,
) -> PyResult<Py<PyAny>> {
    let shape = random_shape(size, extra)?;
    if shape.is_empty() {
        return Ok(gen().to_object(py).into_any());
    }
    let n: usize = shape.iter().product();
    let vals: Vec<f64> = (0..n).map(|_| gen()).collect();
    let arr = IntervalArray::from_raw_parts(&vals, &vec![0.0; n], &shape);
    Ok(Py::new(py, PyIntervalArray { inner: arr })?.into_any())
}

// ── Linear algebra ─────────────────────────────────────────────────────

#[pyfunction]
fn det(a: &Bound<'_, PyIntervalArray>) -> PyResult<(f64, f64)> {
    let iv = linalg::det(&a.borrow().inner).map_err(|e| PyValueError::new_err(e))?;
    Ok((iv.midpoint(), iv.radius()))
}

#[pyfunction]
fn inv(a: &Bound<'_, PyIntervalArray>) -> PyResult<PyIntervalArray> {
    let inner = linalg::inv(&a.borrow().inner).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyIntervalArray { inner })
}

#[pyfunction]
fn solve(
    a: &Bound<'_, PyIntervalArray>,
    b: &Bound<'_, PyIntervalArray>,
) -> PyResult<PyIntervalArray> {
    let inner = linalg::solve(&a.borrow().inner, &b.borrow().inner)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyIntervalArray { inner })
}

#[pyfunction]
fn lstsq(
    a: &Bound<'_, PyIntervalArray>,
    b: &Bound<'_, PyIntervalArray>,
) -> PyResult<PyIntervalArray> {
    let inner = linalg::lstsq(&a.borrow().inner, &b.borrow().inner)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyIntervalArray { inner })
}

#[pyfunction]
fn pinv(a: &Bound<'_, PyIntervalArray>) -> PyResult<PyIntervalArray> {
    let inner = linalg::pinv(&a.borrow().inner).map_err(|e| PyValueError::new_err(e))?;
    Ok(PyIntervalArray { inner })
}

#[pyfunction]
fn eig<'py>(py: Python<'py>, a: &Bound<'py, PyIntervalArray>) -> PyResult<Py<PyAny>> {
    let (evals, evecs) = linalg::eig(&a.borrow().inner).map_err(|e| {
        if e.contains("complex") {
            PyNotImplementedError::new_err(e)
        } else {
            PyValueError::new_err(e)
        }
    })?;
    Ok((Py::new(py, PyIntervalArray { inner: evals })?, Py::new(py, PyIntervalArray { inner: evecs })?)
        .to_object(py)
        .into_any())
}

#[pyfunction]
fn svd<'py>(py: Python<'py>, a: &Bound<'py, PyIntervalArray>) -> PyResult<Py<PyAny>> {
    let (u, s, vt) = linalg::svd(&a.borrow().inner).map_err(|e| PyValueError::new_err(e))?;
    Ok((
        Py::new(py, PyIntervalArray { inner: u })?,
        Py::new(py, PyIntervalArray { inner: s })?,
        Py::new(py, PyIntervalArray { inner: vt })?,
    )
        .to_object(py)
        .into_any())
}

// ══════════════════════════════════════════════════════════════════════
// Module definition
// ══════════════════════════════════════════════════════════════════════

/// Python module definition.
#[pymodule]
fn _precise_numpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIntervalArray>()?;
    m.add_class::<PyBoolArray>()?;
    m.add_function(wrap_pyfunction!(parray, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(identity, m)?)?;
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(where_impl, m)?)?;
    m.add_function(wrap_pyfunction!(from_raw_parts, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(seed, m)?)?;
    m.add_function(wrap_pyfunction!(rand, m)?)?;
    m.add_function(wrap_pyfunction!(random_sample, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(randint, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    m.add_function(wrap_pyfunction!(normal, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(lstsq, m)?)?;
    m.add_function(wrap_pyfunction!(eig, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(pinv, m)?)?;
    m.add("__version__", "0.2.3")?;
    Ok(())
}
