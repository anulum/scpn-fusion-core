// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! Strict Python array and scalar admission before fixed-rank or unsigned extraction.

use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyFloat, PyInt},
};

/// Strip actual ndarray subclasses only after checking native float64 dtype and rank.
pub(super) fn array_input<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a NumPy ndarray"
        )));
    }
    let array = ndarray.getattr("view")?.call1((value, &ndarray))?;
    let dtype = array.getattr("dtype")?;
    if !dtype.eq(np.getattr("dtype")?.call1(("float64",))?)?
        || !dtype.getattr("isnative")?.extract::<bool>()?
    {
        return Err(PyTypeError::new_err(format!(
            "{name} must have native float64 dtype"
        )));
    }
    if array.getattr("ndim")?.extract::<usize>()? != 2 {
        return Err(PyValueError::new_err(format!("{name} must have rank two")));
    }
    Ok(array)
}

/// Check scalar kind without performing conversions or domain checks.
pub(super) fn scalar_kind(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    name: &str,
    integer: bool,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    if value.is_instance_of::<PyBool>() || value.is_instance(&np.getattr("bool_")?)? {
        return Err(PyTypeError::new_err(format!("{name} must not be Boolean")));
    }
    if value.is_instance_of::<PyInt>() || value.is_instance(&np.getattr("integer")?)? {
        return Ok(());
    }
    if !integer
        && (value.is_instance_of::<PyFloat>()
            || (value.is_instance(&np.getattr("floating")?)?
                && np
                    .getattr("generic")?
                    .getattr("dtype")?
                    .call_method1("__get__", (value,))?
                    .getattr("itemsize")?
                    .extract::<usize>()?
                    <= 8))
    {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "{name} has an unsupported scalar kind"
    )))
}

/// Read builtin or NumPy integer storage without invoking an overridden conversion hook.
fn mathematical_integer<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let int = py.import("builtins")?.getattr("int")?;
    if value.is_instance_of::<PyInt>() {
        int.getattr("__int__")?.call1((value,))
    } else {
        py.import("numpy")?
            .getattr("integer")?
            .getattr("__int__")?
            .call1((value,))
    }
}

/// Validate mathematical signed/native index bounds before unsigned narrowing.
pub(super) fn integer_value(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    name: &str,
    minimum: usize,
) -> PyResult<usize> {
    let integer = mathematical_integer(py, value)?;
    let result = integer
        .extract::<i64>()
        .map_err(|_| PyValueError::new_err(format!("{name} exceeds signed index range")))?;
    let result = usize::try_from(result)
        .map_err(|_| PyValueError::new_err(format!("{name} is negative")))?;
    if result < minimum || result > isize::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "{name} is outside its index domain"
        )));
    }
    Ok(result)
}

/// Normalize admitted real kinds, rejecting inexact integers and nonfinite values.
pub(super) fn real_value(py: Python<'_>, value: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let np = py.import("numpy")?;
    let float = py.import("builtins")?.getattr("float")?;
    let normalized =
        if value.is_instance_of::<PyInt>() || value.is_instance(&np.getattr("integer")?)? {
            let integer = mathematical_integer(py, value)?;
            let real = float
                .call1((&integer,))
                .map_err(|_| PyValueError::new_err(format!("{name} exceeds binary64 range")))?;
            if !integer.eq(&real)? {
                return Err(PyValueError::new_err(format!(
                    "{name} must be exactly representable in binary64"
                )));
            }
            real
        } else if value.is_instance_of::<PyFloat>() {
            float.getattr("__float__")?.call1((value,))?
        } else {
            np.getattr("floating")?
                .getattr("__float__")?
                .call1((value,))?
        };
    let result = normalized.extract::<f64>()?;
    if !result.is_finite() {
        return Err(PyValueError::new_err(format!("{name} must be finite")));
    }
    Ok(result)
}

/// Check element and binary64 byte sizes before array copies or grid allocation.
pub(super) fn checked_grid(nr: usize, nz: usize) -> PyResult<()> {
    if nr
        .checked_mul(nz)
        .and_then(|n| n.checked_mul(8))
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(PyValueError::new_err(
            "grid element or byte count exceeds native address space",
        ));
    }
    Ok(())
}

/// Require exact declared extents without rank-specific extraction.
pub(super) fn shape(array: &Bound<'_, PyAny>, nr: usize, nz: usize, name: &str) -> PyResult<()> {
    if array.getattr("shape")?.extract::<(usize, usize)>()? != (nz, nr) {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape ({nz}, {nr})"
        )));
    }
    Ok(())
}

/// Check every input cell before copying; ndarray subclass behavior has been stripped.
pub(super) fn finite_array(py: Python<'_>, array: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if !py
        .import("numpy")?
        .getattr("isfinite")?
        .call1((array,))?
        .call_method0("all")?
        .extract::<bool>()?
    {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

/// Copy by logical index into aligned C storage before Rust fixed-rank borrowing.
pub(super) fn owned_array(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("copy", true)?;
    kwargs.set_item("order", "C")?;
    kwargs.set_item("subok", false)?;
    let owned = py
        .import("numpy")?
        .getattr("array")?
        .call((array,), Some(&kwargs))?;
    fusion_types::array_storage::try_copy(owned.extract::<PyReadonlyArray2<'_, f64>>()?.as_array())
        .map_err(pyo3::exceptions::PyMemoryError::new_err)
}

/// Validate finite ordered endpoints and representable positive nominal spacing.
pub(super) fn grid_spacing(lower: f64, upper: f64, count: usize) -> PyResult<f64> {
    let spacing = (upper - lower) / (count - 1) as f64;
    if lower >= upper || !spacing.is_finite() || spacing <= 0.0 {
        return Err(PyValueError::new_err(
            "bounds must give finite positive spacing",
        ));
    }
    Ok(spacing)
}

/// Check both actual NumPy and ndarray axis generation before native mesh construction.
pub(super) fn actual_axis(
    py: Python<'_>,
    lower: f64,
    upper: f64,
    count: usize,
) -> PyResult<Vec<f64>> {
    let numpy_axis = py
        .import("numpy")?
        .getattr("linspace")?
        .call1((lower, upper, count))?;
    let numpy_axis = numpy_axis.extract::<numpy::PyReadonlyArray1<'_, f64>>()?;
    let axis = numpy_axis.as_array();
    if !axis.iter().all(|v| v.is_finite())
        || !axis.iter().zip(axis.iter().skip(1)).all(|(a, b)| b > a)
    {
        return Err(PyValueError::new_err(
            "actual NumPy coordinates must be finite and increasing",
        ));
    }
    let step = (upper - lower) / (count - 1) as f64;
    let mut previous = lower;
    for index in 1..count {
        let value = lower + step * index as f64;
        if !value.is_finite() || value <= previous {
            return Err(PyValueError::new_err(
                "actual native coordinates must be finite and increasing",
            ));
        }
        previous = value;
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| pyo3::exceptions::PyMemoryError::new_err(error.to_string()))?;
    values.extend(axis.iter().copied());
    Ok(values)
}

/// Map native allocation errors separately from invalid geometry or numerical execution.
pub(super) fn native_error(error: String, geometry: bool) -> PyErr {
    if fusion_types::array_storage::is_allocation_failure(&error) {
        pyo3::exceptions::PyMemoryError::new_err(error)
    } else if geometry {
        PyValueError::new_err(error)
    } else {
        pyo3::exceptions::PyRuntimeError::new_err(error)
    }
}

/// Preserve optional Python arguments verbatim while representing signature defaults natively.
pub(crate) enum ScalarArgument<'py> {
    /// An explicitly supplied object, including invalid kinds that admission must reject.
    Object(Bound<'py, PyAny>),
    /// The documented binary64 default for tolerance.
    RealDefault(f64),
    /// The documented integer default for the cycle budget.
    IntegerDefault(usize),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ScalarArgument<'py> {
    type Error = PyErr;
    /// Retain the object without coercion; kind checks occur in signature order later.
    fn extract(value: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self::Object(value.to_owned()))
    }
}

impl<'py> ScalarArgument<'py> {
    /// Materialize a documented default or recover the actual supplied Python object.
    pub(super) fn into_object(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Self::Object(value) => value,
            Self::RealDefault(value) => value.into_pyobject(py)?.into_any(),
            Self::IntegerDefault(value) => value.into_pyobject(py)?.into_any(),
        })
    }
}
