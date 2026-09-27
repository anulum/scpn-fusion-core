// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Fallible array storage
//! Fallible owned numerical storage for checked public solver boundaries.

use ndarray::{Array1, Array2};

/// Reserve the exact binary64 buffer while retaining allocation failures as a typed Result.
fn buffer(count: usize) -> Result<Vec<f64>, String> {
    if count
        .checked_mul(8)
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(
            "allocation failure: binary64 byte count exceeds native address space".to_owned(),
        );
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| format!("allocation failure: {error}"))?;
    Ok(values)
}

/// Identify the reserved allocation-error category carried by native Result APIs.
pub fn is_allocation_failure(error: &str) -> bool {
    error.starts_with("allocation failure:")
}

/// Allocate a zero-filled C matrix with checked element and byte arithmetic.
pub fn try_zeros(nz: usize, nr: usize) -> Result<Array2<f64>, String> {
    let count = nz
        .checked_mul(nr)
        .ok_or_else(|| "allocation failure: element count overflow".to_owned())?;
    let mut values = buffer(count)?;
    values.resize(count, 0.0);
    Array2::from_shape_vec((nz, nr), values).map_err(|error| format!("allocation failure: {error}"))
}

/// Copy actual logical cells into fallibly reserved, independently owned C storage.
pub fn try_copy(values: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let mut owned = buffer(values.len())?;
    owned.extend(values.iter().copied());
    Array2::from_shape_vec(values.dim(), owned)
        .map_err(|error| format!("allocation failure: {error}"))
}

/// Generate exactly the ndarray linspace operation sequence with fallible storage.
pub fn try_axis(lower: f64, upper: f64, count: usize) -> Result<Array1<f64>, String> {
    if count < 2 {
        return Err("grid axis requires at least two coordinates".to_owned());
    }
    let step = (upper - lower) / (count - 1) as f64;
    let mut values = buffer(count)?;
    values.extend((0..count).map(|index| lower + step * index as f64));
    Ok(Array1::from_vec(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    /// Refuse both element and byte overflow without constructing a giant array.
    #[test]
    fn checked_allocation_arithmetic() {
        for (nz, nr) in [(usize::MAX, 2), (1, isize::MAX as usize / 8 + 1)] {
            let error = try_zeros(nz, nr).unwrap_err();
            assert!(is_allocation_failure(&error));
        }
        assert!(!is_allocation_failure("invalid grid"));
        assert!(try_axis(0.0, 1.0, 1).is_err());
    }

    /// Copy negative-stride logical values independently into standard row-major storage.
    #[test]
    fn logical_copy_is_independent() {
        let mut original = Array2::from_shape_fn((3, 5), |(z, r)| (z * 5 + r) as f64);
        let copied = try_copy(original.slice(s![..;-1, ..;-1])).unwrap();
        assert!(copied.is_standard_layout());
        assert_eq!(copied[[0, 0]], 14.0);
        original.fill(-1.0);
        assert_eq!(copied[[0, 0]], 14.0);
        assert_eq!(try_zeros(3, 5).unwrap().sum(), 0.0);
        assert_eq!(try_axis(1.0, 3.0, 3).unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    }
}
