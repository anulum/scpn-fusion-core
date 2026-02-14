// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — MPI Domain Scaffolding
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! MPI-oriented domain decomposition scaffolding.
//!
//! This module defines deterministic domain partition metadata and halo
//! packing/exchange primitives that can be wired to rsmpi in a later phase.

use fusion_types::error::{FusionError, FusionResult};
use ndarray::{s, Array2};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainSlice {
    pub rank: usize,
    pub nranks: usize,
    pub global_nz: usize,
    pub local_nz: usize,
    pub halo: usize,
    pub z_start: usize,
    pub z_end: usize,
}

impl DomainSlice {
    pub fn has_upper_neighbor(&self) -> bool {
        self.rank > 0
    }

    pub fn has_lower_neighbor(&self) -> bool {
        self.rank + 1 < self.nranks
    }
}

pub fn decompose_z(global_nz: usize, nranks: usize, halo: usize) -> FusionResult<Vec<DomainSlice>> {
    if global_nz < 2 {
        return Err(FusionError::PhysicsViolation(
            "MPI decomposition requires global_nz >= 2".to_string(),
        ));
    }
    if nranks < 1 {
        return Err(FusionError::PhysicsViolation(
            "MPI decomposition requires nranks >= 1".to_string(),
        ));
    }
    if nranks > global_nz {
        return Err(FusionError::PhysicsViolation(format!(
            "Cannot split global_nz={global_nz} across nranks={nranks}"
        )));
    }

    let base = global_nz / nranks;
    let rem = global_nz % nranks;
    let mut out = Vec::with_capacity(nranks);
    let mut cursor = 0usize;
    for rank in 0..nranks {
        let local_nz = base + usize::from(rank < rem);
        let z_start = cursor;
        let z_end = z_start + local_nz;
        cursor = z_end;
        out.push(DomainSlice {
            rank,
            nranks,
            global_nz,
            local_nz,
            halo,
            z_start,
            z_end,
        });
    }
    Ok(out)
}

pub fn pack_halo_rows(
    local: &Array2<f64>,
    halo: usize,
) -> FusionResult<(Array2<f64>, Array2<f64>)> {
    if halo == 0 {
        return Err(FusionError::PhysicsViolation(
            "Halo width must be >= 1".to_string(),
        ));
    }
    if local.nrows() <= 2 * halo {
        return Err(FusionError::PhysicsViolation(format!(
            "Local block has insufficient rows {} for halo={halo}",
            local.nrows()
        )));
    }
    if local.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Local block contains non-finite values".to_string(),
        ));
    }
    let top = local.slice(s![halo..(2 * halo), ..]).to_owned();
    let bottom = local
        .slice(s![(local.nrows() - 2 * halo)..(local.nrows() - halo), ..])
        .to_owned();
    if top.iter().any(|v| !v.is_finite()) || bottom.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Packed halo rows contain non-finite values".to_string(),
        ));
    }
    Ok((top, bottom))
}

pub fn apply_halo_rows(
    local: &mut Array2<f64>,
    halo: usize,
    recv_top: Option<&Array2<f64>>,
    recv_bottom: Option<&Array2<f64>>,
) -> FusionResult<()> {
    if halo == 0 {
        return Err(FusionError::PhysicsViolation(
            "Halo width must be >= 1".to_string(),
        ));
    }
    if local.nrows() <= 2 * halo {
        return Err(FusionError::PhysicsViolation(format!(
            "Local block has insufficient rows {} for halo={halo}",
            local.nrows()
        )));
    }
    if local.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Local block contains non-finite values".to_string(),
        ));
    }
    if let Some(top) = recv_top {
        if top.dim() != (halo, local.ncols()) {
            return Err(FusionError::PhysicsViolation(format!(
                "Top halo shape mismatch: expected ({halo}, {}), got {:?}",
                local.ncols(),
                top.dim()
            )));
        }
        if top.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::PhysicsViolation(
                "Top halo contains non-finite values".to_string(),
            ));
        }
        local.slice_mut(s![0..halo, ..]).assign(top);
    }
    if let Some(bottom) = recv_bottom {
        if bottom.dim() != (halo, local.ncols()) {
            return Err(FusionError::PhysicsViolation(format!(
                "Bottom halo shape mismatch: expected ({halo}, {}), got {:?}",
                local.ncols(),
                bottom.dim()
            )));
        }
        if bottom.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::PhysicsViolation(
                "Bottom halo contains non-finite values".to_string(),
            ));
        }
        let n = local.nrows();
        local.slice_mut(s![(n - halo)..n, ..]).assign(bottom);
    }
    if local.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Applying halo rows produced non-finite values".to_string(),
        ));
    }
    Ok(())
}

pub fn split_with_halo(
    global: &Array2<f64>,
    slices: &[DomainSlice],
) -> FusionResult<Vec<Array2<f64>>> {
    if slices.is_empty() {
        return Err(FusionError::PhysicsViolation(
            "No slices provided for split_with_halo".to_string(),
        ));
    }
    if global.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Global array contains non-finite values".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(slices.len());
    for sdef in slices {
        if sdef.global_nz != global.nrows() {
            return Err(FusionError::PhysicsViolation(format!(
                "Slice/global mismatch: slice.global_nz={} global.nrows()={}",
                sdef.global_nz,
                global.nrows()
            )));
        }
        if sdef.z_start >= sdef.z_end || sdef.z_end > sdef.global_nz {
            return Err(FusionError::PhysicsViolation(format!(
                "Invalid slice bounds z_start={} z_end={} global_nz={}",
                sdef.z_start, sdef.z_end, sdef.global_nz
            )));
        }
        let start = sdef.z_start.saturating_sub(sdef.halo);
        let end = (sdef.z_end + sdef.halo).min(sdef.global_nz);
        let mut local = Array2::zeros((end - start, global.ncols()));
        local.assign(&global.slice(s![start..end, ..]));
        if local.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::PhysicsViolation(
                "Split local block contains non-finite values".to_string(),
            ));
        }
        out.push(local);
    }
    Ok(out)
}

pub fn stitch_without_halo(
    locals: &[Array2<f64>],
    slices: &[DomainSlice],
    ncols: usize,
) -> FusionResult<Array2<f64>> {
    if locals.len() != slices.len() {
        return Err(FusionError::PhysicsViolation(format!(
            "locals/slices mismatch: {} vs {}",
            locals.len(),
            slices.len()
        )));
    }
    if slices.is_empty() {
        return Err(FusionError::PhysicsViolation(
            "No slices provided for stitch_without_halo".to_string(),
        ));
    }
    let global_nz = slices
        .last()
        .map(|s| s.global_nz)
        .ok_or_else(|| FusionError::PhysicsViolation("No slices provided".to_string()))?;
    let mut global = Array2::zeros((global_nz, ncols));
    for (local, sdef) in locals.iter().zip(slices.iter()) {
        if local.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::PhysicsViolation(
                "Local block contains non-finite values".to_string(),
            ));
        }
        if local.ncols() != ncols {
            return Err(FusionError::PhysicsViolation(format!(
                "Local ncols mismatch: expected {ncols}, got {}",
                local.ncols()
            )));
        }
        let core_start = usize::from(sdef.z_start > 0) * sdef.halo;
        let core_end = core_start + sdef.local_nz;
        if core_end > local.nrows() {
            return Err(FusionError::PhysicsViolation(format!(
                "Local core range out of bounds: rows={}, core_end={core_end}",
                local.nrows()
            )));
        }
        global
            .slice_mut(s![sdef.z_start..sdef.z_end, ..])
            .assign(&local.slice(s![core_start..core_end, ..]));
    }
    if global.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Stitched global array contains non-finite values".to_string(),
        ));
    }
    Ok(global)
}

pub fn serial_halo_exchange(
    locals: &mut [Array2<f64>],
    slices: &[DomainSlice],
) -> FusionResult<()> {
    if locals.len() != slices.len() {
        return Err(FusionError::PhysicsViolation(format!(
            "locals/slices mismatch: {} vs {}",
            locals.len(),
            slices.len()
        )));
    }
    let mut top_send: Vec<Option<Array2<f64>>> = vec![None; locals.len()];
    let mut bottom_send: Vec<Option<Array2<f64>>> = vec![None; locals.len()];
    for (i, (local, sdef)) in locals.iter().zip(slices.iter()).enumerate() {
        if local.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::PhysicsViolation(format!(
                "Local block at index {i} contains non-finite values"
            )));
        }
        if sdef.halo == 0 {
            continue;
        }
        let (top, bottom) = pack_halo_rows(local, sdef.halo)?;
        top_send[i] = Some(top);
        bottom_send[i] = Some(bottom);
    }

    for i in 0..locals.len() {
        let halo = slices[i].halo;
        if halo == 0 {
            continue;
        }
        let recv_top = if i > 0 {
            bottom_send[i - 1].as_ref()
        } else {
            None
        };
        let recv_bottom = if i + 1 < locals.len() {
            top_send[i + 1].as_ref()
        } else {
            None
        };
        apply_halo_rows(&mut locals[i], halo, recv_top, recv_bottom)?;
    }
    if locals.iter().any(|arr| arr.iter().any(|v| !v.is_finite())) {
        return Err(FusionError::PhysicsViolation(
            "Serial halo exchange produced non-finite values".to_string(),
        ));
    }
    Ok(())
}

pub fn l2_norm_delta(a: &Array2<f64>, b: &Array2<f64>) -> FusionResult<f64> {
    if a.dim() != b.dim() {
        return Err(FusionError::PhysicsViolation(format!(
            "l2_norm_delta shape mismatch {:?} vs {:?}",
            a.dim(),
            b.dim()
        )));
    }
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "l2_norm_delta inputs must be finite".to_string(),
        ));
    }
    let mut accum = 0.0f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        accum += d * d;
    }
    let out = accum.sqrt();
    if !out.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "l2_norm_delta produced non-finite result".to_string(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grid(nz: usize, nr: usize) -> Array2<f64> {
        Array2::from_shape_fn((nz, nr), |(i, j)| (i as f64) * 10.0 + j as f64)
    }

    #[test]
    fn test_decompose_z_covers_domain() {
        let slices = decompose_z(17, 4, 1).expect("decomposition must succeed");
        assert_eq!(slices.len(), 4);
        assert_eq!(slices[0].z_start, 0);
        assert_eq!(slices.last().expect("slice expected").z_end, 17);
        let covered: usize = slices.iter().map(|s| s.local_nz).sum();
        assert_eq!(covered, 17);
    }

    #[test]
    fn test_serial_halo_exchange_and_stitch_roundtrip() {
        let global = sample_grid(24, 9);
        let slices = decompose_z(global.nrows(), 3, 1).expect("decompose");
        let mut locals = split_with_halo(&global, &slices).expect("split");
        serial_halo_exchange(&mut locals, &slices).expect("exchange");
        let stitched = stitch_without_halo(&locals, &slices, global.ncols()).expect("stitch");
        let delta = l2_norm_delta(&stitched, &global).expect("delta");
        assert!(
            delta < 1e-12,
            "Serial halo exchange should preserve core rows"
        );
    }

    #[test]
    fn test_pack_halo_errors_for_small_local_block() {
        let local = Array2::zeros((2, 4));
        let err = pack_halo_rows(&local, 1).expect_err("small local must error");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("insufficient rows"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_apply_halo_shape_guard() {
        let mut local = Array2::zeros((6, 4));
        let bad_top = Array2::zeros((2, 5));
        let err = apply_halo_rows(&mut local, 1, Some(&bad_top), None).expect_err("shape mismatch");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("shape mismatch")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_l2_norm_delta_zero_for_identical_arrays() {
        let a = sample_grid(8, 8);
        let b = a.clone();
        let d = l2_norm_delta(&a, &b).expect("delta");
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn test_mpi_domain_rejects_non_finite_inputs() {
        let mut local = sample_grid(8, 4);
        local[[2, 2]] = f64::NAN;
        let err = pack_halo_rows(&local, 1).expect_err("non-finite local should fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("non-finite")),
            other => panic!("Unexpected error: {other:?}"),
        }

        let mut a = sample_grid(4, 4);
        let b = sample_grid(4, 4);
        a[[0, 0]] = f64::INFINITY;
        let err = l2_norm_delta(&a, &b).expect_err("non-finite delta input should fail");
        match err {
            FusionError::PhysicsViolation(msg) => assert!(msg.contains("finite")),
            other => panic!("Unexpected error: {other:?}"),
        }
    }
}
