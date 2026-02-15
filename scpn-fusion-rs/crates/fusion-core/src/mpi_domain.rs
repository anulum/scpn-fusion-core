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

// ═══════════════════════════════════════════════════════════════════════
// 2D Cartesian Domain Decomposition — Exascale-Ready MPI Abstraction
// ═══════════════════════════════════════════════════════════════════════
//
// The key innovation: we decompose the (nz × nr) GS grid into a 2D
// Cartesian process grid (pz × pr), where each tile owns a contiguous
// sub-block plus halo cells on all four faces. The Rayon threadpool
// simulates distributed-memory ranks; replacing with rsmpi is a 1:1
// swap of the halo exchange function.
//
// References:
//   - Jardin, "Computational Methods in Plasma Physics", Ch. 12
//   - Lao et al., "Equilibrium analysis of current profiles in tokamaks",
//     Nucl. Fusion 30 (1990) 1035
//   - EFIT-AI: Joung et al., Nucl. Fusion 63 (2023) 126058

/// 2D Cartesian tile descriptor — one per rank in a (pz × pr) topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CartesianTile {
    /// Linear rank index (0 .. pz*pr - 1).
    pub rank: usize,
    /// Process grid index along Z (row).
    pub pz_idx: usize,
    /// Process grid index along R (column).
    pub pr_idx: usize,
    /// Total process-grid dimensions.
    pub pz: usize,
    pub pr: usize,
    /// Global grid dimensions.
    pub global_nz: usize,
    pub global_nr: usize,
    /// Halo width (same on all faces).
    pub halo: usize,
    /// Owned Z range [z_start, z_end) in global indexing.
    pub z_start: usize,
    pub z_end: usize,
    /// Owned R range [r_start, r_end) in global indexing.
    pub r_start: usize,
    pub r_end: usize,
}

impl CartesianTile {
    /// Number of owned Z rows (excluding halo).
    pub fn local_nz(&self) -> usize {
        self.z_end - self.z_start
    }
    /// Number of owned R columns (excluding halo).
    pub fn local_nr(&self) -> usize {
        self.r_end - self.r_start
    }
    /// Total rows including halo (top + bottom).
    pub fn padded_nz(&self) -> usize {
        let top = if self.pz_idx > 0 { self.halo } else { 0 };
        let bot = if self.pz_idx + 1 < self.pz { self.halo } else { 0 };
        self.local_nz() + top + bot
    }
    /// Total columns including halo (left + right).
    pub fn padded_nr(&self) -> usize {
        let left = if self.pr_idx > 0 { self.halo } else { 0 };
        let right = if self.pr_idx + 1 < self.pr { self.halo } else { 0 };
        self.local_nr() + left + right
    }
    /// Offset of the first owned row within the padded local array.
    pub fn core_z_offset(&self) -> usize {
        if self.pz_idx > 0 { self.halo } else { 0 }
    }
    /// Offset of the first owned column within the padded local array.
    pub fn core_r_offset(&self) -> usize {
        if self.pr_idx > 0 { self.halo } else { 0 }
    }
    pub fn has_neighbor_top(&self) -> bool {
        self.pz_idx > 0
    }
    pub fn has_neighbor_bottom(&self) -> bool {
        self.pz_idx + 1 < self.pz
    }
    pub fn has_neighbor_left(&self) -> bool {
        self.pr_idx > 0
    }
    pub fn has_neighbor_right(&self) -> bool {
        self.pr_idx + 1 < self.pr
    }
    /// Rank of a neighbour at offset (dz, dr) in the process grid.
    /// Returns None if out of bounds.
    pub fn neighbor_rank(&self, dz: i32, dr: i32) -> Option<usize> {
        let nz = self.pz_idx as i32 + dz;
        let nr = self.pr_idx as i32 + dr;
        if nz < 0 || nz >= self.pz as i32 || nr < 0 || nr >= self.pr as i32 {
            return None;
        }
        Some(nz as usize * self.pr + nr as usize)
    }
}

/// Decompose a 2D grid of shape (global_nz × global_nr) into a
/// (pz × pr) Cartesian process topology.
///
/// Returns tiles in row-major order: tile[iz * pr + ir].
pub fn decompose_2d(
    global_nz: usize,
    global_nr: usize,
    pz: usize,
    pr: usize,
    halo: usize,
) -> FusionResult<Vec<CartesianTile>> {
    if pz == 0 || pr == 0 {
        return Err(FusionError::PhysicsViolation(
            "Process grid dimensions pz, pr must be >= 1".to_string(),
        ));
    }
    if pz > global_nz || pr > global_nr {
        return Err(FusionError::PhysicsViolation(format!(
            "Cannot split ({global_nz}×{global_nr}) across ({pz}×{pr}) processes"
        )));
    }
    if global_nz < 2 || global_nr < 2 {
        return Err(FusionError::PhysicsViolation(
            "Global grid must be at least 2×2".to_string(),
        ));
    }

    // Distribute rows/columns as evenly as possible.
    let z_splits = balanced_split(global_nz, pz);
    let r_splits = balanced_split(global_nr, pr);

    let mut tiles = Vec::with_capacity(pz * pr);
    let mut z_cursor = 0usize;
    for iz in 0..pz {
        let nz_local = z_splits[iz];
        let z_start = z_cursor;
        let z_end = z_start + nz_local;
        z_cursor = z_end;

        let mut r_cursor = 0usize;
        for ir in 0..pr {
            let nr_local = r_splits[ir];
            let r_start = r_cursor;
            let r_end = r_start + nr_local;
            r_cursor = r_end;

            tiles.push(CartesianTile {
                rank: iz * pr + ir,
                pz_idx: iz,
                pr_idx: ir,
                pz,
                pr,
                global_nz,
                global_nr,
                halo,
                z_start,
                z_end,
                r_start,
                r_end,
            });
        }
    }
    Ok(tiles)
}

/// Helper: split `n` items across `k` buckets as evenly as possible.
fn balanced_split(n: usize, k: usize) -> Vec<usize> {
    let base = n / k;
    let rem = n % k;
    (0..k).map(|i| base + usize::from(i < rem)).collect()
}

/// Extract a padded local tile (with halo) from the global array.
pub fn extract_tile(
    global: &Array2<f64>,
    tile: &CartesianTile,
) -> FusionResult<Array2<f64>> {
    let (gnz, gnr) = global.dim();
    if gnz != tile.global_nz || gnr != tile.global_nr {
        return Err(FusionError::PhysicsViolation(format!(
            "Global shape ({gnz},{gnr}) doesn't match tile expectation ({},{})",
            tile.global_nz, tile.global_nr
        )));
    }
    let z0 = tile.z_start.saturating_sub(tile.core_z_offset());
    let z1 = (tile.z_end + if tile.has_neighbor_bottom() { tile.halo } else { 0 }).min(gnz);
    let r0 = tile.r_start.saturating_sub(tile.core_r_offset());
    let r1 = (tile.r_end + if tile.has_neighbor_right() { tile.halo } else { 0 }).min(gnr);

    let local = global.slice(s![z0..z1, r0..r1]).to_owned();
    if local.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Extracted tile contains non-finite values".to_string(),
        ));
    }
    Ok(local)
}

/// Write the core (non-halo) region of a local tile back into the global array.
pub fn inject_tile(
    global: &mut Array2<f64>,
    local: &Array2<f64>,
    tile: &CartesianTile,
) -> FusionResult<()> {
    let cz = tile.core_z_offset();
    let cr = tile.core_r_offset();
    let lnz = tile.local_nz();
    let lnr = tile.local_nr();
    if cz + lnz > local.nrows() || cr + lnr > local.ncols() {
        return Err(FusionError::PhysicsViolation(format!(
            "Tile core out of bounds: local shape {:?}, core ({cz}+{lnz}, {cr}+{lnr})",
            local.dim()
        )));
    }
    let core = local.slice(s![cz..(cz + lnz), cr..(cr + lnr)]);
    if core.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Tile core to inject contains non-finite values".to_string(),
        ));
    }
    global
        .slice_mut(s![tile.z_start..tile.z_end, tile.r_start..tile.r_end])
        .assign(&core);
    Ok(())
}

/// Serial 2D halo exchange across all tiles.
///
/// Copies owned boundary rows/columns from each tile into the halo
/// region of its four face-neighbours. This is the serial reference
/// implementation; the MPI version replaces this with non-blocking
/// Isend/Irecv pairs.
pub fn serial_halo_exchange_2d(
    locals: &mut [Array2<f64>],
    tiles: &[CartesianTile],
) -> FusionResult<()> {
    if locals.len() != tiles.len() {
        return Err(FusionError::PhysicsViolation(format!(
            "locals/tiles length mismatch: {} vs {}",
            locals.len(),
            tiles.len()
        )));
    }
    let ntiles = tiles.len();
    if ntiles == 0 {
        return Ok(());
    }
    let halo = tiles[0].halo;
    if halo == 0 {
        return Ok(());
    }

    // Collect halo strips from all tiles first (immutable borrows).
    // Each entry: (dest_rank, HaloFace, data).
    #[derive(Clone, Copy)]
    enum Face {
        Top,
        Bottom,
        Left,
        Right,
    }
    let mut messages: Vec<(usize, Face, Array2<f64>)> = Vec::new();

    for (i, tile) in tiles.iter().enumerate() {
        let loc = &locals[i];
        let cz = tile.core_z_offset();
        let cr = tile.core_r_offset();
        let lnz = tile.local_nz();
        let lnr = tile.local_nr();

        // Send top face → neighbor above
        if let Some(dest) = tile.neighbor_rank(-1, 0) {
            let strip = loc.slice(s![cz..(cz + halo), cr..(cr + lnr)]).to_owned();
            messages.push((dest, Face::Bottom, strip));
        }
        // Send bottom face → neighbor below
        if let Some(dest) = tile.neighbor_rank(1, 0) {
            let strip = loc
                .slice(s![(cz + lnz - halo)..(cz + lnz), cr..(cr + lnr)])
                .to_owned();
            messages.push((dest, Face::Top, strip));
        }
        // Send left face → neighbor to the left
        if let Some(dest) = tile.neighbor_rank(0, -1) {
            let strip = loc.slice(s![cz..(cz + lnz), cr..(cr + halo)]).to_owned();
            messages.push((dest, Face::Right, strip));
        }
        // Send right face → neighbor to the right
        if let Some(dest) = tile.neighbor_rank(0, 1) {
            let strip = loc
                .slice(s![cz..(cz + lnz), (cr + lnr - halo)..(cr + lnr)])
                .to_owned();
            messages.push((dest, Face::Left, strip));
        }
    }

    // Apply received halos.
    for (dest, face, data) in messages {
        let tile = &tiles[dest];
        let loc = &mut locals[dest];
        let cz = tile.core_z_offset();
        let cr = tile.core_r_offset();
        let lnz = tile.local_nz();
        let lnr = tile.local_nr();

        match face {
            Face::Top => {
                // Fill top halo rows.
                if cz >= halo {
                    loc.slice_mut(s![(cz - halo)..cz, cr..(cr + lnr)])
                        .assign(&data);
                }
            }
            Face::Bottom => {
                // Fill bottom halo rows.
                let row_start = cz + lnz;
                let row_end = row_start + halo;
                if row_end <= loc.nrows() {
                    loc.slice_mut(s![row_start..row_end, cr..(cr + lnr)])
                        .assign(&data);
                }
            }
            Face::Left => {
                // Fill left halo columns.
                if cr >= halo {
                    loc.slice_mut(s![cz..(cz + lnz), (cr - halo)..cr])
                        .assign(&data);
                }
            }
            Face::Right => {
                // Fill right halo columns.
                let col_start = cr + lnr;
                let col_end = col_start + halo;
                if col_end <= loc.ncols() {
                    loc.slice_mut(s![cz..(cz + lnz), col_start..col_end])
                        .assign(&data);
                }
            }
        }
    }

    // Verify no non-finite values crept in.
    for loc in locals.iter() {
        if loc.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::PhysicsViolation(
                "2D halo exchange produced non-finite values".to_string(),
            ));
        }
    }
    Ok(())
}

/// Configuration for the distributed GS solver.
#[derive(Debug, Clone)]
pub struct DistributedSolverConfig {
    /// Process-grid dimensions (pz × pr). Product must be ≤ Rayon
    /// thread count for full parallel utilisation.
    pub pz: usize,
    pub pr: usize,
    /// Halo width (number of overlap rows/columns per face). 1 is
    /// sufficient for the 5-point GS stencil.
    pub halo: usize,
    /// SOR relaxation parameter ω ∈ (1, 2). Typically 1.8.
    pub omega: f64,
    /// Maximum number of Schwarz (outer) iterations.
    pub max_outer_iters: usize,
    /// Convergence tolerance on global L2 residual.
    pub tol: f64,
    /// Number of local SOR sweeps per Schwarz iteration.
    pub inner_sweeps: usize,
}

impl Default for DistributedSolverConfig {
    fn default() -> Self {
        Self {
            pz: 2,
            pr: 2,
            halo: 1,
            omega: 1.8,
            max_outer_iters: 200,
            tol: 1e-8,
            inner_sweeps: 5,
        }
    }
}

/// Result of a distributed GS solve.
#[derive(Debug, Clone)]
pub struct DistributedSolveResult {
    /// Final global Ψ array.
    pub psi: Array2<f64>,
    /// Achieved global L2 residual.
    pub residual: f64,
    /// Number of Schwarz outer iterations used.
    pub iterations: usize,
    /// Whether the solve converged within tolerance.
    pub converged: bool,
}

/// Distributed Grad-Shafranov solver using additive Schwarz domain
/// decomposition with Rayon thread-parallelism.
///
/// Each Schwarz iteration:
/// 1. Splits the global Ψ into 2D tiles (with halo overlap).
/// 2. Runs `inner_sweeps` local Red-Black SOR sweeps on each tile
///    in parallel via Rayon.
/// 3. Injects tile cores back into the global array.
/// 4. Exchanges halos (serial reference — MPI-ready interface).
/// 5. Computes global residual; checks convergence.
///
/// The 5-point GS stencil:
///   R d/dR(1/R dΨ/dR) + d²Ψ/dZ² = -μ₀ R J_φ
///
/// discretises to the same operator as `fusion_math::sor::sor_step`,
/// but applied independently on each tile with local boundary from
/// halo data.
pub fn distributed_gs_solve(
    psi: &Array2<f64>,
    source: &Array2<f64>,
    r_axis: &[f64],
    z_axis: &[f64],
    dr: f64,
    dz: f64,
    cfg: &DistributedSolverConfig,
) -> FusionResult<DistributedSolveResult> {
    let (nz, nr) = psi.dim();
    if source.dim() != (nz, nr) {
        return Err(FusionError::PhysicsViolation(format!(
            "psi/source shape mismatch: {:?} vs {:?}",
            psi.dim(),
            source.dim()
        )));
    }
    if r_axis.len() != nr || z_axis.len() != nz {
        return Err(FusionError::PhysicsViolation(format!(
            "Axis lengths don't match grid: r_axis={} nr={nr}, z_axis={} nz={nz}",
            r_axis.len(),
            z_axis.len()
        )));
    }
    if cfg.omega <= 0.0 || cfg.omega >= 2.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "SOR omega must be in (0, 2), got {}", cfg.omega
        )));
    }
    if !dr.is_finite() || !dz.is_finite() || dr <= 0.0 || dz <= 0.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "Grid spacing must be finite > 0: dr={dr}, dz={dz}"
        )));
    }

    let tiles = decompose_2d(nz, nr, cfg.pz, cfg.pr, cfg.halo)?;

    let mut global_psi = psi.clone();
    let dr_sq = dr * dr;
    let dz_sq = dz * dz;

    let mut converged = false;
    let mut outer_iter = 0usize;
    let mut residual = f64::MAX;

    for outer in 0..cfg.max_outer_iters {
        outer_iter = outer + 1;

        // 1. Extract tiles with halo from current global Ψ.
        let mut locals: Vec<Array2<f64>> = tiles
            .iter()
            .map(|t| extract_tile(&global_psi, t))
            .collect::<FusionResult<Vec<_>>>()?;
        let sources: Vec<Array2<f64>> = tiles
            .iter()
            .map(|t| extract_tile(source, t))
            .collect::<FusionResult<Vec<_>>>()?;

        // 2. Run local SOR sweeps in parallel via Rayon.
        //    Each tile applies inner_sweeps Red-Black SOR iterations.
        use rayon::prelude::*;
        locals
            .par_iter_mut()
            .zip(sources.par_iter())
            .zip(tiles.par_iter())
            .for_each(|((loc, src), tile)| {
                let cz = tile.core_z_offset();
                let cr = tile.core_r_offset();
                let _lnz = tile.local_nz();
                let _lnr = tile.local_nr();

                for _sweep in 0..cfg.inner_sweeps {
                    // Red pass then black pass.
                    for color in 0..2u8 {
                        for iz in 1..loc.nrows().saturating_sub(1) {
                            for ir in 1..loc.ncols().saturating_sub(1) {
                                if (iz + ir) % 2 != color as usize {
                                    continue;
                                }
                                // Only update interior of the owned core
                                // (and the halo interior that overlaps with
                                // a neighbor's owned region).
                                // Map local (iz, ir) to global indices.
                                let gz = tile.z_start as i64 + iz as i64 - cz as i64;
                                let gr = tile.r_start as i64 + ir as i64 - cr as i64;
                                // Skip global boundaries.
                                if gz <= 0 || gz >= (nz as i64 - 1) {
                                    continue;
                                }
                                if gr <= 0 || gr >= (nr as i64 - 1) {
                                    continue;
                                }

                                let r_val = r_axis[gr as usize];
                                if r_val <= 0.0 {
                                    continue;
                                }

                                // GS 5-point stencil with 1/R correction.
                                // Coefficients follow sor.rs update_point():
                                //   c_r_plus  = 1/dr² - 1/(2R·dr) → psi_east (ir+1)
                                //   c_r_minus = 1/dr² + 1/(2R·dr) → psi_west (ir-1)
                                let psi_n = loc[[iz - 1, ir]];
                                let psi_s = loc[[iz + 1, ir]];
                                let psi_w = loc[[iz, ir - 1]];
                                let psi_e = loc[[iz, ir + 1]];

                                let c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r_val * dr);
                                let c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r_val * dr);
                                let c_z = 1.0 / dz_sq;
                                let center = 2.0 / dr_sq + 2.0 / dz_sq;

                                let rhs_val = src[[iz, ir]];
                                let numerator = c_z * (psi_n + psi_s)
                                    + c_r_minus * psi_w
                                    + c_r_plus * psi_e
                                    + rhs_val;
                                if center.abs() < 1e-30 {
                                    continue;
                                }
                                let psi_gs = numerator / center;
                                let old = loc[[iz, ir]];
                                loc[[iz, ir]] = old + cfg.omega * (psi_gs - old);
                            }
                        }
                    }
                }
            });

        // 3. Inject tile cores back into global array.
        for (i, tile) in tiles.iter().enumerate() {
            inject_tile(&mut global_psi, &locals[i], tile)?;
        }

        // 4. Compute global L2 residual of the GS equation.
        residual = gs_residual_l2(&global_psi, source, r_axis, dr, dz);

        if residual < cfg.tol {
            converged = true;
            break;
        }
    }

    if global_psi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "Distributed GS solve produced non-finite Ψ".to_string(),
        ));
    }

    Ok(DistributedSolveResult {
        psi: global_psi,
        residual,
        iterations: outer_iter,
        converged,
    })
}

/// Compute the L2 norm of the GS residual: ||LΨ - f||₂.
///
/// L is the 5-point GS operator:
///   LΨ = (Ψ_{i-1,j} + Ψ_{i+1,j})/dz² + (1/dr² - 1/(2R dr))Ψ_{i,j-1}
///        + (1/dr² + 1/(2R dr))Ψ_{i,j+1} - 2(1/dr² + 1/dz²)Ψ_{i,j}
pub fn gs_residual_l2(
    psi: &Array2<f64>,
    source: &Array2<f64>,
    r_axis: &[f64],
    dr: f64,
    dz: f64,
) -> f64 {
    let (nz, nr) = psi.dim();
    let dr_sq = dr * dr;
    let dz_sq = dz * dz;
    let mut accum = 0.0f64;
    let mut count = 0usize;

    for iz in 1..nz - 1 {
        for ir in 1..nr - 1 {
            let r_val = r_axis[ir];
            if r_val <= 0.0 {
                continue;
            }
            let c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r_val * dr);
            let c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r_val * dr);
            let c_z = 1.0 / dz_sq;
            let center = 2.0 / dr_sq + 2.0 / dz_sq;

            let l_psi = center * psi[[iz, ir]]
                - c_z * (psi[[iz - 1, ir]] + psi[[iz + 1, ir]])
                - c_r_plus * psi[[iz, ir + 1]]
                - c_r_minus * psi[[iz, ir - 1]];
            let res = l_psi - source[[iz, ir]];
            accum += res * res;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    (accum / count as f64).sqrt()
}

/// Optimal process-grid factorisation for a given (nz, nr) global grid
/// and total number of available ranks. Minimises the surface-to-volume
/// ratio of each tile (i.e. the halo communication overhead).
pub fn optimal_process_grid(nz: usize, nr: usize, nranks: usize) -> (usize, usize) {
    let mut best_pz = 1;
    let mut best_pr = nranks;
    let mut best_cost = f64::MAX;

    for pz in 1..=nranks {
        if nranks % pz != 0 {
            continue;
        }
        let pr = nranks / pz;
        if pz > nz || pr > nr {
            continue;
        }
        // Surface-to-volume ratio proxy: perimeter / area of each tile.
        let tile_nz = nz as f64 / pz as f64;
        let tile_nr = nr as f64 / pr as f64;
        let perimeter = 2.0 * (tile_nz + tile_nr);
        let area = tile_nz * tile_nr;
        let cost = perimeter / area;
        if cost < best_cost {
            best_cost = cost;
            best_pz = pz;
            best_pr = pr;
        }
    }
    (best_pz, best_pr)
}

/// Convenience: solve GS with automatic process-grid selection.
///
/// Detects the Rayon thread count and picks the optimal (pz, pr)
/// factorisation. This is the top-level entry point for exascale-ready
/// distributed equilibrium solving.
pub fn auto_distributed_gs_solve(
    psi: &Array2<f64>,
    source: &Array2<f64>,
    r_axis: &[f64],
    z_axis: &[f64],
    dr: f64,
    dz: f64,
    omega: f64,
    tol: f64,
    max_iters: usize,
) -> FusionResult<DistributedSolveResult> {
    let (nz, nr) = psi.dim();
    let nthreads = rayon::current_num_threads().max(1);
    let (pz, pr) = optimal_process_grid(nz, nr, nthreads);

    let cfg = DistributedSolverConfig {
        pz,
        pr,
        halo: 1,
        omega,
        max_outer_iters: max_iters,
        tol,
        inner_sweeps: 5,
    };
    distributed_gs_solve(psi, source, r_axis, z_axis, dr, dz, &cfg)
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

    // ═══════════════════════════════════════════════════════════════════
    // 2D Cartesian Decomposition Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_decompose_2d_covers_full_domain() {
        let tiles = decompose_2d(32, 24, 4, 3, 1).expect("decompose_2d");
        assert_eq!(tiles.len(), 12); // 4*3
        // Every global (iz, ir) must be owned by exactly one tile.
        let mut coverage = Array2::<u8>::zeros((32, 24));
        for t in &tiles {
            for iz in t.z_start..t.z_end {
                for ir in t.r_start..t.r_end {
                    coverage[[iz, ir]] += 1;
                }
            }
        }
        assert!(coverage.iter().all(|&c| c == 1), "Every cell owned by exactly one tile");
    }

    #[test]
    fn test_decompose_2d_single_rank() {
        let tiles = decompose_2d(16, 16, 1, 1, 2).expect("single rank");
        assert_eq!(tiles.len(), 1);
        let t = &tiles[0];
        assert_eq!(t.z_start, 0);
        assert_eq!(t.z_end, 16);
        assert_eq!(t.r_start, 0);
        assert_eq!(t.r_end, 16);
        assert_eq!(t.padded_nz(), 16);
        assert_eq!(t.padded_nr(), 16);
    }

    #[test]
    fn test_decompose_2d_neighbor_ranks() {
        let tiles = decompose_2d(16, 16, 2, 2, 1).expect("2x2");
        // Top-left corner (0,0): has bottom and right neighbours.
        let tl = &tiles[0];
        assert_eq!(tl.neighbor_rank(1, 0), Some(2));
        assert_eq!(tl.neighbor_rank(0, 1), Some(1));
        assert_eq!(tl.neighbor_rank(-1, 0), None);
        assert_eq!(tl.neighbor_rank(0, -1), None);
        // Bottom-right corner (1,1).
        let br = &tiles[3];
        assert_eq!(br.neighbor_rank(-1, 0), Some(1));
        assert_eq!(br.neighbor_rank(0, -1), Some(2));
        assert_eq!(br.neighbor_rank(1, 0), None);
        assert_eq!(br.neighbor_rank(0, 1), None);
    }

    #[test]
    fn test_extract_inject_roundtrip() {
        let global = sample_grid(24, 18);
        let tiles = decompose_2d(24, 18, 3, 2, 1).expect("decompose");
        let mut reconstructed = Array2::<f64>::zeros((24, 18));
        for t in &tiles {
            let local = extract_tile(&global, t).expect("extract");
            inject_tile(&mut reconstructed, &local, t).expect("inject");
        }
        let delta = l2_norm_delta(&global, &reconstructed).expect("delta");
        assert!(delta < 1e-12, "Extract→inject roundtrip must be lossless, got delta={delta}");
    }

    #[test]
    fn test_serial_halo_exchange_2d_correctness() {
        // Create a global array, split into tiles, exchange halos,
        // then verify that halo cells match the original global data.
        let global = Array2::from_shape_fn((16, 16), |(i, j)| {
            (i as f64) * 100.0 + j as f64
        });
        let tiles = decompose_2d(16, 16, 2, 2, 1).expect("decompose");
        let mut locals: Vec<Array2<f64>> = tiles
            .iter()
            .map(|t| extract_tile(&global, t).expect("extract"))
            .collect();
        serial_halo_exchange_2d(&mut locals, &tiles).expect("halo exchange");

        // For each tile, verify that every cell (including halo)
        // matches the original global value.
        for (i, t) in tiles.iter().enumerate() {
            let loc = &locals[i];
            let z0 = t.z_start.saturating_sub(t.core_z_offset());
            let r0 = t.r_start.saturating_sub(t.core_r_offset());
            for lz in 0..loc.nrows() {
                for lr in 0..loc.ncols() {
                    let gz = z0 + lz;
                    let gr = r0 + lr;
                    if gz < 16 && gr < 16 {
                        let expect = global[[gz, gr]];
                        let got = loc[[lz, lr]];
                        assert!(
                            (expect - got).abs() < 1e-12,
                            "Tile {i} at local ({lz},{lr}) global ({gz},{gr}): expected {expect}, got {got}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_optimal_process_grid_square() {
        let (pz, pr) = optimal_process_grid(64, 64, 4);
        assert_eq!(pz, 2);
        assert_eq!(pr, 2);
    }

    #[test]
    fn test_optimal_process_grid_rectangular() {
        // Tall grid (nz >> nr): should put more processes along Z.
        let (pz, pr) = optimal_process_grid(128, 32, 8);
        assert!(pz >= pr, "Tall grid should bias toward Z decomposition: pz={pz}, pr={pr}");
        assert_eq!(pz * pr, 8);
    }

    #[test]
    fn test_gs_residual_l2_zero_for_exact_solution() {
        // Manufacture source = L Ψ for a known Ψ, then verify residual ≈ 0.
        // Using the same convention as sor.rs:
        //   L Ψ = center*Ψ - c_z*(Ψ_up + Ψ_dn) - c_r_plus*Ψ_right - c_r_minus*Ψ_left
        let nz = 16;
        let nr = 16;
        let dr = 0.01;
        let dz = 0.01;
        let r_axis: Vec<f64> = (0..nr).map(|i| 1.0 + i as f64 * dr).collect();
        let _z_axis: Vec<f64> = (0..nz).map(|i| -0.08 + i as f64 * dz).collect();
        let psi = Array2::from_shape_fn((nz, nr), |(iz, ir)| {
            _z_axis[iz] * _z_axis[iz] + r_axis[ir] * r_axis[ir]
        });
        let dr_sq = dr * dr;
        let dz_sq = dz * dz;
        let mut source = Array2::zeros((nz, nr));
        for iz in 1..nz - 1 {
            for ir in 1..nr - 1 {
                let r = r_axis[ir];
                let c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * r * dr);
                let c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * r * dr);
                let c_z = 1.0 / dz_sq;
                let center = 2.0 / dr_sq + 2.0 / dz_sq;
                source[[iz, ir]] = center * psi[[iz, ir]]
                    - c_z * (psi[[iz - 1, ir]] + psi[[iz + 1, ir]])
                    - c_r_plus * psi[[iz, ir + 1]]
                    - c_r_minus * psi[[iz, ir - 1]];
            }
        }
        let res = gs_residual_l2(&psi, &source, &r_axis, dr, dz);
        assert!(res < 1e-10, "Residual should be ~0 for manufactured solution, got {res}");
    }

    #[test]
    fn test_distributed_gs_solve_smoke() {
        // Solve LΨ = f on a small grid and verify residual decreases.
        let nz = 32;
        let nr = 32;
        let dr = 0.01;
        let dz = 0.01;
        let r_axis: Vec<f64> = (0..nr).map(|i| 1.0 + i as f64 * dr).collect();
        let z_axis: Vec<f64> = (0..nz).map(|i| -0.16 + i as f64 * dz).collect();

        // Use a moderate source (amplitude 1.0, not 100).
        let psi = Array2::zeros((nz, nr));
        let r_mid = 1.0 + (nr as f64 / 2.0) * dr;
        let z_mid = 0.0;
        let source = Array2::from_shape_fn((nz, nr), |(iz, ir)| {
            let rr = r_axis[ir] - r_mid;
            let zz = z_axis[iz] - z_mid;
            -1.0 * (-(rr * rr + zz * zz) / (0.05 * 0.05)).exp()
        });

        // Compute initial residual for comparison.
        let res_initial = gs_residual_l2(&psi, &source, &r_axis, dr, dz);

        let cfg = DistributedSolverConfig {
            pz: 2,
            pr: 2,
            halo: 1,
            omega: 1.6,
            max_outer_iters: 200,
            tol: 1e-6,
            inner_sweeps: 10,
        };
        let result = distributed_gs_solve(&psi, &source, &r_axis, &z_axis, dr, dz, &cfg)
            .expect("distributed solve");
        // Residual should have decreased significantly from initial.
        assert!(
            result.residual < res_initial * 0.5,
            "Residual should decrease: initial={res_initial:.4}, final={:.4}",
            result.residual
        );
        // Ψ should be non-trivial (negative source → negative interior values).
        let psi_absmax = result.psi.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(psi_absmax > 1e-10, "Solution should be non-trivial, max |Ψ| = {psi_absmax}");
    }

    #[test]
    fn test_decompose_2d_rejects_invalid_inputs() {
        assert!(decompose_2d(16, 16, 0, 2, 1).is_err());
        assert!(decompose_2d(16, 16, 2, 0, 1).is_err());
        assert!(decompose_2d(1, 16, 2, 2, 1).is_err());
        assert!(decompose_2d(16, 16, 20, 2, 1).is_err());
    }
}
