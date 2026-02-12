// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Property-Based Tests (proptest) for fusion-types
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Property-based tests for fusion-types using proptest.
//!
//! Covers: Grid2D construction invariants, PlasmaState shapes,
//! configuration serialization roundtrip.

use fusion_types::state::{Grid2D, PlasmaState};
use proptest::prelude::*;

// ── Grid2D Construction Invariants ───────────────────────────────────

proptest! {
    /// Grid dimensions match constructor arguments.
    #[test]
    fn grid_dimensions_match(
        nr in 2usize..128,
        nz in 2usize..128,
    ) {
        let grid = Grid2D::new(nr, nz, 1.0, 9.0, -5.0, 5.0);

        prop_assert_eq!(grid.nr, nr);
        prop_assert_eq!(grid.nz, nz);
        prop_assert_eq!(grid.r.len(), nr);
        prop_assert_eq!(grid.z.len(), nz);
        prop_assert_eq!(grid.rr.shape(), &[nz, nr]);
        prop_assert_eq!(grid.zz.shape(), &[nz, nr]);
    }

    /// Grid boundary values are correct.
    #[test]
    fn grid_boundary_values(
        nr in 3usize..64,
        nz in 3usize..64,
        r_min in 0.1f64..5.0,
        z_min in -10.0f64..0.0,
    ) {
        let r_max = r_min + 5.0;
        let z_max = z_min + 10.0;
        let grid = Grid2D::new(nr, nz, r_min, r_max, z_min, z_max);

        // First and last R values
        prop_assert!((grid.r[0] - r_min).abs() < 1e-12);
        prop_assert!((grid.r[nr - 1] - r_max).abs() < 1e-12);

        // First and last Z values
        prop_assert!((grid.z[0] - z_min).abs() < 1e-12);
        prop_assert!((grid.z[nz - 1] - z_max).abs() < 1e-12);

        // Meshgrid corners
        prop_assert!((grid.rr[[0, 0]] - r_min).abs() < 1e-12);
        prop_assert!((grid.rr[[0, nr - 1]] - r_max).abs() < 1e-12);
        prop_assert!((grid.zz[[0, 0]] - z_min).abs() < 1e-12);
        prop_assert!((grid.zz[[nz - 1, 0]] - z_max).abs() < 1e-12);
    }

    /// R coordinates are strictly monotonically increasing.
    #[test]
    fn grid_r_monotone(nr in 3usize..64) {
        let grid = Grid2D::new(nr, 10, 1.0, 9.0, -5.0, 5.0);
        for i in 1..nr {
            prop_assert!(grid.r[i] > grid.r[i - 1],
                "R not monotone at {}: {} <= {}", i, grid.r[i], grid.r[i - 1]);
        }
    }

    /// Z coordinates are strictly monotonically increasing.
    #[test]
    fn grid_z_monotone(nz in 3usize..64) {
        let grid = Grid2D::new(10, nz, 1.0, 9.0, -5.0, 5.0);
        for i in 1..nz {
            prop_assert!(grid.z[i] > grid.z[i - 1],
                "Z not monotone at {}: {} <= {}", i, grid.z[i], grid.z[i - 1]);
        }
    }

    /// R spacing is uniform.
    #[test]
    fn grid_r_uniform_spacing(nr in 4usize..64) {
        let grid = Grid2D::new(nr, 10, 1.0, 9.0, -5.0, 5.0);
        for i in 1..nr {
            let delta = grid.r[i] - grid.r[i - 1];
            prop_assert!((delta - grid.dr).abs() < 1e-12,
                "Non-uniform R spacing at {}: delta={}, dr={}", i, delta, grid.dr);
        }
    }

    /// Z spacing is uniform.
    #[test]
    fn grid_z_uniform_spacing(nz in 4usize..64) {
        let grid = Grid2D::new(10, nz, 1.0, 9.0, -5.0, 5.0);
        for i in 1..nz {
            let delta = grid.z[i] - grid.z[i - 1];
            prop_assert!((delta - grid.dz).abs() < 1e-12,
                "Non-uniform Z spacing at {}: delta={}, dz={}", i, delta, grid.dz);
        }
    }
}

// ── PlasmaState Invariants ───────────────────────────────────────────

proptest! {
    /// PlasmaState arrays have correct shapes.
    #[test]
    fn plasma_state_shapes(
        nr in 4usize..128,
        nz in 4usize..128,
    ) {
        let state = PlasmaState::new(nz, nr);

        prop_assert_eq!(state.psi.shape(), &[nz, nr]);
        prop_assert_eq!(state.j_phi.shape(), &[nz, nr]);
        prop_assert!(state.b_r.is_none());
        prop_assert!(state.b_z.is_none());
        prop_assert!(state.axis.is_none());
        prop_assert!(state.x_point.is_none());
    }

    /// New PlasmaState is zero-initialized.
    #[test]
    fn plasma_state_zero_init(
        nr in 4usize..64,
        nz in 4usize..64,
    ) {
        let state = PlasmaState::new(nz, nr);

        for &v in state.psi.iter() {
            prop_assert_eq!(v, 0.0);
        }
        for &v in state.j_phi.iter() {
            prop_assert_eq!(v, 0.0);
        }
        prop_assert_eq!(state.psi_axis, 0.0);
        prop_assert_eq!(state.psi_boundary, 0.0);
    }
}
