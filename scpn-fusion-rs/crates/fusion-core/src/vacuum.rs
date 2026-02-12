// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Vacuum
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Vacuum magnetic field from coils via toroidal Green's function.
//!
//! Port of fusion_kernel.py `calculate_vacuum_field()` (lines 59-93).
//! Uses complete elliptic integrals K(m), E(m) for the exact toroidal
//! single-coil flux function (Smythe/Jackson form).

use fusion_math::elliptic::{ellipe, ellipk};
use fusion_types::config::CoilConfig;
use fusion_types::state::Grid2D;
use ndarray::Array2;

/// Calculate vacuum magnetic flux from coils using the toroidal Green's function.
///
/// For each coil at `(Rc, Zc)` with current `I`, the flux contribution is:
///
///   Ψ = (μ₀ I / 2π) · √((R + Rc)² + (Z - Zc)²) · ((2 - k²)K(k²) - 2E(k²)) / k²
///
/// where k² = 4 R Rc / ((R + Rc)² + (Z - Zc)²), clipped to [1e-9, 0.999999].
///
/// Returns Ψ_vacuum `[nz, nr]`.
pub fn calculate_vacuum_field(grid: &Grid2D, coils: &[CoilConfig], mu0: f64) -> Array2<f64> {
    let nz = grid.nz;
    let nr = grid.nr;
    let mut psi_vac = Array2::zeros((nz, nr));

    for coil in coils {
        let rc = coil.r;
        let zc = coil.z;
        let current = coil.current;

        let prefactor = (mu0 * current) / (2.0 * std::f64::consts::PI);

        for iz in 0..nz {
            for ir in 0..nr {
                let r = grid.rr[[iz, ir]];
                let z = grid.zz[[iz, ir]];

                let dz = z - zc;
                let r_plus_rc = r + rc;
                let r_plus_rc_sq = r_plus_rc * r_plus_rc;
                let denom = r_plus_rc_sq + dz * dz;

                // k² parameter
                let mut k2 = (4.0 * r * rc) / denom;
                // CRITICAL: clip to avoid singularity at ellipk(1.0) and division by k2=0
                k2 = k2.clamp(1e-9, 0.999999);

                let k_val = ellipk(k2);
                let e_val = ellipe(k2);

                let sqrt_term = denom.sqrt();
                let term = ((2.0 - k2) * k_val - 2.0 * e_val) / k2;
                let coil_flux = prefactor * sqrt_term * term;

                psi_vac[[iz, ir]] += coil_flux;
            }
        }
    }

    psi_vac
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusion_types::config::ReactorConfig;
    use std::path::PathBuf;

    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
    }

    fn config_path(relative: &str) -> String {
        project_root().join(relative).to_string_lossy().to_string()
    }

    #[test]
    fn test_vacuum_field_not_nan() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        let grid = cfg.create_grid();
        let psi_vac = calculate_vacuum_field(&grid, &cfg.coils, cfg.physics.vacuum_permeability);

        // No NaN values
        assert!(
            !psi_vac.iter().any(|v| v.is_nan()),
            "Vacuum field contains NaN"
        );

        // Should have non-zero values (coils produce flux)
        let max_val = psi_vac.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val.abs() > 1e-10, "Vacuum field is all zeros");
    }

    #[test]
    fn test_vacuum_field_symmetry() {
        // Single coil at Z=0 should produce field symmetric about Z=0
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let coil = CoilConfig {
            name: "test".to_string(),
            r: 5.0,
            z: 0.0,
            current: 1.0,
        };
        let psi = calculate_vacuum_field(&grid, &[coil], 1.0);

        // Check symmetry: psi[iz, ir] ≈ psi[nz-1-iz, ir]
        for ir in 0..33 {
            for iz in 0..16 {
                let diff = (psi[[iz, ir]] - psi[[32 - iz, ir]]).abs();
                assert!(
                    diff < 1e-10,
                    "Symmetry broken at iz={iz}, ir={ir}: {} vs {}",
                    psi[[iz, ir]],
                    psi[[32 - iz, ir]]
                );
            }
        }
    }

    #[test]
    fn test_vacuum_field_single_coil_shape() {
        // Flux from a single coil should peak near the coil position
        let grid = Grid2D::new(33, 33, 1.0, 9.0, -5.0, 5.0);
        let coil = CoilConfig {
            name: "test".to_string(),
            r: 5.0,
            z: 0.0,
            current: 1.0,
        };
        let psi = calculate_vacuum_field(&grid, &[coil], 1.0);

        // Find maximum
        let mut max_val = f64::NEG_INFINITY;
        let mut max_iz = 0;
        let mut max_ir = 0;
        for iz in 0..33 {
            for ir in 0..33 {
                if psi[[iz, ir]] > max_val {
                    max_val = psi[[iz, ir]];
                    max_iz = iz;
                    max_ir = ir;
                }
            }
        }

        // Maximum should be near Z=0 (middle row)
        assert!(
            (max_iz as f64 - 16.0).abs() < 3.0,
            "Peak not near Z=0: iz={max_iz}"
        );
        // Maximum should be near R=5.0 (middle of grid)
        assert!(
            (max_ir as f64 - 16.0).abs() < 3.0,
            "Peak not near R=5: ir={max_ir}"
        );
    }
}
