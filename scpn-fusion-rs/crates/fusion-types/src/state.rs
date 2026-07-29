// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — State
//! Shared grid, plasma, transport, thermodynamic, and solver-result states.

use ndarray::{Array1, Array2};

/// 2D computational grid with precomputed coordinates.
/// Matches Python: self.R, self.Z, self.RR, self.ZZ, self.dR, self.dZ
#[derive(Debug, Clone)]
pub struct Grid2D {
    /// Number of radial grid points.
    pub nr: usize,
    /// Number of vertical grid points.
    pub nz: usize,
    /// Uniform radial coordinates with length `nr`.
    pub r: Array1<f64>,
    /// Uniform vertical coordinates with length `nz`.
    pub z: Array1<f64>,
    /// Radial coordinate spacing.
    pub dr: f64,
    /// Vertical coordinate spacing.
    pub dz: f64,
    /// Radial meshgrid with row-major shape `(nz, nr)`.
    pub rr: Array2<f64>,
    /// Vertical meshgrid with row-major shape `(nz, nr)`.
    pub zz: Array2<f64>,
}

impl Grid2D {
    /// Creates a uniform rectangular grid from inclusive coordinate bounds.
    ///
    /// The meshgrids use NumPy-compatible `(nz, nr)` row-major ordering.
    ///
    /// Python equivalent:
    ///   self.R = np.linspace(R_min, R_max, NR)
    ///   self.Z = np.linspace(Z_min, Z_max, NZ)
    ///   self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
    pub fn new(nr: usize, nz: usize, r_min: f64, r_max: f64, z_min: f64, z_max: f64) -> Self {
        let r = Array1::linspace(r_min, r_max, nr);
        let z = Array1::linspace(z_min, z_max, nz);
        let dr = if nr > 1 { r[1] - r[0] } else { r_max - r_min };
        let dz = if nz > 1 { z[1] - z[0] } else { z_max - z_min };

        // np.meshgrid(R, Z) produces [nz, nr] arrays
        let mut rr = Array2::zeros((nz, nr));
        let mut zz = Array2::zeros((nz, nr));
        for iz in 0..nz {
            for ir in 0..nr {
                rr[[iz, ir]] = r[ir];
                zz[[iz, ir]] = z[iz];
            }
        }

        Grid2D {
            nr,
            nz,
            r,
            z,
            dr,
            dz,
            rr,
            zz,
        }
    }
}

/// Complete plasma equilibrium state.
/// Matches Python: self.Psi, self.J_phi, self.B_R, self.B_Z
#[derive(Debug, Clone)]
pub struct PlasmaState {
    /// Poloidal magnetic flux with shape `(nz, nr)`.
    pub psi: Array2<f64>,
    /// Toroidal current density with shape `(nz, nr)`.
    pub j_phi: Array2<f64>,
    /// Optional radial magnetic-field component with shape `(nz, nr)`.
    pub b_r: Option<Array2<f64>>,
    /// Optional vertical magnetic-field component with shape `(nz, nr)`.
    pub b_z: Option<Array2<f64>>,
    /// Optional magnetic-axis O-point as `(R, Z)`.
    pub axis: Option<(f64, f64)>,
    /// Optional magnetic X-point as `(R, Z)`.
    pub x_point: Option<(f64, f64)>,
    /// Flux value at the magnetic axis.
    pub psi_axis: f64,
    /// Flux value at the plasma boundary.
    pub psi_boundary: f64,
}

impl PlasmaState {
    /// Creates a zero-filled `(nz, nr)` state with unset field and point data.
    pub fn new(nz: usize, nr: usize) -> Self {
        PlasmaState {
            psi: Array2::zeros((nz, nr)),
            j_phi: Array2::zeros((nz, nr)),
            b_r: None,
            b_z: None,
            axis: None,
            x_point: None,
            psi_axis: 0.0,
            psi_boundary: 0.0,
        }
    }
}

/// 1D radial profiles for transport solver.
/// Matches Python: TransportSolver.Te, Ti, ne, n_impurity
#[derive(Debug, Clone)]
pub struct RadialProfiles {
    /// Normalised radial coordinate, conventionally spanning `[0, 1]`.
    pub rho: Array1<f64>,
    /// Electron temperature profile in keV.
    pub te: Array1<f64>,
    /// Ion temperature profile in keV.
    pub ti: Array1<f64>,
    /// Electron density profile in units of `10¹⁹ m⁻³`.
    pub ne: Array1<f64>,
    /// Impurity-density profile in the caller's declared density units.
    pub n_impurity: Array1<f64>,
}

/// Thermodynamics result from ignition calculation.
#[derive(Debug, Clone)]
pub struct ThermodynamicsResult {
    /// Total fusion power in megawatts.
    pub p_fusion_mw: f64,
    /// Alpha-particle heating power in megawatts.
    pub p_alpha_mw: f64,
    /// Total loss power in megawatts.
    pub p_loss_mw: f64,
    /// Auxiliary heating power in megawatts.
    pub p_aux_mw: f64,
    /// Net power balance in megawatts.
    pub net_mw: f64,
    /// Fusion gain factor `Q`.
    pub q_factor: f64,
    /// Peak plasma temperature in keV.
    pub t_peak_kev: f64,
    /// Thermal stored energy in megajoules.
    pub w_thermal_mj: f64,
}

/// Stability analysis result.
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Two eigenvalues of the reduced stability matrix.
    pub eigenvalues: [f64; 2],
    /// Corresponding column eigenvectors of the reduced stability matrix.
    pub eigenvectors: [[f64; 2]; 2],
    /// Vertical-field decay index.
    pub decay_index: f64,
    /// Net radial force in meganewtons.
    pub radial_force_mn: f64,
    /// Net vertical force in meganewtons.
    pub vertical_force_mn: f64,
    /// Whether all configured stability criteria passed.
    pub is_stable: bool,
}

/// Equilibrium solve result.
#[derive(Debug, Clone)]
pub struct EquilibriumResult {
    /// Whether the solver satisfied its convergence gate.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final reported nonlinear residual.
    pub residual: f64,
    /// Magnetic-axis coordinates as `(R, Z)`.
    pub axis_position: (f64, f64),
    /// Magnetic X-point coordinates as `(R, Z)`.
    pub x_point_position: (f64, f64),
    /// Flux value at the magnetic axis.
    pub psi_axis: f64,
    /// Flux value at the plasma boundary.
    pub psi_boundary: f64,
    /// Solver wall-clock duration in milliseconds.
    pub solve_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation_128() {
        let grid = Grid2D::new(128, 128, 1.0, 9.0, -5.0, 5.0);
        assert_eq!(grid.nr, 128);
        assert_eq!(grid.nz, 128);
        assert!((grid.dr - (9.0 - 1.0) / 127.0).abs() < 1e-10);
        assert!((grid.dz - (5.0 - (-5.0)) / 127.0).abs() < 1e-10);
        // Meshgrid check: rr[0, 0] should be R_min, rr[0, nr-1] should be R_max
        assert!((grid.rr[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((grid.rr[[0, 127]] - 9.0).abs() < 1e-10);
        // zz[0, 0] should be Z_min, zz[nz-1, 0] should be Z_max
        assert!((grid.zz[[0, 0]] - (-5.0)).abs() < 1e-10);
        assert!((grid.zz[[127, 0]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_creation_65() {
        let grid = Grid2D::new(65, 65, 2.0, 10.0, -6.0, 6.0);
        assert_eq!(grid.nr, 65);
        assert_eq!(grid.nz, 65);
        assert!((grid.dr - (10.0 - 2.0) / 64.0).abs() < 1e-10);
        assert!((grid.dz - (6.0 - (-6.0)) / 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_meshgrid_consistency() {
        let grid = Grid2D::new(10, 20, 1.0, 5.0, -3.0, 3.0);
        // All values in a row should have the same Z
        for iz in 0..grid.nz {
            let z_val = grid.zz[[iz, 0]];
            for ir in 0..grid.nr {
                assert!(
                    (grid.zz[[iz, ir]] - z_val).abs() < 1e-15,
                    "Z should be constant along a row"
                );
            }
        }
        // All values in a column should have the same R
        for ir in 0..grid.nr {
            let r_val = grid.rr[[0, ir]];
            for iz in 0..grid.nz {
                assert!(
                    (grid.rr[[iz, ir]] - r_val).abs() < 1e-15,
                    "R should be constant along a column"
                );
            }
        }
    }

    #[test]
    fn test_plasma_state_initialization() {
        let state = PlasmaState::new(128, 128);
        assert_eq!(state.psi.shape(), &[128, 128]);
        assert_eq!(state.j_phi.shape(), &[128, 128]);
        assert!(state.b_r.is_none());
        assert!(state.b_z.is_none());
        assert!(state.axis.is_none());
        assert_eq!(state.psi_axis, 0.0);
    }
}
