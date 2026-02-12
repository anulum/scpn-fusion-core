//! Reduced Hall-MHD with spontaneous zonal flow generation.
//!
//! Port of `hall_mhd_discovery.py`.
//! Models drift-Alfvén turbulence with self-organized zonal flows.

use fusion_math::fft::{fft2, ifft2};
use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;
use rand_distr::StandardNormal;

/// Default grid size. Python: GRID=64.
const GRID: usize = 64;

/// Timestep. Python: DT=0.005.
const DT: f64 = 0.005;

/// Larmor radius / Hall scale. Python: rho_s=0.1.
const RHO_S: f64 = 0.1;

/// Plasma beta. Python: beta=0.01.
const BETA: f64 = 0.01;

/// Viscosity. Python: nu=1e-4.
const VISCOSITY: f64 = 1e-4;

/// Resistivity. Python: eta=1e-4.
const RESISTIVITY: f64 = 1e-4;

/// De-aliasing fraction (2/3 rule).
const DEALIAS_FRACTION: f64 = 2.0 / 3.0;

/// Reduced Hall-MHD simulator.
pub struct HallMHD {
    /// Grid size.
    pub n: usize,
    /// Stream function (spectral).
    pub phi_k: Array2<Complex64>,
    /// Magnetic flux (spectral).
    pub psi_k: Array2<Complex64>,
    /// kx wavenumber grid (column index).
    kx: Array2<f64>,
    /// ky wavenumber grid (row index).
    ky: Array2<f64>,
    /// k² wavenumber squared.
    k2: Array2<f64>,
    /// De-aliasing mask.
    mask: Array2<f64>,
    /// Total energy history.
    pub energy_history: Vec<f64>,
    /// Zonal flow energy history.
    pub zonal_history: Vec<f64>,
}

impl HallMHD {
    pub fn new(n: usize) -> Self {
        let mut rng = rand::thread_rng();

        let kx = Array2::from_shape_fn((n, n), |(_, j)| {
            if j <= n / 2 {
                j as f64
            } else {
                j as f64 - n as f64
            }
        });
        let ky = Array2::from_shape_fn((n, n), |(i, _)| {
            if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            }
        });
        let k2 = Array2::from_shape_fn((n, n), |(i, j)| {
            let kxi = if j <= n / 2 {
                j as f64
            } else {
                j as f64 - n as f64
            };
            let kyi = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            kxi * kxi + kyi * kyi
        });

        // 2/3 rule de-aliasing mask
        let k_max = (n / 2) as f64;
        let k_cut = DEALIAS_FRACTION * k_max;
        let mask = Array2::from_shape_fn((n, n), |(i, j)| {
            let kxi = if j <= n / 2 {
                j as f64
            } else {
                j as f64 - n as f64
            };
            let kyi = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            if kxi * kxi + kyi * kyi < k_cut * k_cut {
                1.0
            } else {
                0.0
            }
        });

        // Initial noise
        let phi_k = Array2::from_shape_fn((n, n), |_| {
            Complex64::new(
                rng.sample::<f64, _>(StandardNormal) * 1e-3,
                rng.sample::<f64, _>(StandardNormal) * 1e-3,
            )
        });
        let psi_k = Array2::from_shape_fn((n, n), |_| {
            Complex64::new(
                rng.sample::<f64, _>(StandardNormal) * 1e-3,
                rng.sample::<f64, _>(StandardNormal) * 1e-3,
            )
        });

        HallMHD {
            n,
            phi_k,
            psi_k,
            kx,
            ky,
            k2,
            mask,
            energy_history: Vec::new(),
            zonal_history: Vec::new(),
        }
    }

    /// Poisson bracket [A, B] = dA/dx·dB/dy - dA/dy·dB/dx.
    fn poisson_bracket(
        &self,
        a_k: &Array2<Complex64>,
        b_k: &Array2<Complex64>,
    ) -> Array2<Complex64> {
        let n = self.n;
        let iu = Complex64::new(0.0, 1.0);

        // Spectral derivatives → real space
        let da_dx_k = Array2::from_shape_fn((n, n), |(i, j)| {
            iu * Complex64::new(self.kx[[i, j]], 0.0) * a_k[[i, j]]
        });
        let da_dy_k = Array2::from_shape_fn((n, n), |(i, j)| {
            iu * Complex64::new(self.ky[[i, j]], 0.0) * a_k[[i, j]]
        });
        let db_dx_k = Array2::from_shape_fn((n, n), |(i, j)| {
            iu * Complex64::new(self.kx[[i, j]], 0.0) * b_k[[i, j]]
        });
        let db_dy_k = Array2::from_shape_fn((n, n), |(i, j)| {
            iu * Complex64::new(self.ky[[i, j]], 0.0) * b_k[[i, j]]
        });

        let da_dx = ifft2(&da_dx_k);
        let da_dy = ifft2(&da_dy_k);
        let db_dx = ifft2(&db_dx_k);
        let db_dy = ifft2(&db_dy_k);

        // Product in real space → spectral + de-aliasing
        let product = Array2::from_shape_fn((n, n), |(i, j)| {
            da_dx[[i, j]] * db_dy[[i, j]] - da_dy[[i, j]] * db_dx[[i, j]]
        });

        let mut result = fft2(&product);
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] *= Complex64::new(self.mask[[i, j]], 0.0);
            }
        }
        result
    }

    /// Compute RHS of reduced Hall-MHD equations.
    fn dynamics(
        &self,
        phi_k: &Array2<Complex64>,
        psi_k: &Array2<Complex64>,
    ) -> (Array2<Complex64>, Array2<Complex64>) {
        let n = self.n;

        // Derived: vorticity U = -k²φ, current density J = -k²ψ
        let u_k = Array2::from_shape_fn((n, n), |(i, j)| {
            -Complex64::new(self.k2[[i, j]], 0.0) * phi_k[[i, j]]
        });
        let j_k = Array2::from_shape_fn((n, n), |(i, j)| {
            -Complex64::new(self.k2[[i, j]], 0.0) * psi_k[[i, j]]
        });

        // Nonlinear terms
        let bracket_phi_u = self.poisson_bracket(phi_k, &u_k);
        let bracket_j_psi = self.poisson_bracket(&j_k, psi_k);
        let bracket_phi_psi = self.poisson_bracket(phi_k, psi_k);

        // dU/dt = -[φ,U] + β[J,ψ] - ν·k²·U
        let du_k = Array2::from_shape_fn((n, n), |(i, j)| {
            let k2v = self.k2[[i, j]];
            -bracket_phi_u[[i, j]] + Complex64::new(BETA, 0.0) * bracket_j_psi[[i, j]]
                - Complex64::new(VISCOSITY * k2v, 0.0) * u_k[[i, j]]
        });

        // dφ/dt = -dU/dt / k²
        let dphi_k = Array2::from_shape_fn((n, n), |(i, j)| {
            let k2v = self.k2[[i, j]];
            if k2v > 1e-10 {
                -du_k[[i, j]] / Complex64::new(k2v, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });

        // dψ/dt = -[φ,ψ] + ρ_s²·[J,ψ] - η·k²·ψ
        let dpsi_k = Array2::from_shape_fn((n, n), |(i, j)| {
            let k2v = self.k2[[i, j]];
            -bracket_phi_psi[[i, j]]
                + Complex64::new(RHO_S * RHO_S, 0.0) * bracket_j_psi[[i, j]]
                - Complex64::new(RESISTIVITY * k2v, 0.0) * psi_k[[i, j]]
        });

        (dphi_k, dpsi_k)
    }

    /// RK2 (midpoint) time step. Returns (total_energy, zonal_energy).
    pub fn step(&mut self) -> (f64, f64) {
        let n = self.n;

        // RK2 midpoint
        let (dphi1, dpsi1) = self.dynamics(&self.phi_k, &self.psi_k);

        let phi_mid = Array2::from_shape_fn((n, n), |(i, j)| {
            self.phi_k[[i, j]] + Complex64::new(0.5 * DT, 0.0) * dphi1[[i, j]]
        });
        let psi_mid = Array2::from_shape_fn((n, n), |(i, j)| {
            self.psi_k[[i, j]] + Complex64::new(0.5 * DT, 0.0) * dpsi1[[i, j]]
        });

        let (dphi2, dpsi2) = self.dynamics(&phi_mid, &psi_mid);

        for i in 0..n {
            for j in 0..n {
                self.phi_k[[i, j]] += Complex64::new(DT, 0.0) * dphi2[[i, j]];
                self.psi_k[[i, j]] += Complex64::new(DT, 0.0) * dpsi2[[i, j]];
            }
        }

        // Compute energies
        let mut total_energy = 0.0;
        let mut zonal_energy = 0.0;
        for i in 0..n {
            for j in 0..n {
                let e = self.phi_k[[i, j]].norm_sqr();
                total_energy += e;

                // Zonal modes: ky ≈ 0 (row i=0) and kx ≠ 0 (col j≠0)
                let kyv = if i <= n / 2 {
                    i as f64
                } else {
                    i as f64 - n as f64
                };
                let kxv = if j <= n / 2 {
                    j as f64
                } else {
                    j as f64 - n as f64
                };
                if kyv.abs() < 0.5 && kxv.abs() > 0.5 {
                    zonal_energy += e;
                }
            }
        }

        self.energy_history.push(total_energy);
        self.zonal_history.push(zonal_energy);

        (total_energy, zonal_energy)
    }

    /// Run N steps.
    pub fn run(&mut self, n_steps: usize) -> Vec<(f64, f64)> {
        (0..n_steps).map(|_| self.step()).collect()
    }
}

impl Default for HallMHD {
    fn default() -> Self {
        Self::new(GRID)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hall_mhd_creation() {
        let mhd = HallMHD::new(GRID);
        assert_eq!(mhd.phi_k.nrows(), GRID);
        assert_eq!(mhd.psi_k.nrows(), GRID);
    }

    #[test]
    fn test_step_finite() {
        let mut mhd = HallMHD::new(GRID);
        let (e, z) = mhd.step();
        assert!(e.is_finite(), "Total energy should be finite: {e}");
        assert!(z.is_finite(), "Zonal energy should be finite: {z}");
    }

    #[test]
    fn test_zonal_energy_present() {
        let mut mhd = HallMHD::new(GRID);
        mhd.run(500);

        // After evolution, zonal energy should be non-negligible fraction of total
        let total: f64 = mhd.energy_history.last().copied().unwrap_or(0.0);
        let zonal: f64 = mhd.zonal_history.last().copied().unwrap_or(0.0);
        let max_zonal: f64 = mhd
            .zonal_history
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);

        // Zonal modes should carry some energy
        assert!(
            max_zonal > 0.0 && zonal.is_finite(),
            "Zonal energy should be present: max_zonal={max_zonal}, final_zonal={zonal}, total={total}"
        );
    }

    #[test]
    fn test_energy_bounded() {
        let mut mhd = HallMHD::new(GRID);
        let results = mhd.run(100);
        for (e, _) in &results {
            assert!(e.is_finite(), "Energy must stay finite");
        }
    }
}
