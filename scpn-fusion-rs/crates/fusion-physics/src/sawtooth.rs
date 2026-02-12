// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Sawtooth
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Reduced MHD sawtooth instability model.
//!
//! Port of `mhd_sawtooth.py`.
//! Models m=1 kink instability with Kadomtsev reconnection crash.

use fusion_math::tridiag::thomas_solve;
use num_complex::Complex64;

/// Number of radial grid points. Python: nr=100.
const NR: usize = 100;

/// Resistivity (1/Lundquist number). Python: eta = 1/S, S=1e4.
const ETA: f64 = 1e-4;

/// Viscosity coefficient. Python: nu=1e-4.
const NU: f64 = 1e-4;

/// Azimuthal mode number squared (m=1).
const M2: f64 = 1.0;

/// Crash amplitude threshold. Python: 0.1.
const CRASH_THRESHOLD: f64 = 0.1;

/// Default timestep. Python: dt=0.01.
const DEFAULT_DT: f64 = 0.01;

/// Reduced MHD sawtooth simulator.
pub struct ReducedMHD {
    /// Radial grid [0, 1].
    pub r: Vec<f64>,
    /// Grid spacing.
    dr: f64,
    /// Safety factor profile q(r) = 0.8 + 2r².
    pub q: Vec<f64>,
    /// Magnetic flux perturbation (m=1, complex).
    pub psi_hat: Vec<Complex64>,
    /// Stream function (m=1, complex).
    pub phi_hat: Vec<Complex64>,
    /// Number of Kadomtsev crashes.
    pub crash_count: usize,
    /// Peak amplitude at each step.
    pub amplitude_history: Vec<f64>,
}

impl ReducedMHD {
    pub fn new() -> Self {
        let dr = 1.0 / (NR - 1) as f64;
        let r: Vec<f64> = (0..NR).map(|i| i as f64 * dr).collect();
        let q: Vec<f64> = r.iter().map(|&ri| 0.8 + 2.0 * ri * ri).collect();

        // Initial small perturbation
        let psi_hat: Vec<Complex64> = r
            .iter()
            .map(|&ri| Complex64::new(1e-4 * ri * (1.0 - ri), 0.0))
            .collect();
        let phi_hat = vec![Complex64::new(0.0, 0.0); NR];

        ReducedMHD {
            r,
            dr,
            q,
            psi_hat,
            phi_hat,
            crash_count: 0,
            amplitude_history: Vec::new(),
        }
    }

    /// Radial Laplacian for m=1 mode.
    ///
    /// ∇²f = d²f/dr² + (1/r)df/dr - m²/r² f
    fn laplacian(&self, f: &[Complex64]) -> Vec<Complex64> {
        let n = f.len();
        let dr = self.dr;
        let mut result = vec![Complex64::new(0.0, 0.0); n];

        for i in 1..n - 1 {
            let ri = self.r[i].max(1e-10);
            let d2f = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / (dr * dr);
            let df = (f[i + 1] - f[i - 1]) / (2.0 * dr);
            result[i] = d2f + df / ri - f[i] * M2 / (ri * ri);
        }

        result
    }

    /// Solve ∇²φ = U via tridiagonal system (real and imaginary parts separately).
    fn solve_poisson(&self, u: &[Complex64]) -> Vec<Complex64> {
        let n = u.len();
        let dr = self.dr;
        let interior = n - 2;
        if interior == 0 {
            return vec![Complex64::new(0.0, 0.0); n];
        }

        let mut a_sub = vec![0.0; interior];
        let mut b_main = vec![0.0; interior];
        let mut c_sup = vec![0.0; interior];
        let mut d_re = vec![0.0; interior];
        let mut d_im = vec![0.0; interior];

        for j in 0..interior {
            let i = j + 1;
            let ri = self.r[i].max(1e-10);
            b_main[j] = -2.0 / (dr * dr) - M2 / (ri * ri);
            if j > 0 {
                a_sub[j] = 1.0 / (dr * dr) - 1.0 / (2.0 * ri * dr);
            }
            if j < interior - 1 {
                c_sup[j] = 1.0 / (dr * dr) + 1.0 / (2.0 * ri * dr);
            }
            d_re[j] = u[i].re;
            d_im[j] = u[i].im;
        }

        let x_re = thomas_solve(&a_sub, &b_main, &c_sup, &d_re);
        let x_im = thomas_solve(&a_sub, &b_main, &c_sup, &d_im);

        let mut result = vec![Complex64::new(0.0, 0.0); n];
        for j in 0..interior {
            result[j + 1] = Complex64::new(x_re[j], x_im[j]);
        }
        result
    }

    /// One time step. Returns (amplitude, crashed).
    pub fn step(&mut self, dt: f64) -> (f64, bool) {
        let n = NR;

        // Current density J = ∇²ψ
        let j_current = self.laplacian(&self.psi_hat);
        // Vorticity U = ∇²φ
        let u_vort = self.laplacian(&self.phi_hat);
        // Viscous term: ∇²U
        let lap_u = self.laplacian(&u_vort);

        let mut dpsi = vec![Complex64::new(0.0, 0.0); n];
        let mut du = vec![Complex64::new(0.0, 0.0); n];

        for i in 1..n - 1 {
            let gamma = 1.0 / self.q[i] - 1.0;
            // dψ/dt = k_∥ · φ + η · J
            dpsi[i] = gamma * self.phi_hat[i] + ETA * j_current[i];
            // dU/dt = k_∥ · J + γ · ψ - ν · ∇²U
            du[i] = gamma * j_current[i] + gamma * self.psi_hat[i] - NU * lap_u[i];
        }

        // Update psi
        for (psi_i, dpsi_i) in self.psi_hat[1..n - 1].iter_mut().zip(dpsi[1..n - 1].iter()) {
            *psi_i += dt * *dpsi_i;
        }

        // Update vorticity and solve for phi
        let mut new_u = u_vort;
        for i in 1..n - 1 {
            new_u[i] += dt * du[i];
        }
        self.phi_hat = self.solve_poisson(&new_u);

        // Amplitude check
        let amplitude = self
            .psi_hat
            .iter()
            .map(|c| c.norm())
            .fold(0.0_f64, f64::max);
        self.amplitude_history.push(amplitude);

        // Kadomtsev crash: flatten inside q=1 surface
        let mut crashed = false;
        if amplitude > CRASH_THRESHOLD {
            for i in 0..n {
                if self.q[i] < 1.0 {
                    self.psi_hat[i] *= 0.01;
                    self.phi_hat[i] *= 0.01;
                }
            }
            self.crash_count += 1;
            crashed = true;
        }

        (amplitude, crashed)
    }

    /// Run N steps with default dt.
    pub fn run(&mut self, n_steps: usize) -> Vec<(f64, bool)> {
        (0..n_steps).map(|_| self.step(DEFAULT_DT)).collect()
    }
}

impl Default for ReducedMHD {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sawtooth_creation() {
        let mhd = ReducedMHD::new();
        assert_eq!(mhd.r.len(), NR);
        // q(0) = 0.8, below 1 → unstable
        assert!(mhd.q[0] < 1.0, "q(0) = {} should be < 1.0", mhd.q[0]);
    }

    #[test]
    fn test_sawtooth_step_no_panic() {
        let mut mhd = ReducedMHD::new();
        let (amp, _) = mhd.step(DEFAULT_DT);
        assert!(amp.is_finite(), "Amplitude should be finite: {amp}");
    }

    #[test]
    fn test_sawtooth_crash_occurs() {
        let mut mhd = ReducedMHD::new();
        let results = mhd.run(3000);
        let any_crash = results.iter().any(|(_, crashed)| *crashed);
        assert!(
            any_crash,
            "Should see at least one sawtooth crash in 3000 steps"
        );
    }

    #[test]
    fn test_sawtooth_amplitude_history() {
        let mut mhd = ReducedMHD::new();
        mhd.run(100);
        assert_eq!(mhd.amplitude_history.len(), 100);
        // All amplitudes should be finite
        assert!(mhd.amplitude_history.iter().all(|a| a.is_finite()));
    }
}
