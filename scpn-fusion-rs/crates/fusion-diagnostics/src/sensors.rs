// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Sensors
//! Synthetic diagnostic sensors for tokamak plasmas.
//!
//! Port of `synthetic_sensors.py`.
//! Magnetic probes on D-shaped wall + bolometer fan chords.

use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Major radius [m]. Python: 6.0.
const R0: f64 = 6.0;

/// Minor radius [m]. Python: 3.0.
const A_MINOR: f64 = 3.0;

/// Elongation. Python: 1.8.
const KAPPA: f64 = 1.8;

/// Wall offset [m]. Python: 0.5.
const WALL_OFFSET: f64 = 0.5;

/// Number of magnetic probes. Python: 20.
const N_PROBES: usize = 20;

/// Number of bolometer chords. Python: 16.
const N_BOLO: usize = 16;

/// Bolometer origin (R, Z) [m]. Python: (6.0, 5.0).
const BOLO_ORIGIN: (f64, f64) = (6.0, 5.0);

/// Bolometer target Z [m]. Python: -4.0.
const BOLO_TARGET_Z: f64 = -4.0;

/// Bolometer target R range [m]. Python: (3.0, 9.0).
const BOLO_R_MIN: f64 = 3.0;
const BOLO_R_MAX: f64 = 9.0;

/// Ray march samples for bolometer. Python: 50.
const BOLO_SAMPLES: usize = 50;

/// Standard magnetic-probe sensor-noise σ (Python: 0.01). The measurement kernel
/// is noise-free; simulation callers layer additive noise with this σ on top.
pub const MAG_NOISE: f64 = 0.01;

/// Bolometer noise fraction. Python: 0.05.
const BOLO_NOISE_FRAC: f64 = 0.05;

/// Bolometer noise floor. Python: 0.001.
const BOLO_NOISE_FLOOR: f64 = 0.001;

/// A bolometer chord: start (R,Z), end (R,Z).
#[derive(Debug, Clone)]
pub struct BoloChord {
    pub start: (f64, f64),
    pub end: (f64, f64),
}

/// Synthetic sensor suite for a tokamak.
pub struct SensorSuite {
    /// Magnetic probe positions: (R, Z) for each probe.
    pub probe_r: Vec<f64>,
    pub probe_z: Vec<f64>,
    /// Bolometer viewing chords.
    pub bolo_chords: Vec<BoloChord>,
    /// Grid parameters for mapping to indices.
    pub r_min: f64,
    pub z_min: f64,
    pub dr: f64,
    pub dz: f64,
    pub nr: usize,
    pub nz: usize,
}

impl SensorSuite {
    /// Create sensor suite for a given grid.
    pub fn new(nr: usize, nz: usize, r_min: f64, r_max: f64, z_min: f64, z_max: f64) -> Self {
        let dr = (r_max - r_min) / (nr - 1) as f64;
        let dz = (z_max - z_min) / (nz - 1) as f64;

        // Generate magnetic probe positions on the D-shaped wall. theta is an
        // endpoint-inclusive linspace(0, 2*pi, N_PROBES) to match the NumPy tier
        // (`scpn_fusion.diagnostics.synthetic_sensors.magnetic_probe_positions`).
        let mut probe_r = Vec::with_capacity(N_PROBES);
        let mut probe_z = Vec::with_capacity(N_PROBES);
        let wall_radius = A_MINOR + WALL_OFFSET;
        for i in 0..N_PROBES {
            let theta = 2.0 * PI * i as f64 / (N_PROBES - 1) as f64;
            probe_r.push(R0 + wall_radius * theta.cos());
            probe_z.push(wall_radius * KAPPA * theta.sin());
        }

        // Generate bolometer fan chords
        let mut bolo_chords = Vec::with_capacity(N_BOLO);
        for i in 0..N_BOLO {
            let target_r = BOLO_R_MIN + (BOLO_R_MAX - BOLO_R_MIN) * i as f64 / (N_BOLO - 1) as f64;
            bolo_chords.push(BoloChord {
                start: BOLO_ORIGIN,
                end: (target_r, BOLO_TARGET_Z),
            });
        }

        SensorSuite {
            probe_r,
            probe_z,
            bolo_chords,
            r_min,
            z_min,
            dr,
            dz,
            nr,
            nz,
        }
    }

    /// Map (R, Z) to grid indices, clamped.
    fn to_grid(&self, r: f64, z: f64) -> (usize, usize) {
        let ir = ((r - self.r_min) / self.dr).floor() as isize;
        let iz = ((z - self.z_min) / self.dz).floor() as isize;
        let ir = ir.clamp(0, self.nr as isize - 1) as usize;
        let iz = iz.clamp(0, self.nz as isize - 1) as usize;
        (ir, iz)
    }

    /// Measure magnetic flux at probe positions. Returns array of length N_PROBES.
    ///
    /// Deterministic, noise-free bilinear interpolation of `psi` at the probe
    /// positions — the canonical `measure_magnetics` dispatch contract, evaluated
    /// with the same stencil as the NumPy tier
    /// (`scpn_fusion.diagnostics.synthetic_sensors.measure_magnetics`), so the two
    /// agree to a tight tolerance. Sensor noise is an additive simulation concern
    /// layered on top by the callers, not part of the measurement kernel.
    pub fn measure_magnetics(&self, psi: &Array2<f64>) -> Vec<f64> {
        let nr = self.nr as i64;
        let nz = self.nz as i64;
        let mut measurements = Vec::with_capacity(self.probe_r.len());
        for i in 0..self.probe_r.len() {
            let r = self.probe_r[i];
            let z = self.probe_z[i];
            let ir = ((r - self.r_min) / self.dr) as i64;
            let iz = ((z - self.z_min) / self.dz) as i64;
            let val = if ir >= 0 && ir < nr - 1 && iz >= 0 && iz < nz - 1 {
                let iru = ir as usize;
                let izu = iz as usize;
                let wr = (r - (self.r_min + ir as f64 * self.dr)) / self.dr;
                let wz = (z - (self.z_min + iz as f64 * self.dz)) / self.dz;
                let v00 = psi[[izu, iru]];
                let v10 = psi[[izu, iru + 1]];
                let v01 = psi[[izu + 1, iru]];
                let v11 = psi[[izu + 1, iru + 1]];
                (1.0 - wr) * (1.0 - wz) * v00
                    + wr * (1.0 - wz) * v10
                    + (1.0 - wr) * wz * v01
                    + wr * wz * v11
            } else {
                let irc = ir.clamp(0, nr - 1) as usize;
                let izc = iz.clamp(0, nz - 1) as usize;
                psi[[izc, irc]]
            };
            measurements.push(val);
        }
        measurements
    }

    /// Measure bolometer line integrals. Returns array of length N_BOLO.
    pub fn measure_bolometer(&self, emission: &Array2<f64>) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        let mut signals = Vec::with_capacity(self.bolo_chords.len());
        for chord in &self.bolo_chords {
            let (sr, sz) = chord.start;
            let (er, ez) = chord.end;
            let length = ((er - sr).powi(2) + (ez - sz).powi(2)).sqrt();
            let dl = length / BOLO_SAMPLES as f64;

            let mut integral = 0.0;
            for k in 0..BOLO_SAMPLES {
                let t = k as f64 / BOLO_SAMPLES as f64;
                let r = sr + t * (er - sr);
                let z = sz + t * (ez - sz);
                let (ir, iz) = self.to_grid(r, z);
                if ir < self.nr && iz < self.nz {
                    integral += emission[[iz, ir]] * dl;
                }
            }

            // Photon shot noise
            let noise_sigma = if integral > 0.0 {
                BOLO_NOISE_FRAC * integral
            } else {
                BOLO_NOISE_FLOOR
            };
            let noise_dist = Normal::new(0.0, noise_sigma).unwrap();
            integral += noise_dist.sample(&mut rng);
            signals.push(integral);
        }
        signals
    }

    /// Number of magnetic probes.
    pub fn n_probes(&self) -> usize {
        self.probe_r.len()
    }

    /// Number of bolometer chords.
    pub fn n_chords(&self) -> usize {
        self.bolo_chords.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_suite() -> SensorSuite {
        // ITER-like grid: R ∈ [3, 9], Z ∈ [-5, 5], 65×65
        SensorSuite::new(65, 65, 3.0, 9.0, -5.0, 5.0)
    }

    #[test]
    fn test_probe_count() {
        let suite = make_suite();
        assert_eq!(suite.n_probes(), N_PROBES);
        assert_eq!(suite.n_chords(), N_BOLO);
    }

    #[test]
    fn test_probe_positions_on_wall() {
        let suite = make_suite();
        let wall_r = A_MINOR + WALL_OFFSET;
        for i in 0..N_PROBES {
            let dr = suite.probe_r[i] - R0;
            let dz = suite.probe_z[i] / (KAPPA * wall_r);
            let r_norm = dr / wall_r;
            // Should lie on unit circle
            let d = (r_norm * r_norm + dz * dz).sqrt();
            assert!((d - 1.0).abs() < 1e-10, "Probe {i} not on wall: d={d}");
        }
    }

    #[test]
    fn test_magnetics_measurement_dimension() {
        let suite = make_suite();
        let psi = Array2::from_elem((65, 65), 1.0);
        let meas = suite.measure_magnetics(&psi);
        assert_eq!(meas.len(), N_PROBES);
        // Noise-free bilinear interpolation of a uniform field returns it exactly.
        for &v in &meas {
            assert!(
                (v - 1.0).abs() < 1e-12,
                "Magnetic measurement should equal the uniform field: {v}"
            );
        }
    }

    #[test]
    fn test_magnetics_bilinear_interpolates_linear_field() {
        // A field linear in R is reproduced exactly by bilinear interpolation, so
        // each probe reads back psi at its own R coordinate.
        let suite = make_suite();
        let mut psi = Array2::zeros((65, 65));
        let dr = (9.0 - 3.0) / 64.0;
        for iz in 0..65 {
            for ir in 0..65 {
                psi[[iz, ir]] = 3.0 + ir as f64 * dr; // = R at that column
            }
        }
        let dz = 10.0 / 64.0;
        let meas = suite.measure_magnetics(&psi);
        let mut interior_checked = 0;
        for (i, &v) in meas.iter().enumerate() {
            let r = suite.probe_r[i];
            let z = suite.probe_z[i];
            // Only interior probes interpolate; out-of-grid probes clamp. Mirror
            // the bilinear branch condition on both axes.
            let ir = ((r - 3.0) / dr) as i64;
            let iz = ((z + 5.0) / dz) as i64;
            if (0..64).contains(&ir) && (0..64).contains(&iz) {
                assert!((v - r).abs() < 1e-9, "probe {i}: got {v}, want {r}");
                interior_checked += 1;
            }
        }
        assert!(interior_checked > 0, "no interior probes were exercised");
    }

    #[test]
    fn test_bolometer_measurement_dimension() {
        let suite = make_suite();
        let emission = Array2::from_elem((65, 65), 1.0);
        let signals = suite.measure_bolometer(&emission);
        assert_eq!(signals.len(), N_BOLO);
        // Uniform emission → line integral ≈ chord_length
        for &s in &signals {
            assert!(s > 0.0, "Bolometer signal should be positive: {s}");
        }
    }

    #[test]
    fn test_bolometer_zero_emission() {
        let suite = make_suite();
        let emission = Array2::zeros((65, 65));
        let signals = suite.measure_bolometer(&emission);
        // Zero emission → signals ≈ 0 (just noise)
        for &s in &signals {
            assert!(
                s.abs() < 0.1,
                "Zero emission should give near-zero signal: {s}"
            );
        }
    }
}
