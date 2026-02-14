//! Reduced particle tracker overlay for hybrid MHD/PIC-style current feedback.
//!
//! This module introduces a deterministic charged-particle pusher using the
//! Boris integrator and a simple toroidal current deposition path on the
//! Grad-Shafranov grid.

use fusion_types::error::{FusionError, FusionResult};
use fusion_types::state::Grid2D;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

const MIN_RADIUS_M: f64 = 1e-9;
const MIN_CELL_AREA_M2: f64 = 1e-12;
const MIN_CURRENT_INTEGRAL: f64 = 1e-9;
const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;
const ALPHA_MASS_KG: f64 = 6.644_657_335_7e-27;
const ALPHA_CHARGE_C: f64 = 2.0 * ELEMENTARY_CHARGE_C;

/// Charged macro-particle state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChargedParticle {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
    pub vx_m_s: f64,
    pub vy_m_s: f64,
    pub vz_m_s: f64,
    pub charge_c: f64,
    pub mass_kg: f64,
    pub weight: f64,
}

impl ChargedParticle {
    /// Cylindrical major radius R from Cartesian position.
    pub fn cylindrical_radius_m(&self) -> f64 {
        (self.x_m * self.x_m + self.y_m * self.y_m).sqrt()
    }

    /// Toroidal velocity component v_phi in cylindrical basis.
    pub fn toroidal_velocity_m_s(&self) -> f64 {
        let r = self.cylindrical_radius_m().max(MIN_RADIUS_M);
        (-self.y_m * self.vx_m_s + self.x_m * self.vy_m_s) / r
    }

    /// Non-relativistic kinetic energy [J].
    pub fn kinetic_energy_j(&self) -> f64 {
        let v2 = self.vx_m_s * self.vx_m_s + self.vy_m_s * self.vy_m_s + self.vz_m_s * self.vz_m_s;
        0.5 * self.mass_kg * v2
    }

    /// Non-relativistic kinetic energy [MeV].
    pub fn kinetic_energy_mev(&self) -> f64 {
        self.kinetic_energy_j() / (1.0e6 * ELEMENTARY_CHARGE_C)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticlePopulationSummary {
    pub count: usize,
    pub mean_energy_mev: f64,
    pub p95_energy_mev: f64,
    pub max_energy_mev: f64,
    pub runaway_fraction: f64,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn nearest_index(axis: &Array1<f64>, value: f64) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = f64::INFINITY;
    for (idx, x) in axis.iter().copied().enumerate() {
        let dist = (x - value).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    best_idx
}

/// Create deterministic alpha test particles for hybrid kinetic-fluid overlays.
pub fn seed_alpha_test_particles(
    n_particles: usize,
    major_radius_m: f64,
    z_m: f64,
    kinetic_energy_mev: f64,
    pitch_cos: f64,
    weight_per_particle: f64,
) -> FusionResult<Vec<ChargedParticle>> {
    if n_particles == 0 {
        return Err(FusionError::PhysicsViolation(
            "n_particles must be >= 1".to_string(),
        ));
    }
    if !major_radius_m.is_finite() || major_radius_m <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "major_radius_m must be finite and > 0".to_string(),
        ));
    }
    if !z_m.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "z_m must be finite".to_string(),
        ));
    }
    if !kinetic_energy_mev.is_finite() || kinetic_energy_mev <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "kinetic_energy_mev must be finite and > 0".to_string(),
        ));
    }
    if !pitch_cos.is_finite() || !(-1.0..=1.0).contains(&pitch_cos) {
        return Err(FusionError::PhysicsViolation(
            "pitch_cos must be finite and in [-1, 1]".to_string(),
        ));
    }
    if !weight_per_particle.is_finite() || weight_per_particle <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "weight_per_particle must be finite and > 0".to_string(),
        ));
    }

    let r0 = major_radius_m;
    let energy_j = kinetic_energy_mev * 1.0e6 * ELEMENTARY_CHARGE_C;
    let speed = (2.0 * energy_j / ALPHA_MASS_KG).sqrt();
    let pitch = pitch_cos;
    let v_par = speed * pitch;
    let v_perp = speed * (1.0 - pitch * pitch).sqrt();
    let weight = weight_per_particle;

    let mut out = Vec::with_capacity(n_particles);
    for i in 0..n_particles {
        let phi = 2.0 * PI * (i as f64) / (n_particles as f64);
        let x = r0 * phi.cos();
        let y = r0 * phi.sin();
        let ex = -phi.sin();
        let ey = phi.cos();
        out.push(ChargedParticle {
            x_m: x,
            y_m: y,
            z_m,
            vx_m_s: v_perp * ex,
            vy_m_s: v_perp * ey,
            vz_m_s: v_par,
            charge_c: ALPHA_CHARGE_C,
            mass_kg: ALPHA_MASS_KG,
            weight,
        });
    }
    Ok(out)
}

/// Summarize kinetic state of a particle population.
pub fn summarize_particle_population(
    particles: &[ChargedParticle],
    runaway_threshold_mev: f64,
) -> ParticlePopulationSummary {
    if particles.is_empty() {
        return ParticlePopulationSummary {
            count: 0,
            mean_energy_mev: 0.0,
            p95_energy_mev: 0.0,
            max_energy_mev: 0.0,
            runaway_fraction: 0.0,
        };
    }

    let mut energies: Vec<f64> = particles
        .iter()
        .map(ChargedParticle::kinetic_energy_mev)
        .collect();
    energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let count = energies.len();
    let mean_energy_mev = energies.iter().sum::<f64>() / (count as f64);
    let p95_idx = ((count - 1) as f64 * 0.95).round() as usize;
    let p95_energy_mev = energies[p95_idx.min(count - 1)];
    let max_energy_mev = *energies.last().unwrap_or(&0.0);
    let threshold = runaway_threshold_mev.max(0.0);
    let n_runaway = energies.iter().filter(|&&e| e >= threshold).count();
    let runaway_fraction = (n_runaway as f64) / (count as f64);

    ParticlePopulationSummary {
        count,
        mean_energy_mev,
        p95_energy_mev,
        max_energy_mev,
        runaway_fraction,
    }
}

/// Estimate alpha-particle heating power density [W/m^3] on the R-Z grid.
pub fn estimate_alpha_heating_profile(
    particles: &[ChargedParticle],
    grid: &Grid2D,
    confinement_tau_s: f64,
) -> Array2<f64> {
    let mut heat = Array2::zeros((grid.nz, grid.nr));
    if particles.is_empty() {
        return heat;
    }

    let tau = confinement_tau_s.max(1e-6);
    let cell_volume = (grid.dr.abs() * grid.dz.abs() * 2.0 * PI).max(MIN_CELL_AREA_M2);
    let r_min = grid.r[0];
    let r_max = grid.r[grid.nr - 1];
    let z_min = grid.z[0];
    let z_max = grid.z[grid.nz - 1];

    for particle in particles {
        let r = particle.cylindrical_radius_m();
        let z = particle.z_m;
        if r < r_min || r > r_max || z < z_min || z > z_max {
            continue;
        }
        let ir = nearest_index(&grid.r, r);
        let iz = nearest_index(&grid.z, z);
        let p_w = particle.kinetic_energy_j() * particle.weight / tau;
        let local_volume = (cell_volume * r.max(MIN_RADIUS_M)).max(MIN_CELL_AREA_M2);
        heat[[iz, ir]] += p_w / local_volume;
    }
    heat
}

/// Advance one particle state using the Boris push.
pub fn boris_push_step(
    particle: &mut ChargedParticle,
    electric_v_m: [f64; 3],
    magnetic_t: [f64; 3],
    dt_s: f64,
) {
    if particle.mass_kg <= 0.0 || !dt_s.is_finite() || dt_s <= 0.0 {
        return;
    }

    let qmdt2 = particle.charge_c * dt_s / (2.0 * particle.mass_kg);
    let v_minus = [
        particle.vx_m_s + qmdt2 * electric_v_m[0],
        particle.vy_m_s + qmdt2 * electric_v_m[1],
        particle.vz_m_s + qmdt2 * electric_v_m[2],
    ];

    let t = [
        qmdt2 * magnetic_t[0],
        qmdt2 * magnetic_t[1],
        qmdt2 * magnetic_t[2],
    ];
    let t2 = dot(t, t);
    let s = [
        (2.0 * t[0]) / (1.0 + t2),
        (2.0 * t[1]) / (1.0 + t2),
        (2.0 * t[2]) / (1.0 + t2),
    ];

    let v_prime = {
        let c = cross(v_minus, t);
        [v_minus[0] + c[0], v_minus[1] + c[1], v_minus[2] + c[2]]
    };
    let v_plus = {
        let c = cross(v_prime, s);
        [v_minus[0] + c[0], v_minus[1] + c[1], v_minus[2] + c[2]]
    };

    let vx_new = v_plus[0] + qmdt2 * electric_v_m[0];
    let vy_new = v_plus[1] + qmdt2 * electric_v_m[1];
    let vz_new = v_plus[2] + qmdt2 * electric_v_m[2];

    particle.vx_m_s = vx_new;
    particle.vy_m_s = vy_new;
    particle.vz_m_s = vz_new;
    particle.x_m += vx_new * dt_s;
    particle.y_m += vy_new * dt_s;
    particle.z_m += vz_new * dt_s;
}

/// Advance a particle set for a fixed number of Boris steps.
pub fn advance_particles_boris(
    particles: &mut [ChargedParticle],
    electric_v_m: [f64; 3],
    magnetic_t: [f64; 3],
    dt_s: f64,
    steps: usize,
) {
    for _ in 0..steps {
        for particle in particles.iter_mut() {
            boris_push_step(particle, electric_v_m, magnetic_t, dt_s);
        }
    }
}

/// Deposit particle toroidal current density J_phi on the GS R-Z grid.
pub fn deposit_toroidal_current_density(
    particles: &[ChargedParticle],
    grid: &Grid2D,
) -> Array2<f64> {
    let mut j_phi = Array2::zeros((grid.nz, grid.nr));
    let area = (grid.dr.abs() * grid.dz.abs()).max(MIN_CELL_AREA_M2);
    let r_min = grid.r[0];
    let r_max = grid.r[grid.nr - 1];
    let z_min = grid.z[0];
    let z_max = grid.z[grid.nz - 1];

    for particle in particles {
        let r = particle.cylindrical_radius_m();
        let z = particle.z_m;
        if r < r_min || r > r_max || z < z_min || z > z_max {
            continue;
        }

        let ir = nearest_index(&grid.r, r);
        let iz = nearest_index(&grid.z, z);
        let v_phi = particle.toroidal_velocity_m_s();
        let j_contrib = particle.charge_c * particle.weight * v_phi / area;
        j_phi[[iz, ir]] += j_contrib;
    }

    j_phi
}

/// Blend fluid and particle current maps and renormalize integral to `i_target`.
pub fn blend_particle_current(
    fluid_j_phi: &Array2<f64>,
    particle_j_phi: &Array2<f64>,
    grid: &Grid2D,
    i_target: f64,
    particle_coupling: f64,
) -> FusionResult<Array2<f64>> {
    let expected_shape = (grid.nz, grid.nr);
    if fluid_j_phi.dim() != expected_shape || particle_j_phi.dim() != expected_shape {
        return Err(FusionError::PhysicsViolation(format!(
            "Particle-current blend shape mismatch: expected {:?}, fluid {:?}, particle {:?}",
            expected_shape,
            fluid_j_phi.dim(),
            particle_j_phi.dim(),
        )));
    }
    if !particle_coupling.is_finite() || !(0.0..=1.0).contains(&particle_coupling) {
        return Err(FusionError::PhysicsViolation(
            "particle_coupling must be finite and in [0, 1]".to_string(),
        ));
    }
    if !i_target.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "i_target must be finite".to_string(),
        ));
    }

    let coupling = particle_coupling;
    let fluid_weight = 1.0 - coupling;
    let mut combined = Array2::zeros(expected_shape);

    for iz in 0..grid.nz {
        for ir in 0..grid.nr {
            combined[[iz, ir]] =
                fluid_weight * fluid_j_phi[[iz, ir]] + coupling * particle_j_phi[[iz, ir]];
        }
    }

    let i_current = combined.iter().sum::<f64>() * grid.dr * grid.dz;
    if i_current.abs() > MIN_CURRENT_INTEGRAL {
        let scale = i_target / i_current;
        combined.mapv_inplace(|v| v * scale);
    } else {
        combined.fill(0.0);
    }

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boris_push_preserves_speed_without_electric_field() {
        let particle = ChargedParticle {
            x_m: 2.0,
            y_m: 0.0,
            z_m: 0.0,
            vx_m_s: 80_000.0,
            vy_m_s: 10_000.0,
            vz_m_s: 5_000.0,
            charge_c: 1.602_176_634e-19,
            mass_kg: 1.672_621_923_69e-27,
            weight: 1.0,
        };
        let speed_0 =
            (particle.vx_m_s.powi(2) + particle.vy_m_s.powi(2) + particle.vz_m_s.powi(2)).sqrt();
        let mut particles = vec![particle];
        advance_particles_boris(&mut particles, [0.0, 0.0, 0.0], [0.0, 0.0, 2.5], 5e-10, 600);
        let updated = particles[0];
        let speed_1 =
            (updated.vx_m_s.powi(2) + updated.vy_m_s.powi(2) + updated.vz_m_s.powi(2)).sqrt();
        let rel = (speed_1 - speed_0).abs() / speed_0;
        assert!(rel < 1e-10, "Speed drift too high for Boris push: {rel}");
    }

    #[test]
    fn test_toroidal_current_deposition_is_nonzero() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -4.0, 4.0);
        let particles = vec![
            ChargedParticle {
                x_m: 5.0,
                y_m: 0.0,
                z_m: 0.0,
                vx_m_s: 0.0,
                vy_m_s: 100_000.0,
                vz_m_s: 0.0,
                charge_c: 1.602_176_634e-19,
                mass_kg: 1.672_621_923_69e-27,
                weight: 2.0e16,
            },
            ChargedParticle {
                x_m: 5.0,
                y_m: 0.2,
                z_m: 0.3,
                vx_m_s: 10_000.0,
                vy_m_s: 90_000.0,
                vz_m_s: -2_000.0,
                charge_c: 1.602_176_634e-19,
                mass_kg: 1.672_621_923_69e-27,
                weight: 1.5e16,
            },
        ];

        let j = deposit_toroidal_current_density(&particles, &grid);
        let sum_abs = j.iter().map(|v| v.abs()).sum::<f64>();
        assert!(
            sum_abs > 0.0,
            "Expected non-zero toroidal current deposition"
        );
    }

    #[test]
    fn test_blend_particle_current_renormalizes_target_current() {
        let grid = Grid2D::new(8, 8, 1.0, 5.0, -2.0, 2.0);
        let fluid = Array2::from_elem((8, 8), 2.0);
        let particle = Array2::from_elem((8, 8), 6.0);
        let i_target = 15.0e6;
        let blended = blend_particle_current(&fluid, &particle, &grid, i_target, 0.25).unwrap();
        let i_actual = blended.iter().sum::<f64>() * grid.dr * grid.dz;
        let rel = ((i_actual - i_target) / i_target).abs();
        assert!(
            rel < 1e-12,
            "Blended current should match target after renormalization"
        );
    }

    #[test]
    fn test_blend_particle_current_shape_mismatch_errors() {
        let grid = Grid2D::new(8, 8, 1.0, 5.0, -2.0, 2.0);
        let fluid = Array2::zeros((8, 8));
        let particle_bad = Array2::zeros((7, 8));
        let err = blend_particle_current(&fluid, &particle_bad, &grid, 1.0, 0.5).unwrap_err();
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("shape mismatch"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_blend_particle_current_rejects_invalid_coupling_and_target() {
        let grid = Grid2D::new(8, 8, 1.0, 5.0, -2.0, 2.0);
        let fluid = Array2::from_elem((8, 8), 2.0);
        let particle = Array2::from_elem((8, 8), 6.0);
        for bad_coupling in [f64::NAN, -0.1, 1.1] {
            let err = blend_particle_current(&fluid, &particle, &grid, 1.0, bad_coupling)
                .expect_err("invalid coupling must error");
            match err {
                FusionError::PhysicsViolation(msg) => {
                    assert!(msg.contains("particle_coupling"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
        let err = blend_particle_current(&fluid, &particle, &grid, f64::INFINITY, 0.5)
            .expect_err("non-finite target current must error");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("i_target"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_seed_alpha_particles_matches_requested_energy_band() {
        let particles =
            seed_alpha_test_particles(24, 6.2, 0.0, 3.5, 0.6, 5.0e12).expect("valid seeds");
        assert_eq!(particles.len(), 24);
        let summary = summarize_particle_population(&particles, 2.0);
        assert!(summary.mean_energy_mev > 3.0);
        assert!(summary.max_energy_mev < 4.2);
        assert!(summary.runaway_fraction > 0.9);
    }

    #[test]
    fn test_alpha_heating_profile_is_positive_when_particles_in_domain() {
        let grid = Grid2D::new(33, 33, 3.0, 9.0, -2.5, 2.5);
        let particles =
            seed_alpha_test_particles(16, 6.0, 0.1, 3.5, 0.4, 1.0e13).expect("valid seeds");
        let heat = estimate_alpha_heating_profile(&particles, &grid, 0.25);
        let total = heat.iter().sum::<f64>();
        assert!(total > 0.0, "Expected positive deposited alpha heating");
        assert!(!heat.iter().any(|v| !v.is_finite()));
    }

    #[test]
    fn test_seed_alpha_particles_rejects_invalid_parameters() {
        let bad = [
            seed_alpha_test_particles(0, 6.2, 0.0, 3.5, 0.6, 1.0),
            seed_alpha_test_particles(8, 0.0, 0.0, 3.5, 0.6, 1.0),
            seed_alpha_test_particles(8, 6.2, f64::NAN, 3.5, 0.6, 1.0),
            seed_alpha_test_particles(8, 6.2, 0.0, -1.0, 0.6, 1.0),
            seed_alpha_test_particles(8, 6.2, 0.0, 3.5, 1.2, 1.0),
            seed_alpha_test_particles(8, 6.2, 0.0, 3.5, 0.6, 0.0),
        ];
        for candidate in bad {
            assert!(
                candidate.is_err(),
                "Expected invalid seed parameters to return an error"
            );
        }
    }

    #[test]
    fn test_population_summary_empty_is_zero() {
        let summary = summarize_particle_population(&[], 1.0);
        assert_eq!(summary.count, 0);
        assert_eq!(summary.mean_energy_mev, 0.0);
        assert_eq!(summary.runaway_fraction, 0.0);
    }
}
