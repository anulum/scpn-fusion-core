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

// Coulomb collision constants
const ELECTRON_MASS_KG: f64 = 9.109_383_701_5e-31;
const PROTON_MASS_KG: f64 = 1.672_621_923_69e-27;
const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12;
const BOLTZMANN_J_PER_KEV: f64 = 1.602_176_634e-16;

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

fn validate_particle_state(particle: &ChargedParticle, label: &str) -> FusionResult<()> {
    if !particle.x_m.is_finite() || !particle.y_m.is_finite() || !particle.z_m.is_finite() {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} position components must be finite"
        )));
    }
    if !particle.vx_m_s.is_finite() || !particle.vy_m_s.is_finite() || !particle.vz_m_s.is_finite()
    {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} velocity components must be finite"
        )));
    }
    if !particle.charge_c.is_finite() {
        return Err(FusionError::PhysicsViolation(format!(
            "{label}.charge_c must be finite"
        )));
    }
    if !particle.mass_kg.is_finite() || particle.mass_kg <= 0.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "{label}.mass_kg must be finite and > 0"
        )));
    }
    if !particle.weight.is_finite() || particle.weight <= 0.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "{label}.weight must be finite and > 0"
        )));
    }
    Ok(())
}

fn validate_particle_projection_grid(grid: &Grid2D, label: &str) -> FusionResult<()> {
    if grid.nr == 0 || grid.nz == 0 {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} requires non-empty grid dimensions"
        )));
    }
    if grid.r.len() != grid.nr || grid.z.len() != grid.nz {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} grid axis length mismatch: r_len={}, nr={}, z_len={}, nz={}",
            grid.r.len(),
            grid.nr,
            grid.z.len(),
            grid.nz
        )));
    }
    if grid.rr.dim() != (grid.nz, grid.nr) || grid.zz.dim() != (grid.nz, grid.nr) {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} grid mesh shape mismatch: rr={:?}, zz={:?}, expected=({}, {})",
            grid.rr.dim(),
            grid.zz.dim(),
            grid.nz,
            grid.nr
        )));
    }
    if grid.rr.iter().any(|v| !v.is_finite()) || grid.zz.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} grid mesh coordinates must be finite"
        )));
    }
    if !grid.dr.is_finite() || !grid.dz.is_finite() || grid.dr == 0.0 || grid.dz == 0.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} grid spacing must be finite and non-zero, got dr={}, dz={}",
            grid.dr, grid.dz
        )));
    }
    if grid.r.iter().any(|v| !v.is_finite()) || grid.z.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} grid axes must be finite"
        )));
    }
    Ok(())
}

fn nearest_index(axis: &Array1<f64>, value: f64, label: &str) -> FusionResult<usize> {
    if axis.is_empty() {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} axis must be non-empty"
        )));
    }
    if !value.is_finite() {
        return Err(FusionError::PhysicsViolation(format!(
            "{label} lookup coordinate must be finite"
        )));
    }
    let mut best_idx = 0usize;
    let mut best_dist = f64::INFINITY;
    for (idx, x) in axis.iter().copied().enumerate() {
        if !x.is_finite() {
            return Err(FusionError::PhysicsViolation(format!(
                "{label} axis contains non-finite coordinate at index {idx}"
            )));
        }
        let dist = (x - value).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    Ok(best_idx)
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
    if !energy_j.is_finite() || energy_j <= 0.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "seeded particle energy_j must be finite and > 0, got {energy_j}"
        )));
    }
    let speed = (2.0 * energy_j / ALPHA_MASS_KG).sqrt();
    if !speed.is_finite() || speed <= 0.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "seeded particle speed must be finite and > 0, got {speed}"
        )));
    }
    let pitch = pitch_cos;
    let v_par = speed * pitch;
    let perp_factor = (1.0 - pitch * pitch).max(0.0);
    let v_perp = speed * perp_factor.sqrt();
    if !v_par.is_finite() || !v_perp.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "seeded particle velocity components became non-finite".to_string(),
        ));
    }
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
) -> FusionResult<ParticlePopulationSummary> {
    if !runaway_threshold_mev.is_finite() || runaway_threshold_mev < 0.0 {
        return Err(FusionError::PhysicsViolation(
            "runaway_threshold_mev must be finite and >= 0".to_string(),
        ));
    }
    if particles.is_empty() {
        return Ok(ParticlePopulationSummary {
            count: 0,
            mean_energy_mev: 0.0,
            p95_energy_mev: 0.0,
            max_energy_mev: 0.0,
            runaway_fraction: 0.0,
        });
    }

    let mut energies: Vec<f64> = Vec::with_capacity(particles.len());
    for (idx, particle) in particles.iter().enumerate() {
        validate_particle_state(particle, &format!("particle[{idx}]"))?;
        let energy = particle.kinetic_energy_mev();
        if !energy.is_finite() || energy < 0.0 {
            return Err(FusionError::PhysicsViolation(format!(
                "particle[{idx}] kinetic energy must be finite and >= 0, got {energy}"
            )));
        }
        energies.push(energy);
    }
    energies.sort_by(f64::total_cmp);
    let count = energies.len();
    let mean_energy_mev = energies.iter().sum::<f64>() / (count as f64);
    if !mean_energy_mev.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "mean particle energy became non-finite".to_string(),
        ));
    }
    let p95_idx = ((count - 1) as f64 * 0.95).round() as usize;
    let p95_energy_mev = energies[p95_idx.min(count - 1)];
    let max_energy_mev = energies[count - 1];
    let n_runaway = energies
        .iter()
        .filter(|&&e| e >= runaway_threshold_mev)
        .count();
    let runaway_fraction = (n_runaway as f64) / (count as f64);
    if !runaway_fraction.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "runaway fraction became non-finite".to_string(),
        ));
    }

    Ok(ParticlePopulationSummary {
        count,
        mean_energy_mev,
        p95_energy_mev,
        max_energy_mev,
        runaway_fraction,
    })
}

/// Estimate alpha-particle heating power density [W/m^3] on the R-Z grid.
pub fn estimate_alpha_heating_profile(
    particles: &[ChargedParticle],
    grid: &Grid2D,
    confinement_tau_s: f64,
) -> FusionResult<Array2<f64>> {
    if !confinement_tau_s.is_finite() || confinement_tau_s <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "confinement_tau_s must be finite and > 0".to_string(),
        ));
    }
    validate_particle_projection_grid(grid, "alpha heating profile")?;
    let mut heat = Array2::zeros((grid.nz, grid.nr));
    if particles.is_empty() {
        return Ok(heat);
    }

    let tau = confinement_tau_s;
    let cell_volume = (grid.dr.abs() * grid.dz.abs() * 2.0 * PI).max(MIN_CELL_AREA_M2);
    let r_min = grid.r[0].min(grid.r[grid.nr - 1]);
    let r_max = grid.r[0].max(grid.r[grid.nr - 1]);
    let z_min = grid.z[0].min(grid.z[grid.nz - 1]);
    let z_max = grid.z[0].max(grid.z[grid.nz - 1]);

    for (idx, particle) in particles.iter().enumerate() {
        validate_particle_state(particle, &format!("particle[{idx}]"))?;
        let r = particle.cylindrical_radius_m();
        let z = particle.z_m;
        if r < r_min || r > r_max || z < z_min || z > z_max {
            continue;
        }
        let ir = nearest_index(&grid.r, r, "alpha heating R-axis")?;
        let iz = nearest_index(&grid.z, z, "alpha heating Z-axis")?;
        let p_w = particle.kinetic_energy_j() * particle.weight / tau;
        if !p_w.is_finite() {
            return Err(FusionError::PhysicsViolation(format!(
                "particle[{idx}] deposited heating power became non-finite"
            )));
        }
        let local_volume = (cell_volume * r.max(MIN_RADIUS_M)).max(MIN_CELL_AREA_M2);
        let contribution = p_w / local_volume;
        if !contribution.is_finite() {
            return Err(FusionError::PhysicsViolation(format!(
                "particle[{idx}] heating contribution became non-finite"
            )));
        }
        heat[[iz, ir]] += contribution;
    }
    if heat.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "alpha heating profile contains non-finite values".to_string(),
        ));
    }
    Ok(heat)
}

/// Advance one particle state using the Boris push.
pub fn boris_push_step(
    particle: &mut ChargedParticle,
    electric_v_m: [f64; 3],
    magnetic_t: [f64; 3],
    dt_s: f64,
) -> FusionResult<()> {
    validate_particle_state(particle, "particle")?;
    if electric_v_m.iter().any(|v| !v.is_finite()) || magnetic_t.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "electric_v_m and magnetic_t must be finite vectors".to_string(),
        ));
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "dt_s must be finite and > 0".to_string(),
        ));
    }

    let qmdt2 = particle.charge_c * dt_s / (2.0 * particle.mass_kg);
    if !qmdt2.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "boris qmdt2 became non-finite".to_string(),
        ));
    }
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
    if !vx_new.is_finite() || !vy_new.is_finite() || !vz_new.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "boris velocity update became non-finite".to_string(),
        ));
    }

    particle.vx_m_s = vx_new;
    particle.vy_m_s = vy_new;
    particle.vz_m_s = vz_new;
    particle.x_m += vx_new * dt_s;
    particle.y_m += vy_new * dt_s;
    particle.z_m += vz_new * dt_s;
    if !particle.x_m.is_finite() || !particle.y_m.is_finite() || !particle.z_m.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "boris position update became non-finite".to_string(),
        ));
    }
    Ok(())
}

/// Advance a particle set for a fixed number of Boris steps.
pub fn advance_particles_boris(
    particles: &mut [ChargedParticle],
    electric_v_m: [f64; 3],
    magnetic_t: [f64; 3],
    dt_s: f64,
    steps: usize,
) -> FusionResult<()> {
    if particles.is_empty() {
        return Err(FusionError::PhysicsViolation(
            "particles must be non-empty".to_string(),
        ));
    }
    if steps == 0 {
        return Err(FusionError::PhysicsViolation(
            "steps must be >= 1".to_string(),
        ));
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "dt_s must be finite and > 0".to_string(),
        ));
    }
    for _ in 0..steps {
        for (idx, particle) in particles.iter_mut().enumerate() {
            boris_push_step(particle, electric_v_m, magnetic_t, dt_s).map_err(|err| match err {
                FusionError::PhysicsViolation(msg) => FusionError::PhysicsViolation(format!(
                    "particle[{idx}] boris push failed: {msg}"
                )),
                other => other,
            })?;
        }
    }
    Ok(())
}

/// Deposit particle toroidal current density J_phi on the GS R-Z grid.
pub fn deposit_toroidal_current_density(
    particles: &[ChargedParticle],
    grid: &Grid2D,
) -> FusionResult<Array2<f64>> {
    validate_particle_projection_grid(grid, "particle current deposition")?;
    if particles.is_empty() {
        return Err(FusionError::PhysicsViolation(
            "particles must be non-empty".to_string(),
        ));
    }
    let mut j_phi: Array2<f64> = Array2::zeros((grid.nz, grid.nr));
    let area = (grid.dr.abs() * grid.dz.abs()).max(MIN_CELL_AREA_M2);
    let r_min = grid.r[0].min(grid.r[grid.nr - 1]);
    let r_max = grid.r[0].max(grid.r[grid.nr - 1]);
    let z_min = grid.z[0].min(grid.z[grid.nz - 1]);
    let z_max = grid.z[0].max(grid.z[grid.nz - 1]);

    for (idx, particle) in particles.iter().enumerate() {
        validate_particle_state(particle, &format!("particle[{idx}]"))?;
        let r = particle.cylindrical_radius_m();
        let z = particle.z_m;
        if r < r_min || r > r_max || z < z_min || z > z_max {
            continue;
        }

        let ir = nearest_index(&grid.r, r, "particle current R-axis")?;
        let iz = nearest_index(&grid.z, z, "particle current Z-axis")?;
        let v_phi = particle.toroidal_velocity_m_s();
        let j_contrib = particle.charge_c * particle.weight * v_phi / area;
        if !j_contrib.is_finite() {
            return Err(FusionError::PhysicsViolation(format!(
                "particle[{idx}] toroidal current contribution became non-finite"
            )));
        }
        j_phi[[iz, ir]] += j_contrib;
    }

    if j_phi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "deposited particle current contains non-finite values".to_string(),
        ));
    }
    Ok(j_phi)
}

/// Blend fluid and particle current maps and renormalize integral to `i_target`.
pub fn blend_particle_current(
    fluid_j_phi: &Array2<f64>,
    particle_j_phi: &Array2<f64>,
    grid: &Grid2D,
    i_target: f64,
    particle_coupling: f64,
) -> FusionResult<Array2<f64>> {
    validate_particle_projection_grid(grid, "particle current blend")?;
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
    if fluid_j_phi.iter().any(|v| !v.is_finite()) || particle_j_phi.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "fluid_j_phi and particle_j_phi must be finite".to_string(),
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
    if !i_current.is_finite() {
        return Err(FusionError::PhysicsViolation(
            "blended current integral became non-finite".to_string(),
        ));
    }
    if i_current.abs() > MIN_CURRENT_INTEGRAL {
        let scale = i_target / i_current;
        if !scale.is_finite() {
            return Err(FusionError::PhysicsViolation(
                "blended current scale became non-finite".to_string(),
            ));
        }
        combined.mapv_inplace(|v| v * scale);
    } else if i_target.abs() > MIN_CURRENT_INTEGRAL {
        return Err(FusionError::PhysicsViolation(format!(
            "cannot renormalize blended current: |i_current| <= {MIN_CURRENT_INTEGRAL} while |i_target| > {MIN_CURRENT_INTEGRAL}"
        )));
    } else {
        combined.fill(0.0);
    }
    if combined.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::PhysicsViolation(
            "blended current map contains non-finite values".to_string(),
        ));
    }

    Ok(combined)
}

// ═══════════════════════════════════════════════════════════════════════
// Coulomb Collision Operator (Fokker-Planck Monte Carlo)
// ═══════════════════════════════════════════════════════════════════════

/// Parameters describing the background plasma for Coulomb collisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoulombCollisionParams {
    /// Electron density [m^-3].
    pub n_e: f64,
    /// Electron temperature [keV].
    pub t_e_kev: f64,
    /// Ion temperature [keV].
    pub t_i_kev: f64,
    /// Ion mass number (e.g. 2 for deuterium).
    pub a_i: f64,
    /// Ion charge number.
    pub z_i: f64,
    /// Effective charge Z_eff.
    pub z_eff: f64,
}

fn validate_collision_params(p: &CoulombCollisionParams) -> FusionResult<()> {
    if !p.n_e.is_finite() || p.n_e <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "n_e must be finite and > 0".into(),
        ));
    }
    if !p.t_e_kev.is_finite() || p.t_e_kev <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "t_e_kev must be finite and > 0".into(),
        ));
    }
    if !p.t_i_kev.is_finite() || p.t_i_kev <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "t_i_kev must be finite and > 0".into(),
        ));
    }
    if !p.a_i.is_finite() || p.a_i <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "a_i must be finite and > 0".into(),
        ));
    }
    if !p.z_i.is_finite() || p.z_i <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "z_i must be finite and > 0".into(),
        ));
    }
    if !p.z_eff.is_finite() || p.z_eff < 1.0 {
        return Err(FusionError::PhysicsViolation(
            "z_eff must be finite and >= 1".into(),
        ));
    }
    Ok(())
}

/// Coulomb logarithm via NRL formula, clamped to [5, 30].
pub fn coulomb_logarithm(n_e_m3: f64, t_e_kev: f64) -> FusionResult<f64> {
    if !n_e_m3.is_finite() || n_e_m3 <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "n_e must be finite and > 0".into(),
        ));
    }
    if !t_e_kev.is_finite() || t_e_kev <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "t_e_kev must be finite and > 0".into(),
        ));
    }
    // NRL Formulary: ln Λ ≈ 24 − ln(sqrt(n_e) / T_e)  [T_e in eV]
    let t_e_ev = t_e_kev * 1000.0;
    let ln_lambda = 24.0 - (n_e_m3.sqrt() / t_e_ev).ln();
    Ok(ln_lambda.clamp(5.0, 30.0))
}

/// Spitzer slowing-down time [s] for a test particle on field electrons.
///
/// τ_s = 3(2π)^{3/2} ε₀² m_a T_e^{3/2} / (n_e Z_a² e⁴ m_e^{1/2} ln Λ)
pub fn spitzer_slowing_down_time(
    mass_kg: f64,
    charge_number: f64,
    n_e_m3: f64,
    t_e_kev: f64,
    ln_lambda: f64,
) -> FusionResult<f64> {
    if !mass_kg.is_finite() || mass_kg <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "mass_kg must be finite and > 0".into(),
        ));
    }
    if !charge_number.is_finite() || charge_number <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "charge_number must be > 0".into(),
        ));
    }
    if !n_e_m3.is_finite() || n_e_m3 <= 0.0 {
        return Err(FusionError::PhysicsViolation("n_e must be > 0".into()));
    }
    if !t_e_kev.is_finite() || t_e_kev <= 0.0 {
        return Err(FusionError::PhysicsViolation("t_e_kev must be > 0".into()));
    }
    if !ln_lambda.is_finite() || ln_lambda <= 0.0 {
        return Err(FusionError::PhysicsViolation(
            "ln_lambda must be > 0".into(),
        ));
    }
    let t_e_j = t_e_kev * BOLTZMANN_J_PER_KEV;
    let e = ELEMENTARY_CHARGE_C;
    let numerator = 3.0
        * (2.0 * PI).powf(1.5)
        * VACUUM_PERMITTIVITY
        * VACUUM_PERMITTIVITY
        * mass_kg
        * t_e_j.powf(1.5);
    let denominator =
        n_e_m3 * (charge_number * e).powi(2) * e * e * ELECTRON_MASS_KG.sqrt() * ln_lambda;
    let tau = numerator / denominator;
    if !tau.is_finite() || tau <= 0.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "spitzer time became non-physical: {tau}"
        )));
    }
    Ok(tau)
}

/// Critical velocity where electron drag equals ion drag [m/s].
pub fn critical_velocity(t_e_kev: f64, a_i: f64, z_i: f64, z_eff: f64) -> FusionResult<f64> {
    if !t_e_kev.is_finite() || t_e_kev <= 0.0 {
        return Err(FusionError::PhysicsViolation("t_e_kev must be > 0".into()));
    }
    if !a_i.is_finite() || a_i <= 0.0 {
        return Err(FusionError::PhysicsViolation("a_i must be > 0".into()));
    }
    if !z_i.is_finite() || z_i <= 0.0 {
        return Err(FusionError::PhysicsViolation("z_i must be > 0".into()));
    }
    if !z_eff.is_finite() || z_eff < 1.0 {
        return Err(FusionError::PhysicsViolation("z_eff must be >= 1".into()));
    }
    let t_e_j = t_e_kev * BOLTZMANN_J_PER_KEV;
    let v_te = (2.0 * t_e_j / ELECTRON_MASS_KG).sqrt();
    // v_c = v_te * (3 sqrt(π) m_e / (4 m_i))^{1/3} * Z_eff^{1/3}
    let mass_ratio = ELECTRON_MASS_KG / (a_i * PROTON_MASS_KG);
    let factor = (0.75 * PI.sqrt() * mass_ratio).powf(1.0 / 3.0);
    let v_c = v_te * factor * z_eff.powf(1.0 / 3.0);
    if !v_c.is_finite() || v_c <= 0.0 {
        return Err(FusionError::PhysicsViolation(format!(
            "critical velocity non-physical: {v_c}"
        )));
    }
    Ok(v_c)
}

/// Collision frequencies (ν_slow, ν_defl, ν_energy) for a test particle at given speed.
pub fn collision_frequencies(
    speed: f64,
    params: &CoulombCollisionParams,
    ln_lambda: f64,
    tau_s: f64,
    v_c: f64,
) -> FusionResult<(f64, f64, f64)> {
    if !speed.is_finite() || speed < 0.0 {
        return Err(FusionError::PhysicsViolation(
            "speed must be finite and >= 0".into(),
        ));
    }
    if !tau_s.is_finite() || tau_s <= 0.0 {
        return Err(FusionError::PhysicsViolation("tau_s must be > 0".into()));
    }
    if !v_c.is_finite() || v_c <= 0.0 {
        return Err(FusionError::PhysicsViolation("v_c must be > 0".into()));
    }
    let _ = ln_lambda; // used indirectly via tau_s
    let _ = params;

    let v_safe = speed.max(1e-6);
    let x3 = (v_safe / v_c).powi(3);

    // Slowing-down frequency: ν_s = (1 + x³) / τ_s  where x = v/v_c
    let nu_slow = (1.0 + x3) / tau_s;

    // Deflection (pitch-angle scattering): ν_d ≈ (1 + Z_eff/2) / (τ_s · x³)
    let nu_defl = if x3 > 1e-30 {
        1.0 / (tau_s * x3)
    } else {
        1.0 / (tau_s * 1e-30)
    };

    // Energy diffusion: ν_ε ≈ 2 T_e / (m v² τ_s)
    let nu_energy = nu_slow * 0.5; // simplified

    Ok((nu_slow, nu_defl, nu_energy))
}

/// XORshift64 PRNG: returns next state and a uniform f64 in [0, 1).
fn xorshift_uniform(state: &mut u64) -> f64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    (s as f64) / (u64::MAX as f64)
}

/// Box-Muller transform producing one standard normal variate from xorshift.
fn xorshift_normal(state: &mut u64) -> f64 {
    let u1 = xorshift_uniform(state).max(1e-300);
    let u2 = xorshift_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Find an orthonormal basis (e1, e2) perpendicular to unit vector `v_hat`.
fn perpendicular_basis(v_hat: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Choose the axis most perpendicular to v_hat
    let abs_x = v_hat[0].abs();
    let abs_y = v_hat[1].abs();
    let abs_z = v_hat[2].abs();
    let seed = if abs_x <= abs_y && abs_x <= abs_z {
        [1.0, 0.0, 0.0]
    } else if abs_y <= abs_z {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };

    // e1 = normalize(seed × v_hat)
    let raw = cross(seed, v_hat);
    let norm = dot(raw, raw).sqrt().max(1e-30);
    let e1 = [raw[0] / norm, raw[1] / norm, raw[2] / norm];

    // e2 = v_hat × e1
    let e2 = cross(v_hat, e1);
    (e1, e2)
}

/// Apply Coulomb collision Monte Carlo kick to a single particle.
///
/// Uses Langevin approach: drag (slowing-down) + stochastic pitch-angle
/// scattering and energy diffusion over timestep dt_s.
pub fn collision_step(
    particle: &mut ChargedParticle,
    params: &CoulombCollisionParams,
    dt_s: f64,
    rng_state: &mut u64,
) -> FusionResult<()> {
    validate_particle_state(particle, "collision particle")?;
    validate_collision_params(params)?;
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(FusionError::PhysicsViolation("dt_s must be > 0".into()));
    }

    let v = [particle.vx_m_s, particle.vy_m_s, particle.vz_m_s];
    let speed = dot(v, v).sqrt();
    if speed < 1e-10 {
        return Ok(()); // particle at rest — no collision
    }

    let ln_lam = coulomb_logarithm(params.n_e, params.t_e_kev)?;
    let za = (particle.charge_c / ELEMENTARY_CHARGE_C).abs();
    let tau_s =
        spitzer_slowing_down_time(particle.mass_kg, za, params.n_e, params.t_e_kev, ln_lam)?;
    let v_c = critical_velocity(params.t_e_kev, params.a_i, params.z_i, params.z_eff)?;
    let (nu_slow, nu_defl, _nu_energy) = collision_frequencies(speed, params, ln_lam, tau_s, v_c)?;

    // Unit velocity direction
    let v_hat = [v[0] / speed, v[1] / speed, v[2] / speed];

    // 1. Slowing-down drag: Δv_∥ = -ν_s · v · dt
    let dv_par = -nu_slow * speed * dt_s;

    // 2. Pitch-angle scattering: random perpendicular kick
    //    σ_perp = v · sqrt(ν_d · dt)
    let sigma_perp = speed * (nu_defl * dt_s).sqrt();
    let kick1 = xorshift_normal(rng_state) * sigma_perp;
    let kick2 = xorshift_normal(rng_state) * sigma_perp;

    let (e1, e2) = perpendicular_basis(v_hat);

    // New velocity = (v + dv_par) v_hat + kick1 e1 + kick2 e2
    let new_speed_par = speed + dv_par;
    // Ensure speed doesn't go negative (particle thermalized)
    let new_speed_par = new_speed_par.max(0.0);

    particle.vx_m_s = new_speed_par * v_hat[0] + kick1 * e1[0] + kick2 * e2[0];
    particle.vy_m_s = new_speed_par * v_hat[1] + kick1 * e1[1] + kick2 * e2[1];
    particle.vz_m_s = new_speed_par * v_hat[2] + kick1 * e1[2] + kick2 * e2[2];

    if !particle.vx_m_s.is_finite() || !particle.vy_m_s.is_finite() || !particle.vz_m_s.is_finite()
    {
        return Err(FusionError::PhysicsViolation(
            "collision step produced non-finite velocity".into(),
        ));
    }
    Ok(())
}

/// Apply Coulomb collisions to all particles in a batch.
pub fn apply_coulomb_collisions(
    particles: &mut [ChargedParticle],
    params: &CoulombCollisionParams,
    dt_s: f64,
    seed: u64,
) -> FusionResult<()> {
    validate_collision_params(params)?;
    if particles.is_empty() {
        return Ok(());
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(FusionError::PhysicsViolation("dt_s must be > 0".into()));
    }
    if seed == 0 {
        return Err(FusionError::PhysicsViolation(
            "seed must be != 0 for xorshift".into(),
        ));
    }

    for (idx, particle) in particles.iter_mut().enumerate() {
        // Each particle gets a deterministic RNG state derived from seed + index
        let mut rng = seed
            .wrapping_add(idx as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        if rng == 0 {
            rng = 1;
        }
        collision_step(particle, params, dt_s, &mut rng).map_err(|e| match e {
            FusionError::PhysicsViolation(msg) => {
                FusionError::PhysicsViolation(format!("particle[{idx}] collision failed: {msg}"))
            }
            other => other,
        })?;
    }
    Ok(())
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
        advance_particles_boris(&mut particles, [0.0, 0.0, 0.0], [0.0, 0.0, 2.5], 5e-10, 600)
            .expect("valid boris advance should succeed");
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

        let j = deposit_toroidal_current_density(&particles, &grid)
            .expect("valid particles/grid should deposit current");
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
        let summary = summarize_particle_population(&particles, 2.0).expect("valid threshold");
        assert!(summary.mean_energy_mev > 3.0);
        assert!(summary.max_energy_mev < 4.2);
        assert!(summary.runaway_fraction > 0.9);
    }

    #[test]
    fn test_alpha_heating_profile_is_positive_when_particles_in_domain() {
        let grid = Grid2D::new(33, 33, 3.0, 9.0, -2.5, 2.5);
        let particles =
            seed_alpha_test_particles(16, 6.0, 0.1, 3.5, 0.4, 1.0e13).expect("valid seeds");
        let heat =
            estimate_alpha_heating_profile(&particles, &grid, 0.25).expect("valid confinement");
        let total = heat.iter().sum::<f64>();
        assert!(total > 0.0, "Expected positive deposited alpha heating");
        assert!(!heat.iter().any(|v| !v.is_finite()));
    }

    #[test]
    fn test_alpha_heating_profile_supports_descending_axes() {
        let grid = Grid2D::new(33, 33, 9.0, 3.0, 2.5, -2.5);
        let particles =
            seed_alpha_test_particles(16, 6.0, 0.1, 3.5, 0.4, 1.0e13).expect("valid seeds");
        let heat =
            estimate_alpha_heating_profile(&particles, &grid, 0.25).expect("valid confinement");
        let total = heat.iter().sum::<f64>();
        assert!(
            total > 0.0,
            "Expected positive deposited alpha heating on descending axes"
        );
    }

    #[test]
    fn test_alpha_heating_profile_rejects_invalid_confinement_time() {
        let grid = Grid2D::new(17, 17, 3.0, 9.0, -1.5, 1.5);
        let particles =
            seed_alpha_test_particles(8, 6.0, 0.0, 3.5, 0.2, 1.0e12).expect("valid seeds");
        for bad_tau in [0.0, f64::NAN] {
            let err = estimate_alpha_heating_profile(&particles, &grid, bad_tau)
                .expect_err("invalid confinement time must error");
            match err {
                FusionError::PhysicsViolation(msg) => {
                    assert!(msg.contains("confinement_tau_s"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn test_blend_particle_current_rejects_non_finite_maps() {
        let grid = Grid2D::new(8, 8, 1.0, 5.0, -2.0, 2.0);
        let mut fluid = Array2::from_elem((8, 8), 2.0);
        let particle = Array2::from_elem((8, 8), 6.0);
        fluid[[0, 0]] = f64::NAN;
        let err = blend_particle_current(&fluid, &particle, &grid, 1.0, 0.5)
            .expect_err("non-finite current map must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("must be finite"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_blend_particle_current_rejects_zero_integral_nonzero_target() {
        let grid = Grid2D::new(8, 8, 1.0, 5.0, -2.0, 2.0);
        let fluid = Array2::zeros((8, 8));
        let particle = Array2::zeros((8, 8));
        let err = blend_particle_current(&fluid, &particle, &grid, 1.0e6, 0.5)
            .expect_err("non-zero target with zero blended current must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("cannot renormalize blended current"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_blend_particle_current_rejects_invalid_grid_spacing() {
        let mut grid = Grid2D::new(8, 8, 1.0, 5.0, -2.0, 2.0);
        grid.dr = 0.0;
        let fluid = Array2::from_elem((8, 8), 2.0);
        let particle = Array2::from_elem((8, 8), 6.0);
        let err = blend_particle_current(&fluid, &particle, &grid, 1.0, 0.5)
            .expect_err("invalid grid spacing must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("grid spacing"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
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
    fn test_seed_alpha_particles_rejects_non_finite_kinematics() {
        let err = seed_alpha_test_particles(4, 6.2, 0.0, f64::MAX, 0.6, 1.0)
            .expect_err("overflowing kinetic seed should fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("energy_j") || msg.contains("speed"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_population_summary_empty_is_zero() {
        let summary = summarize_particle_population(&[], 1.0).expect("valid threshold");
        assert_eq!(summary.count, 0);
        assert_eq!(summary.mean_energy_mev, 0.0);
        assert_eq!(summary.runaway_fraction, 0.0);
    }

    #[test]
    fn test_population_summary_rejects_invalid_threshold() {
        let particles = seed_alpha_test_particles(4, 6.2, 0.0, 3.5, 0.2, 1.0).expect("valid seeds");
        for bad in [f64::NAN, -0.01] {
            let err = summarize_particle_population(&particles, bad)
                .expect_err("invalid threshold must error");
            match err {
                FusionError::PhysicsViolation(msg) => {
                    assert!(msg.contains("runaway_threshold_mev"));
                }
                other => panic!("Unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn test_blend_particle_current_rejects_non_finite_grid_mesh() {
        let mut grid = Grid2D::new(8, 8, 1.0, 5.0, -2.0, 2.0);
        grid.rr[[0, 0]] = f64::NAN;
        let fluid = Array2::from_elem((8, 8), 2.0);
        let particle = Array2::from_elem((8, 8), 6.0);
        let err = blend_particle_current(&fluid, &particle, &grid, 1.0, 0.5)
            .expect_err("non-finite grid mesh must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("mesh coordinates"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_population_summary_rejects_non_finite_particle_state() {
        let mut particles =
            seed_alpha_test_particles(4, 6.2, 0.0, 3.5, 0.2, 1.0).expect("valid seeds");
        particles[0].vx_m_s = f64::INFINITY;
        let err = summarize_particle_population(&particles, 1.0)
            .expect_err("non-finite particle state must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("velocity"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_toroidal_current_deposition_rejects_invalid_particle_state() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -4.0, 4.0);
        let mut particles =
            seed_alpha_test_particles(4, 6.2, 0.0, 3.5, 0.2, 1.0).expect("valid seeds");
        particles[1].weight = f64::NAN;
        let err = deposit_toroidal_current_density(&particles, &grid)
            .expect_err("invalid particle state must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("weight"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_toroidal_current_deposition_rejects_empty_population() {
        let grid = Grid2D::new(17, 17, 1.0, 9.0, -4.0, 4.0);
        let err = deposit_toroidal_current_density(&[], &grid)
            .expect_err("empty particle population must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("non-empty"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_toroidal_current_deposition_supports_descending_axes() {
        let grid = Grid2D::new(17, 17, 9.0, 1.0, 4.0, -4.0);
        let particles = vec![ChargedParticle {
            x_m: 5.0,
            y_m: 0.0,
            z_m: 0.0,
            vx_m_s: 0.0,
            vy_m_s: 100_000.0,
            vz_m_s: 0.0,
            charge_c: 1.602_176_634e-19,
            mass_kg: 1.672_621_923_69e-27,
            weight: 2.0e16,
        }];
        let j = deposit_toroidal_current_density(&particles, &grid)
            .expect("descending axes should still deposit current");
        let sum_abs = j.iter().map(|v| v.abs()).sum::<f64>();
        assert!(
            sum_abs > 0.0,
            "Expected non-zero toroidal deposition on descending axes"
        );
    }

    #[test]
    fn test_boris_push_rejects_invalid_runtime_inputs() {
        let mut particle =
            seed_alpha_test_particles(1, 6.2, 0.0, 3.5, 0.2, 1.0).expect("valid seeds")[0];
        let err = boris_push_step(&mut particle, [0.0, 0.0, f64::NAN], [0.0, 0.0, 2.5], 1e-9)
            .expect_err("non-finite electric field must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("electric_v_m"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
        let err = boris_push_step(&mut particle, [0.0, 0.0, 0.0], [0.0, 0.0, 2.5], 0.0)
            .expect_err("non-positive dt must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("dt_s"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_advance_particles_boris_rejects_invalid_particle_state() {
        let mut particles =
            seed_alpha_test_particles(2, 6.2, 0.0, 3.5, 0.2, 1.0).expect("valid seeds");
        particles[0].mass_kg = 0.0;
        let err =
            advance_particles_boris(&mut particles, [0.0, 0.0, 0.0], [0.0, 0.0, 2.5], 1e-9, 1)
                .expect_err("invalid particle mass must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("particle[0]"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_advance_particles_boris_rejects_empty_slice_and_zero_steps() {
        let mut empty: Vec<ChargedParticle> = Vec::new();
        let err = advance_particles_boris(&mut empty, [0.0, 0.0, 0.0], [0.0, 0.0, 2.5], 1e-9, 1)
            .expect_err("empty particle slice must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("non-empty"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }

        let mut particles =
            seed_alpha_test_particles(1, 6.2, 0.0, 3.5, 0.2, 1.0).expect("valid seeds");
        let err =
            advance_particles_boris(&mut particles, [0.0, 0.0, 0.0], [0.0, 0.0, 2.5], 1e-9, 0)
                .expect_err("zero steps must fail");
        match err {
            FusionError::PhysicsViolation(msg) => {
                assert!(msg.contains("steps"));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Coulomb collision tests
    // ═══════════════════════════════════════════════════════════════════

    fn default_collision_params() -> CoulombCollisionParams {
        CoulombCollisionParams {
            n_e: 1.0e20,
            t_e_kev: 10.0,
            t_i_kev: 8.0,
            a_i: 2.0, // deuterium
            z_i: 1.0,
            z_eff: 1.5,
        }
    }

    #[test]
    fn test_coulomb_logarithm_range() {
        let ln_lam = coulomb_logarithm(1e20, 10.0).unwrap();
        assert!((5.0..=30.0).contains(&ln_lam), "ln_lambda={ln_lam}");
        // For fusion plasma, NRL formula gives ln Λ ≈ 10-20 range
        assert!(
            ln_lam > 5.0 && ln_lam < 25.0,
            "fusion plasma ln_lambda={ln_lam}"
        );
    }

    #[test]
    fn test_coulomb_logarithm_clamp() {
        // Very cold dense plasma -> clamped to 5
        let low = coulomb_logarithm(1e30, 0.001).unwrap();
        assert!((low - 5.0).abs() < 1e-12, "should clamp to 5, got {low}");
        // Moderately hot tenuous plasma -> still within [5,30]
        let mid = coulomb_logarithm(1e10, 1000.0).unwrap();
        assert!(
            (5.0..=30.0).contains(&mid),
            "should be in [5,30], got {mid}"
        );
    }

    #[test]
    fn test_coulomb_logarithm_rejects_invalid() {
        assert!(coulomb_logarithm(-1.0, 10.0).is_err());
        assert!(coulomb_logarithm(1e20, 0.0).is_err());
        assert!(coulomb_logarithm(1e20, f64::NAN).is_err());
    }

    #[test]
    fn test_spitzer_slowing_down_time() {
        let ln_lam = coulomb_logarithm(1e20, 10.0).unwrap();
        let tau = spitzer_slowing_down_time(ALPHA_MASS_KG, 2.0, 1e20, 10.0, ln_lam).unwrap();
        // For 3.5 MeV alpha in ITER plasma, τ_s ~ 0.1-1.0 s
        assert!(
            tau > 0.01 && tau < 10.0,
            "spitzer time {tau} s out of range"
        );
    }

    #[test]
    fn test_critical_velocity() {
        let v_c = critical_velocity(10.0, 2.0, 1.0, 1.5).unwrap();
        // v_c should be much less than speed of light but significant
        assert!(v_c > 1e5 && v_c < 1e8, "v_c={v_c} m/s");
    }

    #[test]
    fn test_collision_frequencies_positive() {
        let params = default_collision_params();
        let ln_lam = coulomb_logarithm(params.n_e, params.t_e_kev).unwrap();
        let tau_s =
            spitzer_slowing_down_time(ALPHA_MASS_KG, 2.0, params.n_e, params.t_e_kev, ln_lam)
                .unwrap();
        let v_c = critical_velocity(params.t_e_kev, params.a_i, params.z_i, params.z_eff).unwrap();
        let speed = 1e7; // fast alpha
        let (nu_s, nu_d, nu_e) = collision_frequencies(speed, &params, ln_lam, tau_s, v_c).unwrap();
        assert!(nu_s > 0.0, "slowing-down frequency must be > 0");
        assert!(nu_d > 0.0, "deflection frequency must be > 0");
        assert!(nu_e > 0.0, "energy diffusion frequency must be > 0");
    }

    #[test]
    fn test_collision_step_slows_alpha() {
        let params = default_collision_params();
        let mut particles = seed_alpha_test_particles(1, 6.2, 0.0, 3.5, 0.6, 1e12).unwrap();
        let e0 = particles[0].kinetic_energy_mev();
        let mut rng: u64 = 12345;
        // Many small collision steps should reduce energy
        for _ in 0..100 {
            collision_step(&mut particles[0], &params, 1e-3, &mut rng).unwrap();
        }
        let e1 = particles[0].kinetic_energy_mev();
        assert!(e1 < e0, "collisions should slow alpha: {e0} -> {e1} MeV");
    }

    #[test]
    fn test_batch_collisions_deterministic() {
        let params = default_collision_params();
        let mut p1 = seed_alpha_test_particles(8, 6.2, 0.0, 3.5, 0.6, 1e12).unwrap();
        let mut p2 = p1.clone();
        apply_coulomb_collisions(&mut p1, &params, 1e-4, 42).unwrap();
        apply_coulomb_collisions(&mut p2, &params, 1e-4, 42).unwrap();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a.vx_m_s - b.vx_m_s).abs() < 1e-12, "determinism broken");
            assert!((a.vy_m_s - b.vy_m_s).abs() < 1e-12, "determinism broken");
            assert!((a.vz_m_s - b.vz_m_s).abs() < 1e-12, "determinism broken");
        }
    }

    #[test]
    fn test_collision_rejects_invalid_params() {
        let mut bad = default_collision_params();
        bad.n_e = -1.0;
        let mut p = seed_alpha_test_particles(1, 6.2, 0.0, 3.5, 0.6, 1e12).unwrap();
        assert!(apply_coulomb_collisions(&mut p, &bad, 1e-4, 42).is_err());
    }

    #[test]
    fn test_perpendicular_basis_orthonormal() {
        for v_hat in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.577, 0.577, 0.577],
        ] {
            let norm = dot(v_hat, v_hat).sqrt();
            let v_hat = [v_hat[0] / norm, v_hat[1] / norm, v_hat[2] / norm];
            let (e1, e2) = perpendicular_basis(v_hat);
            // e1 ⊥ v_hat
            assert!(dot(e1, v_hat).abs() < 1e-10, "e1 not perp to v_hat");
            // e2 ⊥ v_hat
            assert!(dot(e2, v_hat).abs() < 1e-10, "e2 not perp to v_hat");
            // e1 ⊥ e2
            assert!(dot(e1, e2).abs() < 1e-10, "e1 not perp to e2");
            // |e1| ≈ 1, |e2| ≈ 1
            assert!((dot(e1, e1).sqrt() - 1.0).abs() < 1e-10);
            assert!((dot(e2, e2).sqrt() - 1.0).abs() < 1e-10);
        }
    }
}
