// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — FRC Rigid-Rotor Solver
//! Steinhauer no-rotation analytical FRC rigid-rotor solver.

use super::data::{FrcEquilibriumState, FrcSolverError, RigidRotorFrcInputs};
use ndarray::Array1;

const MU_0: f64 = 4.0 * std::f64::consts::PI * 1.0e-7;
const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;
const ATOMIC_MASS_KG: f64 = 1.660_539_066_60e-27;
const DEUTERIUM_MASS_AMU: f64 = 2.014;
const MODEL_NAME: &str = "steinhauer_2011_no_rotation_analytical";

/// Return thermal ion gyroradius in metres using `sqrt(2 m_i T_i) / (e B)`.
pub fn ion_gyroradius_m(t_i_ev: f64, b_t: f64, mass_amu: f64) -> Result<f64, FrcSolverError> {
    if !t_i_ev.is_finite() || t_i_ev <= 0.0 {
        return Err(FrcSolverError::InvalidInput("T_i_eV must be positive"));
    }
    if !b_t.is_finite() || b_t == 0.0 {
        return Err(FrcSolverError::InvalidInput("B_T must be non-zero"));
    }
    if !mass_amu.is_finite() || mass_amu <= 0.0 {
        return Err(FrcSolverError::InvalidInput("mass_amu must be positive"));
    }
    let ion_mass_kg = mass_amu * ATOMIC_MASS_KG;
    let thermal_momentum = (2.0 * ion_mass_kg * t_i_ev * ELEMENTARY_CHARGE_C).sqrt();
    Ok(thermal_momentum / (ELEMENTARY_CHARGE_C * b_t.abs()))
}

/// Return Steinhauer Eq. 27 `s = R_s^-1 integral_0^R_s r / rho_i(r) dr`.
pub fn s_parameter_from_profile(
    rho: &Array1<f64>,
    b_z: &Array1<f64>,
    r_s: f64,
    t_i_ev: f64,
    mass_amu: f64,
) -> Result<f64, FrcSolverError> {
    if !r_s.is_finite() || r_s <= 0.0 {
        return Err(FrcSolverError::InvalidInput("R_s must be positive"));
    }
    if !t_i_ev.is_finite() || t_i_ev <= 0.0 {
        return Err(FrcSolverError::InvalidInput("T_i_eV must be positive"));
    }
    if !mass_amu.is_finite() || mass_amu <= 0.0 {
        return Err(FrcSolverError::InvalidInput("mass_amu must be positive"));
    }
    if rho.len() != b_z.len() {
        return Err(FrcSolverError::InvalidInput(
            "rho and B_z profiles must have matching lengths",
        ));
    }
    let ion_mass_kg = mass_amu * ATOMIC_MASS_KG;
    let thermal_momentum = (2.0 * ion_mass_kg * t_i_ev * ELEMENTARY_CHARGE_C).sqrt();
    let (r_clip, b_clip) = clip_to_separatrix(rho, b_z, r_s)?;
    let integrand = Array1::from_iter(
        r_clip
            .iter()
            .zip(b_clip.iter())
            .map(|(r, b)| r * ELEMENTARY_CHARGE_C * b.abs() / thermal_momentum),
    );
    Ok(trapezoid(&r_clip, &integrand) / r_s)
}

/// Solve the Steinhauer no-rotation FRC analytical limit on a radial grid.
pub fn solve_frc_equilibrium(
    inputs: &RigidRotorFrcInputs,
    rho_grid: &Array1<f64>,
    tolerance: f64,
) -> Result<FrcEquilibriumState, FrcSolverError> {
    validate_inputs(inputs, tolerance)?;
    validate_grid(rho_grid, inputs.r_s)?;
    let rho = rho_grid.clone();
    let delta = match inputs.delta {
        Some(value) => value,
        None => ion_gyroradius_m(inputs.t_i_ev, inputs.b_ext, DEUTERIUM_MASS_AMU)?,
    };

    let argument = rho.mapv(|r| (r * r - inputs.r_s * inputs.r_s) / (2.0 * inputs.r_s * delta));
    let b_z = argument.mapv(|a| -inputs.b_ext * a.tanh());
    let b_theta = Array1::zeros(b_z.len());
    let psi = cylindrical_flux_from_bz(&rho, &b_z);
    let r_null = zero_crossing_radius(&rho, &b_z);
    let separatrix_index = nearest_index(&rho, r_null);

    let p0 = inputs.n0 * (inputs.t_i_ev + inputs.t_e_ev) * ELEMENTARY_CHARGE_C;
    let psi_axis = interpolate(&rho, &psi, r_null);
    let pressure_span = (inputs.b_ext * inputs.r_s).abs().max(tolerance);
    let p = psi.mapv(|value| p0 * (-2.0 * ((value - psi_axis) / pressure_span).powi(2)).exp());

    let force_balance_residual = radial_force_balance_residual(&rho, &b_z, &p);
    let grad_p = gradient_edge_order2(&rho, &p);
    let grad_bz = gradient_edge_order2(&rho, &b_z);
    let lorentz_scale = Array1::from_iter(
        b_z.iter()
            .zip(grad_bz.iter())
            .map(|(b, db)| ((b / MU_0) * db).abs()),
    );
    let residual_scale = tolerance.max(max_abs(&grad_p)).max(max_abs(&lorentz_scale));
    let force_balance_residual_linf = max_abs(&force_balance_residual) / residual_scale;
    let force_balance_residual_l2 = (force_balance_residual
        .iter()
        .map(|value| (value / residual_scale).powi(2))
        .sum::<f64>()
        / force_balance_residual.len() as f64)
        .sqrt();

    let magnetic_energy_density = b_z.mapv(|b| b * b / (2.0 * MU_0));
    let total_energy_density = &magnetic_energy_density + &p;
    let energy_integrand = Array1::from_iter(
        total_energy_density
            .iter()
            .zip(rho.iter())
            .map(|(density, r)| density * 2.0 * std::f64::consts::PI * r),
    );
    let energy_j = trapezoid(&rho, &energy_integrand);
    let pressure_integrand = Array1::from_iter(
        p.iter()
            .zip(rho.iter())
            .map(|(pressure, r)| pressure * 2.0 * std::f64::consts::PI * r),
    );
    let pressure_integral = trapezoid(&rho, &pressure_integrand);
    let external_pressure_energy = (inputs.b_ext * inputs.b_ext / (2.0 * MU_0))
        * std::f64::consts::PI
        * inputs.r_s
        * inputs.r_s;
    let pressure_balance_ratio = pressure_integral / external_pressure_energy.max(tolerance);
    let s_parameter =
        s_parameter_from_profile(&rho, &b_z, inputs.r_s, inputs.t_i_ev, DEUTERIUM_MASS_AMU)?;

    let residual = b_z
        .iter()
        .zip(argument.iter())
        .map(|(actual, arg)| (actual - (-inputs.b_ext * arg.tanh())).abs())
        .fold(0.0, f64::max);

    Ok(FrcEquilibriumState {
        rho,
        psi,
        b_z,
        b_theta,
        p,
        r_null,
        separatrix_index,
        s_parameter,
        energy_j,
        converged: true,
        residual,
        delta,
        pressure_balance_ratio,
        force_balance_residual,
        force_balance_residual_linf,
        force_balance_residual_l2,
        model: MODEL_NAME,
    })
}

fn validate_inputs(inputs: &RigidRotorFrcInputs, tolerance: f64) -> Result<(), FrcSolverError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(FrcSolverError::InvalidInput("tolerance must be positive"));
    }
    if !inputs.n0.is_finite() || inputs.n0 <= 0.0 {
        return Err(FrcSolverError::InvalidInput("n0 must be positive"));
    }
    if !inputs.t_i_ev.is_finite()
        || !inputs.t_e_ev.is_finite()
        || inputs.t_i_ev <= 0.0
        || inputs.t_e_ev <= 0.0
    {
        return Err(FrcSolverError::InvalidInput(
            "ion and electron temperatures must be positive",
        ));
    }
    if !inputs.r_s.is_finite() || inputs.r_s <= 0.0 {
        return Err(FrcSolverError::InvalidInput("R_s must be positive"));
    }
    if !inputs.b_ext.is_finite() || inputs.b_ext == 0.0 {
        return Err(FrcSolverError::InvalidInput("B_ext must be non-zero"));
    }
    if let Some(delta) = inputs.delta {
        if !delta.is_finite() || delta <= 0.0 {
            return Err(FrcSolverError::InvalidInput(
                "delta must be positive when provided",
            ));
        }
    }
    if !inputs.theta_dot.is_finite() {
        return Err(FrcSolverError::InvalidInput("theta_dot must be finite"));
    }
    if inputs.theta_dot.abs() > tolerance {
        return Err(FrcSolverError::RotatingBvpNotImplemented);
    }
    Ok(())
}

fn validate_grid(rho: &Array1<f64>, r_s: f64) -> Result<(), FrcSolverError> {
    if rho.len() < 4 {
        return Err(FrcSolverError::InvalidInput(
            "rho_grid must contain at least four points",
        ));
    }
    if rho.iter().any(|value| !value.is_finite()) {
        return Err(FrcSolverError::InvalidInput(
            "rho_grid must contain finite values",
        ));
    }
    if rho[0] != 0.0 {
        return Err(FrcSolverError::InvalidInput(
            "rho_grid must start at the magnetic axis radius 0",
        ));
    }
    for i in 1..rho.len() {
        if rho[i] <= rho[i - 1] {
            return Err(FrcSolverError::InvalidInput(
                "rho_grid must be strictly increasing",
            ));
        }
    }
    if rho[rho.len() - 1] < r_s {
        return Err(FrcSolverError::InvalidInput(
            "rho_grid must include the separatrix radius R_s",
        ));
    }
    Ok(())
}

fn cylindrical_flux_from_bz(rho: &Array1<f64>, b_z: &Array1<f64>) -> Array1<f64> {
    let integrand = Array1::from_iter(rho.iter().zip(b_z.iter()).map(|(r, b)| r * b));
    let mut psi = Array1::zeros(rho.len());
    for i in 1..rho.len() {
        let dr = rho[i] - rho[i - 1];
        psi[i] = psi[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * dr;
    }
    psi
}

fn radial_force_balance_residual(
    rho: &Array1<f64>,
    b_z: &Array1<f64>,
    p: &Array1<f64>,
) -> Array1<f64> {
    let dp_dr = gradient_edge_order2(rho, p);
    let dbz_dr = gradient_edge_order2(rho, b_z);
    Array1::from_iter(
        dp_dr
            .iter()
            .zip(b_z.iter().zip(dbz_dr.iter()))
            .map(|(dp, (b, db))| dp - (-(b / MU_0) * db)),
    )
}

fn gradient_edge_order2(x: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let n = x.len();
    let mut out = Array1::zeros(n);
    let h0 = x[1] - x[0];
    let h1 = x[2] - x[1];
    out[0] = -((2.0 * h0 + h1) / (h0 * (h0 + h1))) * y[0] + ((h0 + h1) / (h0 * h1)) * y[1]
        - (h0 / (h1 * (h0 + h1))) * y[2];
    for i in 1..(n - 1) {
        let hs = x[i] - x[i - 1];
        let hd = x[i + 1] - x[i];
        let a = -hd / (hs * (hs + hd));
        let b = (hd - hs) / (hs * hd);
        let c = hs / (hd * (hs + hd));
        out[i] = a * y[i - 1] + b * y[i] + c * y[i + 1];
    }
    let hm1 = x[n - 1] - x[n - 2];
    let hm2 = x[n - 2] - x[n - 3];
    out[n - 1] = (hm1 / (hm2 * (hm1 + hm2))) * y[n - 3] - ((hm1 + hm2) / (hm1 * hm2)) * y[n - 2]
        + ((2.0 * hm1 + hm2) / (hm1 * (hm1 + hm2))) * y[n - 1];
    out
}

fn zero_crossing_radius(rho: &Array1<f64>, values: &Array1<f64>) -> f64 {
    for i in 0..(values.len() - 1) {
        if values[i].is_sign_negative() != values[i + 1].is_sign_negative() {
            let y0 = values[i];
            let y1 = values[i + 1];
            if y1 == y0 {
                return rho[i];
            }
            let weight = -y0 / (y1 - y0);
            return rho[i] + weight * (rho[i + 1] - rho[i]);
        }
    }
    let idx = values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    rho[idx]
}

fn nearest_index(rho: &Array1<f64>, target: f64) -> usize {
    rho.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - target)
                .abs()
                .partial_cmp(&(**b - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn interpolate(x: &Array1<f64>, y: &Array1<f64>, target: f64) -> f64 {
    if target <= x[0] {
        return y[0];
    }
    for i in 0..(x.len() - 1) {
        if target <= x[i + 1] {
            let span = x[i + 1] - x[i];
            let weight = if span == 0.0 {
                0.0
            } else {
                (target - x[i]) / span
            };
            return y[i] + weight * (y[i + 1] - y[i]);
        }
    }
    y[y.len() - 1]
}

fn trapezoid(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let mut total = 0.0;
    for i in 1..x.len() {
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    }
    total
}

fn clip_to_separatrix(
    rho: &Array1<f64>,
    values: &Array1<f64>,
    r_s: f64,
) -> Result<(Array1<f64>, Array1<f64>), FrcSolverError> {
    if rho.is_empty() || values.is_empty() || rho.len() != values.len() {
        return Err(FrcSolverError::InvalidInput(
            "rho and value profiles must have matching non-empty lengths",
        ));
    }
    let mut r_clip = Vec::new();
    let mut value_clip = Vec::new();
    for (r, value) in rho.iter().zip(values.iter()) {
        if *r <= r_s {
            r_clip.push(*r);
            value_clip.push(*value);
        } else {
            break;
        }
    }
    if r_clip.is_empty() {
        return Err(FrcSolverError::InvalidInput(
            "rho_grid must contain points below R_s",
        ));
    }
    if *r_clip.last().unwrap_or(&0.0) < r_s {
        r_clip.push(r_s);
        value_clip.push(interpolate(rho, values, r_s));
    }
    Ok((Array1::from_vec(r_clip), Array1::from_vec(value_clip)))
}

fn max_abs(values: &Array1<f64>) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(delta: Option<f64>, theta_dot: f64) -> RigidRotorFrcInputs {
        RigidRotorFrcInputs {
            n0: 2.0e20,
            t_i_ev: 10_000.0,
            t_e_ev: 5_000.0,
            theta_dot,
            r_s: 0.20,
            b_ext: 5.0,
            delta,
        }
    }

    fn linspace(start: f64, end: f64, n: usize) -> Array1<f64> {
        let step = (end - start) / (n as f64 - 1.0);
        Array1::from_iter((0..n).map(|idx| start + idx as f64 * step))
    }

    #[test]
    fn no_rotation_field_matches_steinhauer_formula() {
        let inputs = inputs(Some(0.018), 0.0);
        for n in [32_usize, 64, 128, 256] {
            let rho = linspace(0.0, 0.35, n);
            let state = solve_frc_equilibrium(&inputs, &rho, 1.0e-10).expect("valid state");
            for (r, b_z) in rho.iter().zip(state.b_z.iter()) {
                let expected = -inputs.b_ext
                    * ((r * r - inputs.r_s * inputs.r_s) / (2.0 * inputs.r_s * 0.018)).tanh();
                assert!((b_z - expected).abs() <= 1.0e-14);
            }
            assert!(state.converged);
            assert!(state.residual <= 1.0e-14);
        }
    }

    #[test]
    fn validation_diagnostics_are_finite() {
        let inputs = inputs(Some(0.02), 0.0);
        let rho = linspace(0.0, 0.4, 401);
        let state = solve_frc_equilibrium(&inputs, &rho, 1.0e-10).expect("valid state");
        assert!((state.r_null - inputs.r_s).abs() < 2.5e-4);
        assert!(state.energy_j > 0.0);
        assert!(state.pressure_balance_ratio > 0.0);
        assert!(state.force_balance_residual_linf.is_finite());
        assert!(state.force_balance_residual_l2.is_finite());
        assert_eq!(state.force_balance_residual.len(), rho.len());
    }

    #[test]
    fn default_delta_uses_deuterium_gyroradius() {
        let inputs = inputs(None, 0.0);
        let rho = linspace(0.0, 0.4, 129);
        let state = solve_frc_equilibrium(&inputs, &rho, 1.0e-10).expect("valid state");
        let expected =
            ion_gyroradius_m(inputs.t_i_ev, inputs.b_ext, DEUTERIUM_MASS_AMU).expect("delta");
        assert!((state.delta - expected).abs() <= expected * 1.0e-15);
        assert!(state.s_parameter > 0.0);
        assert_ne!(state.s_parameter, inputs.r_s / (2.0 * expected));
    }

    #[test]
    fn s_parameter_matches_profile_integral() {
        let inputs = inputs(Some(0.02), 0.0);
        let rho = linspace(0.0, 0.4, 1025);
        let state = solve_frc_equilibrium(&inputs, &rho, 1.0e-10).expect("valid state");
        let expected = s_parameter_from_profile(
            &rho,
            &state.b_z,
            inputs.r_s,
            inputs.t_i_ev,
            DEUTERIUM_MASS_AMU,
        )
        .expect("s parameter");
        assert!((state.s_parameter - expected).abs() <= expected * 1.0e-14);
    }

    #[test]
    fn rejects_invalid_inputs_and_rotating_bvp() {
        let rho = linspace(0.0, 0.4, 32);
        assert!(solve_frc_equilibrium(&inputs(Some(0.02), 1.0), &rho, 1.0e-10).is_err());
        let bad_grid = Array1::from_vec(vec![0.0, 0.1, 0.1, 0.3]);
        assert!(solve_frc_equilibrium(&inputs(Some(0.02), 0.0), &bad_grid, 1.0e-10).is_err());
        let off_axis_grid = linspace(0.01, 0.4, 32);
        assert!(solve_frc_equilibrium(&inputs(Some(0.02), 0.0), &off_axis_grid, 1.0e-10).is_err());
    }
}
