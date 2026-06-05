// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Pulsed Hall-MHD
//! Axisymmetric Ono Eq. 8 Hall-MHD flux carrier.

pub const MU_0: f64 = 4.0e-7 * std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq)]
pub struct HallMhdPulsedConfig {
    pub rho_m: Vec<f64>,
    pub tau_psi_s: f64,
    pub r_null_m: f64,
    pub electron_temperature_ev: f64,
    pub z_eff: f64,
    pub ln_lambda: f64,
    pub hall_scale: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HallMhdPulsedState {
    pub t_s: f64,
    pub psi: Vec<f64>,
    pub b_z: Vec<f64>,
    pub e_theta: Vec<f64>,
    pub j_theta: Vec<f64>,
    pub energy_proxy_j_m: f64,
    pub hall_drive_l2: f64,
    pub resistive_sink_l2: f64,
    pub damping_sink_l2: f64,
    pub source_residual_linf: f64,
    pub external_parity_status: &'static str,
}

fn require_positive(name: &str, value: f64) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    if value <= 0.0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

fn validate_grid(rho: &[f64]) -> Result<(), String> {
    if rho.len() < 3 {
        return Err("rho_m must contain at least three points".to_string());
    }
    if rho[0] < 0.0 || !rho[0].is_finite() {
        return Err("rho_m must start at a non-negative finite radius".to_string());
    }
    for pair in rho.windows(2) {
        if !pair[1].is_finite() || pair[1] <= pair[0] {
            return Err("rho_m must be strictly increasing".to_string());
        }
    }
    Ok(())
}

fn validate_profile(name: &str, values: &[f64], expected: usize) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!("{name} must have length {expected}"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must be finite"));
    }
    Ok(())
}

pub fn spitzer_resistivity_ohm_m(
    temperature_ev: f64,
    z_eff: f64,
    ln_lambda: f64,
) -> Result<f64, String> {
    let temperature = require_positive("temperature_eV", temperature_ev)?;
    let z = require_positive("Z_eff", z_eff)?;
    let log = require_positive("ln_lambda", ln_lambda)?;
    Ok(1.65e-9 * z * log / temperature.powf(1.5))
}

pub fn faraday_e_theta_from_b_ramp(rho_m: &[f64], d_b_ext_dt_t_s: f64) -> Result<Vec<f64>, String> {
    validate_grid(rho_m)?;
    if !d_b_ext_dt_t_s.is_finite() {
        return Err("dB_ext_dt must be finite".to_string());
    }
    Ok(rho_m.iter().map(|r| -0.5 * r * d_b_ext_dt_t_s).collect())
}

pub fn axial_field_from_flux(rho_m: &[f64], psi: &[f64]) -> Result<Vec<f64>, String> {
    validate_grid(rho_m)?;
    validate_profile("psi", psi, rho_m.len())?;
    let gradient = gradient_edge_order2(rho_m, psi);
    let mut field = vec![0.0; psi.len()];
    for index in 1..psi.len() {
        field[index] = gradient[index] / rho_m[index];
    }
    field[0] = field[1];
    Ok(field)
}

pub fn initial_hall_mhd_pulsed_state(
    cfg: &HallMhdPulsedConfig,
    psi: Vec<f64>,
    e_theta: Vec<f64>,
    j_theta: Vec<f64>,
) -> Result<HallMhdPulsedState, String> {
    validate_config(cfg)?;
    validate_profile("psi", &psi, cfg.rho_m.len())?;
    validate_profile("E_theta", &e_theta, cfg.rho_m.len())?;
    validate_profile("J_theta", &j_theta, cfg.rho_m.len())?;
    let b_z = axial_field_from_flux(&cfg.rho_m, &psi)?;
    Ok(HallMhdPulsedState {
        t_s: 0.0,
        energy_proxy_j_m: magnetic_energy_proxy(&cfg.rho_m, &psi),
        hall_drive_l2: l2(&cfg.rho_m, &scale(&e_theta, cfg.hall_scale * cfg.r_null_m)),
        resistive_sink_l2: l2(&cfg.rho_m, &scale(&j_theta, eta(cfg)?)),
        damping_sink_l2: l2(&cfg.rho_m, &scale(&psi, 1.0 / cfg.tau_psi_s)),
        source_residual_linf: 0.0,
        external_parity_status: "blocked_missing_public_same_case_reference",
        psi,
        b_z,
        e_theta,
        j_theta,
    })
}

pub fn step_hall_mhd_pulsed(
    state: &HallMhdPulsedState,
    cfg: &HallMhdPulsedConfig,
    e_theta: Vec<f64>,
    j_theta: Vec<f64>,
    dt_s: f64,
) -> Result<HallMhdPulsedState, String> {
    validate_config(cfg)?;
    validate_profile("state.psi", &state.psi, cfg.rho_m.len())?;
    validate_profile("E_theta", &e_theta, cfg.rho_m.len())?;
    validate_profile("J_theta", &j_theta, cfg.rho_m.len())?;
    let dt = require_positive("dt_s", dt_s)?;
    let eta_value = eta(cfg)?;
    let source = ono_source(cfg, &e_theta, &j_theta, eta_value);
    let denominator = 1.0 + dt / cfg.tau_psi_s;
    let psi = state
        .psi
        .iter()
        .zip(source.iter())
        .map(|(old, src)| (old + dt * src) / denominator)
        .collect::<Vec<_>>();
    let residual = psi
        .iter()
        .zip(state.psi.iter())
        .zip(source.iter())
        .map(|((new_value, old_value), src)| {
            (new_value - old_value) / dt - src + new_value / cfg.tau_psi_s
        })
        .collect::<Vec<_>>();
    let scale_value = source
        .iter()
        .chain(psi.iter())
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let b_z = axial_field_from_flux(&cfg.rho_m, &psi)?;
    Ok(HallMhdPulsedState {
        t_s: state.t_s + dt,
        energy_proxy_j_m: magnetic_energy_proxy(&cfg.rho_m, &psi),
        hall_drive_l2: l2(&cfg.rho_m, &scale(&e_theta, cfg.hall_scale * cfg.r_null_m)),
        resistive_sink_l2: l2(&cfg.rho_m, &scale(&j_theta, eta_value)),
        damping_sink_l2: l2(&cfg.rho_m, &scale(&psi, 1.0 / cfg.tau_psi_s)),
        source_residual_linf: residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            / scale_value,
        external_parity_status: "blocked_missing_public_same_case_reference",
        psi,
        b_z,
        e_theta,
        j_theta,
    })
}

fn validate_config(cfg: &HallMhdPulsedConfig) -> Result<(), String> {
    validate_grid(&cfg.rho_m)?;
    require_positive("tau_psi_s", cfg.tau_psi_s)?;
    require_positive("r_null_m", cfg.r_null_m)?;
    require_positive("electron_temperature_eV", cfg.electron_temperature_ev)?;
    require_positive("Z_eff", cfg.z_eff)?;
    require_positive("ln_lambda", cfg.ln_lambda)?;
    if !cfg.hall_scale.is_finite() {
        return Err("hall_scale must be finite".to_string());
    }
    Ok(())
}

fn eta(cfg: &HallMhdPulsedConfig) -> Result<f64, String> {
    spitzer_resistivity_ohm_m(cfg.electron_temperature_ev, cfg.z_eff, cfg.ln_lambda)
}

fn ono_source(
    cfg: &HallMhdPulsedConfig,
    e_theta: &[f64],
    j_theta: &[f64],
    eta_value: f64,
) -> Vec<f64> {
    e_theta
        .iter()
        .zip(j_theta.iter())
        .map(|(e, j)| cfg.hall_scale * cfg.r_null_m * e - eta_value * j)
        .collect()
}

fn gradient_edge_order2(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut gradient = vec![0.0; n];
    gradient[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (x[2] - x[0]);
    gradient[n - 1] = (3.0 * y[n - 1] - 4.0 * y[n - 2] + y[n - 3]) / (x[n - 1] - x[n - 3]);
    for index in 1..(n - 1) {
        gradient[index] = (y[index + 1] - y[index - 1]) / (x[index + 1] - x[index - 1]);
    }
    gradient
}

fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xw, yw)| 0.5 * (yw[0] + yw[1]) * (xw[1] - xw[0]))
        .sum()
}

fn magnetic_energy_proxy(rho: &[f64], psi: &[f64]) -> f64 {
    let integrand = rho
        .iter()
        .zip(psi.iter())
        .map(|(r, p)| 0.5 * p * p * 2.0 * std::f64::consts::PI * r / MU_0)
        .collect::<Vec<_>>();
    trapezoid(rho, &integrand)
}

fn l2(rho: &[f64], values: &[f64]) -> f64 {
    let integrand = rho
        .iter()
        .zip(values.iter())
        .map(|(r, value)| value * value * 2.0 * std::f64::consts::PI * r)
        .collect::<Vec<_>>();
    trapezoid(rho, &integrand).max(0.0).sqrt()
}

fn scale(values: &[f64], factor: f64) -> Vec<f64> {
    values.iter().map(|value| value * factor).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HallMhdPulsedConfig {
        HallMhdPulsedConfig {
            rho_m: (0..129).map(|i| i as f64 * 0.4 / 128.0).collect(),
            tau_psi_s: 2.0e-6,
            r_null_m: 0.2,
            electron_temperature_ev: 5_000.0,
            z_eff: 1.0,
            ln_lambda: 17.0,
            hall_scale: 1.0,
        }
    }

    #[test]
    fn faraday_drive_matches_circular_loop_formula() {
        let field = faraday_e_theta_from_b_ramp(&[0.0, 0.1, 0.2], 2.0e4).unwrap();
        assert!((field[1] + 1000.0).abs() < 1.0e-10);
        assert!((field[2] + 2000.0).abs() < 1.0e-10);
    }

    #[test]
    fn spitzer_temperature_scaling() {
        let eta_100 = spitzer_resistivity_ohm_m(100.0, 1.0, 17.0).unwrap();
        let eta_400 = spitzer_resistivity_ohm_m(400.0, 1.0, 17.0).unwrap();
        assert!((eta_400 - eta_100 / 8.0).abs() / eta_400 < 1.0e-14);
    }

    #[test]
    fn implicit_damping_matches_closed_form_decay() {
        let mut case = cfg();
        case.hall_scale = 0.0;
        let psi = case.rho_m.iter().map(|r| 1.0 + r).collect::<Vec<_>>();
        let zeros = vec![0.0; case.rho_m.len()];
        let state = initial_hall_mhd_pulsed_state(&case, psi.clone(), zeros.clone(), zeros.clone())
            .unwrap();
        let next = step_hall_mhd_pulsed(&state, &case, zeros.clone(), zeros, 1.0e-7).unwrap();
        let expected_factor = 1.0 / (1.0 + 1.0e-7 / case.tau_psi_s);
        assert!((next.psi[7] - psi[7] * expected_factor).abs() < 1.0e-12);
    }

    #[test]
    fn hall_drive_increases_flux() {
        let mut case = cfg();
        case.tau_psi_s = 1.0e9;
        let psi = case.rho_m.iter().map(|r| 1.0 + r).collect::<Vec<_>>();
        let zeros = vec![0.0; case.rho_m.len()];
        let drive = vec![2.5; case.rho_m.len()];
        let state =
            initial_hall_mhd_pulsed_state(&case, psi, drive.clone(), zeros.clone()).unwrap();
        let next = step_hall_mhd_pulsed(&state, &case, drive, zeros, 1.0e-8).unwrap();
        assert!(next
            .psi
            .iter()
            .zip(state.psi.iter())
            .all(|(new_value, old_value)| new_value > old_value));
        assert!(next.hall_drive_l2 > 0.0);
    }

    #[test]
    fn small_hall_limit_recovers_resistive_sink() {
        let mut case = cfg();
        case.hall_scale = 0.0;
        case.tau_psi_s = 1.0e9;
        let psi = vec![1.0; case.rho_m.len()];
        let drive = vec![10.0; case.rho_m.len()];
        let current = vec![2.0e5; case.rho_m.len()];
        let state =
            initial_hall_mhd_pulsed_state(&case, psi, drive.clone(), current.clone()).unwrap();
        let next = step_hall_mhd_pulsed(&state, &case, drive, current, 1.0e-8).unwrap();
        let expected =
            (1.0 - 1.0e-8 * eta(&case).unwrap() * 2.0e5) / (1.0 + 1.0e-8 / case.tau_psi_s);
        assert!((next.psi[9] - expected).abs() < 1.0e-14);
        assert_eq!(next.hall_drive_l2, 0.0);
    }
}
