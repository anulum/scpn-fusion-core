// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Non-Adiabatic Current Diffusion Carrier
//! Non-adiabatic flux-evolution carrier for pulsed MIF/FRC workflows.

/// Input histories for the local non-adiabatic flux equation.
#[derive(Debug, Clone)]
pub struct NonadiabaticFluxInput {
    pub rho: Vec<f64>,
    pub psi0: Vec<f64>,
    pub tau_psi_s: Vec<Vec<f64>>,
    pub r_null_m: Vec<f64>,
    pub e_theta_v_m: Vec<Vec<f64>>,
    pub eta_ohm_m: Vec<Vec<f64>>,
    pub j_theta_a_m2: Vec<Vec<f64>>,
    pub dt_s: f64,
}

/// Output histories for the local non-adiabatic flux equation.
#[derive(Debug, Clone)]
pub struct FluxEvolutionTrajectory {
    pub time_s: Vec<f64>,
    pub rho: Vec<f64>,
    pub psi: Vec<Vec<f64>>,
    pub hall_drive: Vec<Vec<f64>>,
    pub resistive_loss: Vec<Vec<f64>>,
    pub source: Vec<Vec<f64>>,
    pub damping_rate: Vec<Vec<f64>>,
    pub dt_s: f64,
}

/// Solve `dpsi/dt = -psi/tau_psi + R_null E_theta - eta J_theta`.
pub fn solve_flux_evolution_nonadiabatic(
    input: &NonadiabaticFluxInput,
) -> Result<FluxEvolutionTrajectory, String> {
    validate_input(input)?;
    let n_times = input.tau_psi_s.len();
    let n_steps = n_times - 1;
    let n_rho = input.rho.len();

    let mut time_s = vec![0.0_f64; n_times];
    for (index, value) in time_s.iter_mut().enumerate() {
        *value = index as f64 * input.dt_s;
    }

    let mut psi = vec![vec![0.0_f64; n_rho]; n_times];
    let mut hall_drive = vec![vec![0.0_f64; n_rho]; n_times];
    let mut resistive_loss = vec![vec![0.0_f64; n_rho]; n_times];
    let mut source = vec![vec![0.0_f64; n_rho]; n_times];
    let mut damping_rate = vec![vec![0.0_f64; n_rho]; n_times];
    psi[0].clone_from(&input.psi0);

    for time_index in 0..n_times {
        for rho_index in 0..n_rho {
            let tau = input.tau_psi_s[time_index][rho_index];
            damping_rate[time_index][rho_index] = if tau.is_infinite() { 0.0 } else { 1.0 / tau };
            hall_drive[time_index][rho_index] =
                input.r_null_m[time_index] * input.e_theta_v_m[time_index][rho_index];
            resistive_loss[time_index][rho_index] =
                input.eta_ohm_m[time_index][rho_index] * input.j_theta_a_m2[time_index][rho_index];
            source[time_index][rho_index] =
                hall_drive[time_index][rho_index] - resistive_loss[time_index][rho_index];
        }
    }

    for step_index in 0..n_steps {
        for rho_index in 0..n_rho {
            let gamma = 0.5
                * (damping_rate[step_index][rho_index] + damping_rate[step_index + 1][rho_index]);
            let source_midpoint =
                0.5 * (source[step_index][rho_index] + source[step_index + 1][rho_index]);
            let decay = (-gamma * input.dt_s).exp();
            let driven_increment = if gamma > 0.0 {
                source_midpoint * (1.0 - decay) / gamma
            } else {
                input.dt_s * source_midpoint
            };
            psi[step_index + 1][rho_index] = psi[step_index][rho_index] * decay + driven_increment;
        }
    }

    Ok(FluxEvolutionTrajectory {
        time_s,
        rho: input.rho.clone(),
        psi,
        hall_drive,
        resistive_loss,
        source,
        damping_rate,
        dt_s: input.dt_s,
    })
}

fn validate_input(input: &NonadiabaticFluxInput) -> Result<(), String> {
    if input.rho.len() < 2 {
        return Err("rho must contain at least two points".to_string());
    }
    for pair in input.rho.windows(2) {
        if !pair[0].is_finite() || !pair[1].is_finite() || pair[1] <= pair[0] {
            return Err("rho must be finite and strictly increasing".to_string());
        }
    }
    if input.psi0.len() != input.rho.len() || input.psi0.iter().any(|value| !value.is_finite()) {
        return Err("psi0 must match rho and contain finite values".to_string());
    }
    if !input.dt_s.is_finite() || input.dt_s <= 0.0 {
        return Err("dt_s must be positive".to_string());
    }
    let n_times = input.tau_psi_s.len();
    if n_times < 2 {
        return Err("time histories must contain at least two samples".to_string());
    }
    if input.r_null_m.len() != n_times
        || input.e_theta_v_m.len() != n_times
        || input.eta_ohm_m.len() != n_times
        || input.j_theta_a_m2.len() != n_times
    {
        return Err("all time histories must have the same length".to_string());
    }
    for time_index in 0..n_times {
        if !input.r_null_m[time_index].is_finite() || input.r_null_m[time_index] < 0.0 {
            return Err("r_null_m must be finite and non-negative".to_string());
        }
        validate_row(
            &input.tau_psi_s[time_index],
            input.rho.len(),
            "tau_psi_s",
            true,
        )?;
        validate_row(
            &input.e_theta_v_m[time_index],
            input.rho.len(),
            "e_theta_v_m",
            false,
        )?;
        validate_row(
            &input.eta_ohm_m[time_index],
            input.rho.len(),
            "eta_ohm_m",
            false,
        )?;
        validate_row(
            &input.j_theta_a_m2[time_index],
            input.rho.len(),
            "j_theta_a_m2",
            false,
        )?;
        for value in &input.tau_psi_s[time_index] {
            if value.is_sign_negative() || (*value <= 0.0 && !value.is_infinite()) {
                return Err("tau_psi_s must be positive or infinite".to_string());
            }
        }
        for value in &input.eta_ohm_m[time_index] {
            if *value < 0.0 {
                return Err("eta_ohm_m must be non-negative".to_string());
            }
        }
    }
    Ok(())
}

fn validate_row(
    row: &[f64],
    expected_len: usize,
    name: &str,
    allow_infinite: bool,
) -> Result<(), String> {
    if row.len() != expected_len {
        return Err(format!("{name} row must match rho length"));
    }
    if row
        .iter()
        .any(|value| !(value.is_finite() || allow_infinite && value.is_infinite()))
    {
        return Err(format!("{name} must contain valid numeric values"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{solve_flux_evolution_nonadiabatic, NonadiabaticFluxInput};

    fn base_input(n_rho: usize, n_steps: usize) -> NonadiabaticFluxInput {
        let rho = (0..n_rho)
            .map(|index| index as f64 / (n_rho as f64 - 1.0))
            .collect::<Vec<_>>();
        let n_times = n_steps + 1;
        NonadiabaticFluxInput {
            rho,
            psi0: vec![0.2; n_rho],
            tau_psi_s: vec![vec![f64::INFINITY; n_rho]; n_times],
            r_null_m: vec![0.0; n_times],
            e_theta_v_m: vec![vec![0.0; n_rho]; n_times],
            eta_ohm_m: vec![vec![0.0; n_rho]; n_times],
            j_theta_a_m2: vec![vec![0.0; n_rho]; n_times],
            dt_s: 1.0e-8,
        }
    }

    #[test]
    fn zero_drive_preserves_flux() {
        let input = base_input(16, 5);
        let trajectory = solve_flux_evolution_nonadiabatic(&input).unwrap();
        assert_eq!(trajectory.psi[0], trajectory.psi[5]);
        assert!(trajectory
            .source
            .iter()
            .flatten()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn hall_drive_integrates_linearly_without_damping() {
        let mut input = base_input(12, 4);
        for row in &mut input.e_theta_v_m {
            for value in row {
                *value = 15.0;
            }
        }
        for value in &mut input.r_null_m {
            *value = 0.2;
        }
        let trajectory = solve_flux_evolution_nonadiabatic(&input).unwrap();
        let expected = 0.2 + 4.0 * input.dt_s * 0.2 * 15.0;
        assert!((trajectory.psi[4][0] - expected).abs() < 1.0e-15);
    }

    #[test]
    fn damping_matches_constant_tau_exponential() {
        let mut input = base_input(10, 8);
        for row in &mut input.tau_psi_s {
            for value in row {
                *value = 4.0e-7;
            }
        }
        let trajectory = solve_flux_evolution_nonadiabatic(&input).unwrap();
        let expected = 0.2 * (-(8.0 * input.dt_s) / 4.0e-7).exp();
        assert!((trajectory.psi[8][0] - expected).abs() < 1.0e-15);
    }
}
