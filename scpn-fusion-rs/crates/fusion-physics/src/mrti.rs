// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — MRTI Growth Spectrum
//! Magneto-Rayleigh-Taylor instability growth and spectrum tracking.

use crate::compression::PulsedCompressionState;

pub const MU_0: f64 = 4.0e-7 * std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq)]
pub struct MrtiSpectrumState {
    pub t_s: f64,
    pub k_modes_m_inv: Vec<f64>,
    pub amplitudes_m: Vec<f64>,
    pub log_amplitudes: Vec<f64>,
    pub growth_rates_s_inv: Vec<f64>,
    pub fastest_growing_k_m_inv: f64,
    pub max_amplitude_m: f64,
    pub max_log_amplitude: f64,
    pub saturation_warning: bool,
    pub time_of_breach_s: Option<f64>,
    pub amplitude_overflow_limited: bool,
}

#[derive(Debug, Clone)]
pub struct MrtiSpectrumTracker {
    k_modes_m_inv: Vec<f64>,
    amplitudes_m: Vec<f64>,
    log_amplitudes: Vec<f64>,
    growth_rates_s_inv: Vec<f64>,
    rho_kg_m3: f64,
    saturation_threshold_m: f64,
    t_s: f64,
    time_of_breach_s: Option<f64>,
    amplitude_overflow_limited: bool,
}

fn require_finite(name: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn require_positive(name: &str, value: f64) -> Result<f64, String> {
    let checked = require_finite(name, value)?;
    if checked > 0.0 {
        Ok(checked)
    } else {
        Err(format!("{name} must be positive"))
    }
}

pub fn mrti_growth_rate(
    k_m_inv: f64,
    a_eff_m_s2: f64,
    b_perp_t: f64,
    rho_kg_m3: f64,
) -> Result<f64, String> {
    let k = require_finite("k_m_inv", k_m_inv)?;
    if k < 0.0 {
        return Err("k_m_inv must be non-negative".to_string());
    }
    let acceleration = require_finite("a_eff_m_s2", a_eff_m_s2)?;
    let b_perp = require_finite("b_perp_t", b_perp_t)?;
    let density = require_positive("rho_kg_m3", rho_kg_m3)?;
    let radicand = k * acceleration - (k * k * b_perp * b_perp) / (MU_0 * density);
    Ok(radicand.max(0.0).sqrt())
}

pub fn mrti_growth_rates(
    k_modes_m_inv: &[f64],
    a_eff_m_s2: f64,
    b_perp_t: f64,
    rho_kg_m3: f64,
) -> Result<Vec<f64>, String> {
    k_modes_m_inv
        .iter()
        .map(|k| mrti_growth_rate(*k, a_eff_m_s2, b_perp_t, rho_kg_m3))
        .collect()
}

pub fn effective_acceleration_from_radius_rate(
    time_s: &[f64],
    d_radius_dt_m_s: &[f64],
    smoothing_window: usize,
) -> Result<Vec<f64>, String> {
    if time_s.len() != d_radius_dt_m_s.len() {
        return Err("time_s and d_radius_dt_m_s must have identical length".to_string());
    }
    if time_s.len() < 2 {
        return Err("at least two samples are required".to_string());
    }
    if smoothing_window == 0 || smoothing_window.is_multiple_of(2) {
        return Err("smoothing_window must be a positive odd integer".to_string());
    }
    if smoothing_window > time_s.len() {
        return Err("smoothing_window cannot exceed the number of samples".to_string());
    }
    for value in time_s.iter().chain(d_radius_dt_m_s.iter()) {
        if !value.is_finite() {
            return Err("trajectory samples must be finite".to_string());
        }
    }
    for pair in time_s.windows(2) {
        if pair[1] <= pair[0] {
            return Err("time_s must be strictly increasing".to_string());
        }
    }

    let n = time_s.len();
    let mut acceleration = vec![0.0; n];
    if n == 2 {
        let slope = (d_radius_dt_m_s[1] - d_radius_dt_m_s[0]) / (time_s[1] - time_s[0]);
        acceleration[0] = slope;
        acceleration[1] = slope;
    } else {
        acceleration[0] = (d_radius_dt_m_s[1] - d_radius_dt_m_s[0]) / (time_s[1] - time_s[0]);
        acceleration[n - 1] =
            (d_radius_dt_m_s[n - 1] - d_radius_dt_m_s[n - 2]) / (time_s[n - 1] - time_s[n - 2]);
        for index in 1..(n - 1) {
            acceleration[index] = (d_radius_dt_m_s[index + 1] - d_radius_dt_m_s[index - 1])
                / (time_s[index + 1] - time_s[index - 1]);
        }
    }

    if smoothing_window == 1 {
        return Ok(acceleration);
    }

    let half = smoothing_window / 2;
    let mut smoothed = Vec::with_capacity(n);
    for index in 0..n {
        let mut sum = 0.0;
        for window_index in 0..smoothing_window {
            let raw = index + window_index;
            let source = raw.saturating_sub(half).min(n - 1);
            sum += acceleration[source];
        }
        smoothed.push(sum / smoothing_window as f64);
    }
    Ok(smoothed)
}

pub fn effective_acceleration_from_pulsed_compression(
    states: &[PulsedCompressionState],
    smoothing_window: usize,
    radial_projection_sign: f64,
) -> Result<Vec<f64>, String> {
    if states.len() < 2 {
        return Err("at least two pulsed-compression states are required".to_string());
    }
    let projection = require_finite("radial_projection_sign", radial_projection_sign)?;
    if projection == 0.0 {
        return Err("radial_projection_sign must be non-zero".to_string());
    }
    for state in states {
        require_finite("t_s", state.t_s)?;
        require_positive("r_s_m", state.r_s_m)?;
        require_finite("d_r_s_dt_m_s", state.d_r_s_dt_m_s)?;
        require_finite("b_ext_t", state.b_ext_t)?;
    }
    let time_s = states.iter().map(|state| state.t_s).collect::<Vec<_>>();
    let speed_m_s = states
        .iter()
        .map(|state| state.d_r_s_dt_m_s)
        .collect::<Vec<_>>();
    let signed = effective_acceleration_from_radius_rate(&time_s, &speed_m_s, smoothing_window)?;
    Ok(signed.into_iter().map(|value| projection * value).collect())
}

pub fn track_mrti_from_pulsed_compression(
    states: &[PulsedCompressionState],
    tracker: &mut MrtiSpectrumTracker,
    smoothing_window: usize,
    radial_projection_sign: f64,
    b_perp_scale: f64,
) -> Result<Vec<MrtiSpectrumState>, String> {
    if states.len() < 2 {
        return Err("at least two pulsed-compression states are required".to_string());
    }
    let field_scale = require_finite("b_perp_scale", b_perp_scale)?;
    if field_scale < 0.0 {
        return Err("b_perp_scale must be non-negative".to_string());
    }
    let accelerations = effective_acceleration_from_pulsed_compression(
        states,
        smoothing_window,
        radial_projection_sign,
    )?;
    let mut snapshots = Vec::with_capacity(states.len() - 1);
    for index in 1..states.len() {
        let dt_s = require_positive("trajectory dt_s", states[index].t_s - states[index - 1].t_s)?;
        snapshots.push(tracker.step(
            dt_s,
            accelerations[index],
            states[index].b_ext_t * field_scale,
        )?);
    }
    Ok(snapshots)
}

impl MrtiSpectrumTracker {
    pub fn new(
        k_max_m_inv: f64,
        n_modes: usize,
        initial_perturbation_m: f64,
        rho_kg_m3: f64,
        saturation_threshold_m: f64,
    ) -> Result<Self, String> {
        let k_max = require_positive("k_max_m_inv", k_max_m_inv)?;
        if n_modes < 2 {
            return Err("n_modes must be at least 2".to_string());
        }
        let k_modes_m_inv = (1..=n_modes)
            .map(|mode| k_max * mode as f64 / n_modes as f64)
            .collect::<Vec<_>>();
        Self::from_modes(
            k_modes_m_inv,
            initial_perturbation_m,
            rho_kg_m3,
            saturation_threshold_m,
        )
    }

    pub fn from_modes(
        k_modes_m_inv: Vec<f64>,
        initial_perturbation_m: f64,
        rho_kg_m3: f64,
        saturation_threshold_m: f64,
    ) -> Result<Self, String> {
        if k_modes_m_inv.len() < 2 {
            return Err("at least two MRTI modes are required".to_string());
        }
        for mode in &k_modes_m_inv {
            if !mode.is_finite() {
                return Err("k_modes_m_inv must be finite".to_string());
            }
            if *mode < 0.0 {
                return Err("k_modes_m_inv must be non-negative".to_string());
            }
        }
        for pair in k_modes_m_inv.windows(2) {
            if pair[1] <= pair[0] {
                return Err("k_modes_m_inv must be strictly increasing".to_string());
            }
        }
        let initial = require_positive("initial_perturbation_m", initial_perturbation_m)?;
        let density = require_positive("rho_kg_m3", rho_kg_m3)?;
        let threshold = require_positive("saturation_threshold_m", saturation_threshold_m)?;
        Ok(Self {
            amplitudes_m: vec![initial; k_modes_m_inv.len()],
            log_amplitudes: vec![initial.ln(); k_modes_m_inv.len()],
            growth_rates_s_inv: vec![0.0; k_modes_m_inv.len()],
            k_modes_m_inv,
            rho_kg_m3: density,
            saturation_threshold_m: threshold,
            t_s: 0.0,
            time_of_breach_s: None,
            amplitude_overflow_limited: false,
        })
    }

    pub fn state(&self) -> MrtiSpectrumState {
        let fastest_index = self
            .growth_rates_s_inv
            .iter()
            .enumerate()
            .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let max_amplitude_m = self
            .amplitudes_m
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_log_amplitude = self
            .log_amplitudes
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        MrtiSpectrumState {
            t_s: self.t_s,
            k_modes_m_inv: self.k_modes_m_inv.clone(),
            amplitudes_m: self.amplitudes_m.clone(),
            log_amplitudes: self.log_amplitudes.clone(),
            growth_rates_s_inv: self.growth_rates_s_inv.clone(),
            fastest_growing_k_m_inv: self.k_modes_m_inv[fastest_index],
            max_amplitude_m,
            max_log_amplitude,
            saturation_warning: max_amplitude_m >= self.saturation_threshold_m,
            time_of_breach_s: self.time_of_breach_s,
            amplitude_overflow_limited: self.amplitude_overflow_limited,
        }
    }

    pub fn step(
        &mut self,
        dt_s: f64,
        a_eff_m_s2: f64,
        b_perp_t: f64,
    ) -> Result<MrtiSpectrumState, String> {
        let dt = require_positive("dt_s", dt_s)?;
        self.growth_rates_s_inv =
            mrti_growth_rates(&self.k_modes_m_inv, a_eff_m_s2, b_perp_t, self.rho_kg_m3)?;
        for ((amplitude, log_amplitude), gamma) in self
            .amplitudes_m
            .iter_mut()
            .zip(self.log_amplitudes.iter_mut())
            .zip(self.growth_rates_s_inv.iter())
        {
            let growth_exponent = (gamma * dt).max(0.0);
            *log_amplitude += growth_exponent;
            if *log_amplitude > f64::MAX.ln() {
                self.amplitude_overflow_limited = true;
                *amplitude = f64::MAX;
            } else {
                *amplitude = (*log_amplitude).exp();
            }
        }
        self.t_s += dt;
        if self.time_of_breach_s.is_none() && self.saturation_threshold_breached(None)? {
            self.time_of_breach_s = Some(self.t_s);
        }
        Ok(self.state())
    }

    pub fn saturation_threshold_breached(&self, threshold_m: Option<f64>) -> Result<bool, String> {
        let threshold = match threshold_m {
            Some(value) => require_positive("threshold_m", value)?,
            None => self.saturation_threshold_m,
        };
        Ok(self
            .amplitudes_m
            .iter()
            .any(|amplitude| *amplitude >= threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        effective_acceleration_from_pulsed_compression, effective_acceleration_from_radius_rate,
        mrti_growth_rate, track_mrti_from_pulsed_compression, MrtiSpectrumTracker, MU_0,
    };
    use crate::compression::{
        coil_field_t, plasma_volume_m3, run_pulsed_compression, CoilGeometry,
        PulsedCompressionConfig, PulsedCompressionState,
    };

    #[test]
    fn growth_rate_matches_hydrodynamic_limit() {
        let gamma = mrti_growth_rate(4.0, 9.0, 0.0, 1.0e-3).expect("valid growth rate");
        assert!((gamma - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn magnetic_tension_stabilizes_short_modes() {
        let density = 1.2e-3;
        let b_perp = 1.0e-3;
        let acceleration = 8.0e6;
        let k = 2.0e7;
        let raw = k * acceleration - (k * k * b_perp * b_perp) / (MU_0 * density);
        let gamma = mrti_growth_rate(k, acceleration, b_perp, density).expect("valid growth rate");
        assert!(raw < 0.0);
        assert_eq!(gamma, 0.0);
    }

    #[test]
    fn spectrum_tracker_matches_exponential_growth() {
        let mut tracker = MrtiSpectrumTracker::from_modes(vec![2.0, 8.0], 2.0e-9, 1.0e-3, 1.0e-3)
            .expect("valid tracker");
        for _ in 0..12 {
            tracker.step(2.5e-7, 4.0e6, 0.0).expect("valid step");
        }
        let state = tracker.state();
        let gamma = (8.0_f64 * 4.0e6_f64).sqrt();
        let expected = 2.0e-9 * (gamma * 2.5e-7 * 12.0).exp();
        assert!((state.amplitudes_m[1] - expected).abs() / expected < 1.0e-12);
        assert!((state.log_amplitudes[1] - expected.ln()).abs() < 1.0e-12);
        assert_eq!(state.fastest_growing_k_m_inv, 8.0);
        assert!(!state.saturation_warning);
        assert!(!state.amplitude_overflow_limited);
    }

    #[test]
    fn spectrum_tracker_keeps_extreme_growth_finite_in_log_space() {
        let mut tracker = MrtiSpectrumTracker::from_modes(vec![1.0, 4.0], 1.0e-9, 1.0e-3, 1.0e-3)
            .expect("valid tracker");
        let state = tracker.step(1.0, 1.0e8, 0.0).expect("valid step");

        assert!(state.amplitudes_m.iter().all(|value| value.is_finite()));
        assert!(state.max_amplitude_m.is_finite());
        assert!(state.log_amplitudes.iter().all(|value| value.is_finite()));
        assert!(state.max_log_amplitude > f64::MAX.ln());
        assert!(state.amplitude_overflow_limited);
        assert!(state.saturation_warning);
    }

    #[test]
    fn spectrum_tracker_records_first_saturation_breach() {
        let mut tracker = MrtiSpectrumTracker::from_modes(vec![10.0, 40.0], 1.0e-8, 1.0e-3, 1.0e-6)
            .expect("valid tracker");
        let state = tracker.step(1.0e-4, 1.0e8, 0.0).expect("valid step");
        assert!(state.saturation_warning);
        assert_eq!(state.time_of_breach_s, Some(state.t_s));
    }

    #[test]
    fn acceleration_helper_recovers_linear_ramp() {
        let time = vec![0.0, 0.25e-6, 0.5e-6, 0.75e-6, 1.0e-6];
        let acceleration = -3.25e11;
        let speed = time
            .iter()
            .map(|t| 5.0e4 + acceleration * t)
            .collect::<Vec<_>>();
        let estimated =
            effective_acceleration_from_radius_rate(&time, &speed, 3).expect("valid acceleration");
        for value in estimated {
            assert!((value - acceleration).abs() < 2.0e-4);
        }
    }

    fn compression_coil() -> CoilGeometry {
        CoilGeometry {
            n_turns: 80,
            l_coil_m: 1.0,
            r_coil_m: 0.35,
            l_inductance_h: 2.0e-6,
            r_resistance_ohm: 0.02,
            bank_voltage_max_v: 20_000.0,
        }
    }

    fn compression_config() -> PulsedCompressionConfig {
        PulsedCompressionConfig {
            coil: compression_coil(),
            plasma_mass_kg: 2.0e-5,
            plasma_length_m: 1.0,
            gamma: 5.0 / 3.0,
            radial_loss_time_s: None,
            tau_psi_s: f64::INFINITY,
            e_theta_v_m: None,
            j_theta_a_m2: None,
            z_eff: 1.0,
            ln_lambda: 17.0,
            min_radius_m: 1.0e-4,
        }
    }

    fn compression_state() -> PulsedCompressionState {
        let density = 2.0e21;
        let t_i = 10_000.0;
        let t_e = 5_000.0;
        let radius = 0.2;
        let field = coil_field_t(&compression_coil(), 2.0e5).expect("valid field");
        let volume = plasma_volume_m3(radius, 1.0).expect("valid volume");
        let thermal_energy_j = 1.5 * density * volume * (t_i + t_e) * 1.602_176_634e-19;
        PulsedCompressionState {
            t_s: 0.0,
            r_s_m: radius,
            d_r_s_dt_m_s: 0.0,
            t_i_ev: t_i,
            t_e_ev: t_e,
            density_m3: density,
            beta: 0.0,
            b_ext_t: field,
            internal_pressure_pa: density * (t_i + t_e) * 1.602_176_634e-19,
            external_magnetic_pressure_pa: field * field / (2.0 * 4.0e-7 * std::f64::consts::PI),
            thermal_energy_j,
            compression_work_j: 0.0,
            radiated_loss_j: 0.0,
            energy_balance_residual: 0.0,
            flux_state: Default::default(),
        }
    }

    #[test]
    fn pulsed_compression_trajectory_drives_mrti_tracker() {
        let states = run_pulsed_compression(
            compression_state(),
            &compression_config(),
            5.0e5,
            1.0e-9,
            16,
        )
        .expect("valid compression states");
        let acceleration = effective_acceleration_from_pulsed_compression(&states, 1, -1.0)
            .expect("valid acceleration");
        let mut tracker =
            MrtiSpectrumTracker::new(1.0e4, 64, 1.0e-9, 1.0e-3, 1.0e-3).expect("valid tracker");
        let snapshots = track_mrti_from_pulsed_compression(&states, &mut tracker, 1, -1.0, 1.0)
            .expect("valid coupling");

        assert_eq!(acceleration.len(), states.len());
        assert_eq!(snapshots.len(), states.len() - 1);
        assert!(acceleration.iter().all(|value| value.is_finite()));
        assert!(
            acceleration
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                > 0.0
        );
        assert!((snapshots.last().unwrap().t_s - states.last().unwrap().t_s).abs() < 1.0e-20);
        assert!(snapshots.last().unwrap().max_amplitude_m >= snapshots[0].max_amplitude_m);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(mrti_growth_rate(-1.0, 1.0, 0.0, 1.0e-3).is_err());
        assert!(mrti_growth_rate(1.0, 1.0, 0.0, 0.0).is_err());
        assert!(MrtiSpectrumTracker::new(1.0, 1, 1.0e-9, 1.0e-3, 1.0e-3).is_err());
        assert!(effective_acceleration_from_pulsed_compression(&[], 1, -1.0).is_err());
        assert!(effective_acceleration_from_radius_rate(&[0.0, 0.0], &[1.0, 2.0], 1).is_err());
        assert!(effective_acceleration_from_radius_rate(&[0.0, 1.0], &[1.0, 2.0], 2).is_err());
    }
}
