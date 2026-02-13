//! Symplectic integration utilities for canonical Hamiltonian systems.
//!
//! This module provides a reduced velocity-Verlet integrator for long-horizon
//! invariant-preserving integration, plus an RK4 reference stepper for
//! regression comparison.

/// Canonical 1D phase-space state `(q, p)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanonicalState {
    pub q: f64,
    pub p: f64,
}

/// Hamiltonian system contract in canonical coordinates.
pub trait HamiltonianSystem {
    /// Partial derivative of Hamiltonian wrt coordinate: ∂H/∂q.
    fn d_h_dq(&self, q: f64, p: f64) -> f64;
    /// Partial derivative of Hamiltonian wrt momentum: ∂H/∂p.
    fn d_h_dp(&self, q: f64, p: f64) -> f64;
    /// Hamiltonian energy `H(q, p)`.
    fn hamiltonian(&self, q: f64, p: f64) -> f64;
}

/// Perform one velocity-Verlet step.
pub fn velocity_verlet_step<S: HamiltonianSystem>(state: &mut CanonicalState, system: &S, dt: f64) {
    if !dt.is_finite() || dt == 0.0 {
        return;
    }

    let p_half = state.p - 0.5 * dt * system.d_h_dq(state.q, state.p);
    let q_new = state.q + dt * system.d_h_dp(state.q, p_half);
    let p_new = p_half - 0.5 * dt * system.d_h_dq(q_new, p_half);

    state.q = q_new;
    state.p = p_new;
}

/// Perform one RK4 step on canonical equations:
/// `q_dot = ∂H/∂p`, `p_dot = -∂H/∂q`.
pub fn rk4_canonical_step<S: HamiltonianSystem>(state: &mut CanonicalState, system: &S, dt: f64) {
    if !dt.is_finite() || dt == 0.0 {
        return;
    }

    let f = |q: f64, p: f64| -> (f64, f64) { (system.d_h_dp(q, p), -system.d_h_dq(q, p)) };

    let (k1q, k1p) = f(state.q, state.p);
    let (k2q, k2p) = f(state.q + 0.5 * dt * k1q, state.p + 0.5 * dt * k1p);
    let (k3q, k3p) = f(state.q + 0.5 * dt * k2q, state.p + 0.5 * dt * k2p);
    let (k4q, k4p) = f(state.q + dt * k3q, state.p + dt * k3p);

    state.q += dt * (k1q + 2.0 * k2q + 2.0 * k3q + k4q) / 6.0;
    state.p += dt * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0;
}

/// Integrate a trajectory with velocity-Verlet.
pub fn integrate_velocity_verlet<S: HamiltonianSystem>(
    initial: CanonicalState,
    system: &S,
    dt: f64,
    steps: usize,
) -> Vec<CanonicalState> {
    let mut traj = Vec::with_capacity(steps + 1);
    let mut state = initial;
    traj.push(state);
    for _ in 0..steps {
        velocity_verlet_step(&mut state, system, dt);
        traj.push(state);
    }
    traj
}

/// Integrate a trajectory with RK4.
pub fn integrate_rk4<S: HamiltonianSystem>(
    initial: CanonicalState,
    system: &S,
    dt: f64,
    steps: usize,
) -> Vec<CanonicalState> {
    let mut traj = Vec::with_capacity(steps + 1);
    let mut state = initial;
    traj.push(state);
    for _ in 0..steps {
        rk4_canonical_step(&mut state, system, dt);
        traj.push(state);
    }
    traj
}

/// Maximum absolute Hamiltonian drift along trajectory relative to first point.
pub fn max_energy_drift<S: HamiltonianSystem>(trajectory: &[CanonicalState], system: &S) -> f64 {
    let Some(first) = trajectory.first().copied() else {
        return 0.0;
    };
    let e0 = system.hamiltonian(first.q, first.p);
    trajectory
        .iter()
        .map(|s| (system.hamiltonian(s.q, s.p) - e0).abs())
        .fold(0.0_f64, f64::max)
}

/// Harmonic oscillator benchmark system:
/// `H = 0.5 * (p^2 + (ω q)^2)`.
#[derive(Debug, Clone, Copy)]
pub struct HarmonicOscillator {
    pub omega: f64,
}

impl HamiltonianSystem for HarmonicOscillator {
    fn d_h_dq(&self, q: f64, _p: f64) -> f64 {
        self.omega * self.omega * q
    }

    fn d_h_dp(&self, _q: f64, p: f64) -> f64 {
        p
    }

    fn hamiltonian(&self, q: f64, p: f64) -> f64 {
        0.5 * (p * p + (self.omega * q) * (self.omega * q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_verlet_zero_dt_identity() {
        let sys = HarmonicOscillator { omega: 1.0 };
        let mut s = CanonicalState { q: 1.2, p: -0.4 };
        let original = s;
        velocity_verlet_step(&mut s, &sys, 0.0);
        assert_eq!(s, original);
    }

    #[test]
    fn test_velocity_verlet_energy_bounded_on_long_horizon() {
        let sys = HarmonicOscillator { omega: 1.0 };
        let initial = CanonicalState { q: 1.0, p: 0.0 };
        let traj = integrate_velocity_verlet(initial, &sys, 0.3, 5000);
        let drift = max_energy_drift(&traj, &sys);
        assert!(
            drift < 0.02,
            "Expected bounded long-horizon drift for symplectic integrator, got {drift}"
        );
    }

    #[test]
    fn test_velocity_verlet_outperforms_rk4_drift_on_coarse_step() {
        let sys = HarmonicOscillator { omega: 1.0 };
        let initial = CanonicalState { q: 1.0, p: 0.0 };
        let verlet = integrate_velocity_verlet(initial, &sys, 0.3, 5000);
        let rk4 = integrate_rk4(initial, &sys, 0.3, 5000);
        let verlet_drift = max_energy_drift(&verlet, &sys);
        let rk4_drift = max_energy_drift(&rk4, &sys);
        assert!(
            verlet_drift < rk4_drift,
            "Expected symplectic drift < RK4 drift on long horizon, got verlet={verlet_drift}, rk4={rk4_drift}"
        );
    }
}
