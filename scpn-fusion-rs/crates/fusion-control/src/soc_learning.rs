// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — SOC Learning
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Self-Organized Criticality with Q-Learning controller.
//!
//! Port of `advanced_soc_fusion_learning.py`.
//! Coupled sandpile + zonal flow + tabular RL agent.

use rand::Rng;

/// System size. Python: 60.
const L: usize = 60;

/// Base critical gradient. Python: 6.0.
const Z_CRIT_BASE: f64 = 6.0;

/// Flow generation rate. Python: 0.2.
const FLOW_GEN: f64 = 0.2;

/// Flow damping rate. Python: 0.05.
const FLOW_DAMP: f64 = 0.05;

/// Flow → threshold efficiency. Python: 3.0.
const SHEAR_EFF: f64 = 3.0;

/// Q-learning rate. Python: 0.1.
const ALPHA: f64 = 0.1;

/// Discount factor. Python: 0.95.
const GAMMA: f64 = 0.95;

/// Exploration rate. Python: 0.1.
const EPSILON: f64 = 0.1;

/// State discretization bins.
const N_TURB: usize = 5;
const N_FLOW: usize = 5;
const N_ACTIONS: usize = 3;

/// Coupled sandpile reactor with zonal flows.
pub struct CoupledSandpile {
    pub z: Vec<f64>,
    pub h: Vec<f64>,
    pub flow: f64,
}

impl CoupledSandpile {
    pub fn new() -> Self {
        CoupledSandpile {
            z: vec![0.0; L],
            h: vec![0.0; L],
            flow: 0.0,
        }
    }

    /// Drive: add gradient at core.
    pub fn drive(&mut self) {
        self.z[0] += 1.0;
        self.h[0] += 1.0;
    }

    /// Relax with effective critical gradient. Returns avalanche size.
    pub fn relax(&mut self, ext_shear: f64) -> usize {
        let eff_shear = self.flow + ext_shear;
        let z_crit = Z_CRIT_BASE + SHEAR_EFF * eff_shear;
        let mut total_topple = 0;

        for _ in 0..50 {
            let mut any_active = false;
            for i in 0..L {
                if self.z[i] >= z_crit {
                    self.z[i] -= 2.0;
                    if i + 1 < L {
                        self.z[i + 1] += 1.0;
                    }
                    if i > 0 {
                        self.z[i - 1] += 1.0;
                    }
                    total_topple += 1;
                    any_active = true;
                }
            }
            if !any_active {
                break;
            }
        }

        // Update zonal flow
        self.flow += total_topple as f64 * FLOW_GEN / L as f64;
        self.flow *= 1.0 - FLOW_DAMP;
        self.flow = self.flow.clamp(0.0, 5.0);

        total_topple
    }
}

impl Default for CoupledSandpile {
    fn default() -> Self {
        Self::new()
    }
}

/// Tabular Q-learning agent.
pub struct FusionAgent {
    pub q_table: Vec<f64>, // [N_TURB × N_FLOW × N_ACTIONS]
    pub total_reward: f64,
    last_state: (usize, usize),
    last_action: usize,
}

impl FusionAgent {
    pub fn new() -> Self {
        FusionAgent {
            q_table: vec![0.0; N_TURB * N_FLOW * N_ACTIONS],
            total_reward: 0.0,
            last_state: (0, 0),
            last_action: 0,
        }
    }

    fn idx(&self, s: (usize, usize), a: usize) -> usize {
        s.0 * N_FLOW * N_ACTIONS + s.1 * N_ACTIONS + a
    }

    fn discretize(&self, turb: f64, flow: f64) -> (usize, usize) {
        let s_turb = ((turb + 1.0).ln() as usize).min(N_TURB - 1);
        let s_flow = (flow as usize).min(N_FLOW - 1);
        (s_turb, s_flow)
    }

    /// Choose action (ε-greedy).
    pub fn choose_action(&self, state: (usize, usize)) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < EPSILON {
            rng.gen_range(0..N_ACTIONS)
        } else {
            let mut best_a = 0;
            let mut best_q = f64::NEG_INFINITY;
            for a in 0..N_ACTIONS {
                let q = self.q_table[self.idx(state, a)];
                if q > best_q {
                    best_q = q;
                    best_a = a;
                }
            }
            best_a
        }
    }

    /// Q-learning update.
    pub fn learn(&mut self, new_state: (usize, usize), reward: f64) {
        let old_idx = self.idx(self.last_state, self.last_action);
        let old_q = self.q_table[old_idx];

        let max_future = (0..N_ACTIONS)
            .map(|a| self.q_table[self.idx(new_state, a)])
            .fold(f64::NEG_INFINITY, f64::max);

        self.q_table[old_idx] = old_q + ALPHA * (reward + GAMMA * max_future - old_q);
        self.total_reward += reward;
    }

    /// Full step: observe → choose → learn.
    pub fn step(&mut self, turb: f64, flow: f64, reward: f64) -> f64 {
        let state = self.discretize(turb, flow);
        self.learn(state, reward);
        let action = self.choose_action(state);
        self.last_state = state;
        self.last_action = action;
        // Action: 0=decrease, 1=maintain, 2=increase external shear
        match action {
            0 => -0.1,
            2 => 0.1,
            _ => 0.0,
        }
    }
}

impl Default for FusionAgent {
    fn default() -> Self {
        Self::new()
    }
}

/// Run coupled SOC + RL simulation.
pub fn run_soc_learning(n_steps: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut sandpile = CoupledSandpile::new();
    let mut agent = FusionAgent::new();
    let mut ext_shear = 0.0;

    let mut turb_history = Vec::with_capacity(n_steps);
    let mut flow_history = Vec::with_capacity(n_steps);
    let mut reward_history = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        sandpile.drive();
        let avalanche = sandpile.relax(ext_shear);

        let core_temp = sandpile.h[0];
        let reward = core_temp * 0.1 - avalanche as f64 * 0.5 - ext_shear.abs() * 2.0;

        ext_shear = agent.step(avalanche as f64, sandpile.flow, reward);

        turb_history.push(avalanche as f64);
        flow_history.push(sandpile.flow);
        reward_history.push(reward);
    }

    (turb_history, flow_history, reward_history)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandpile_drives() {
        let mut sp = CoupledSandpile::new();
        sp.drive();
        assert!((sp.z[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_avalanche_occurs() {
        let mut sp = CoupledSandpile::new();
        let mut any_avalanche = false;
        for _ in 0..1000 {
            sp.drive();
            let a = sp.relax(0.0);
            if a > 0 {
                any_avalanche = true;
            }
        }
        assert!(any_avalanche, "Should have avalanches in 1000 steps");
    }

    #[test]
    fn test_q_learning_runs() {
        let (turb, flow, reward) = run_soc_learning(1000);
        assert_eq!(turb.len(), 1000);
        assert!(flow.iter().all(|v| v.is_finite()));
        assert!(reward.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_flow_bounded() {
        let mut sp = CoupledSandpile::new();
        for _ in 0..10000 {
            sp.drive();
            sp.relax(0.5);
        }
        assert!(
            sp.flow >= 0.0 && sp.flow <= 5.0,
            "Flow should be bounded: {}",
            sp.flow
        );
    }
}
