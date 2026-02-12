// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — SPI
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Shattered Pellet Injection (SPI) disruption mitigation.
//!
//! Port of `spi_mitigation.py`.
//! Models thermal quench via radiation and current quench via resistive decay.

/// Default thermal energy [MJ]. Python: 300.
const W_TH_MJ: f64 = 300.0;

/// Default plasma current [MA]. Python: 15.
const IP_MA: f64 = 15.0;

/// Default electron temperature [keV]. Python: 20.
const TE_INIT: f64 = 20.0;

/// Temperature floor [keV].
const TE_FLOOR: f64 = 0.01;

/// Thermal quench threshold [keV]. Python: 0.1.
const TQ_THRESHOLD: f64 = 0.1;

/// Pellet assimilation time [s]. Python: 0.002.
const T_MIX: f64 = 0.002;

/// Simulation timestep [s]. Python: 1e-5.
const DT: f64 = 1e-5;

/// Total simulation time [s]. Python: 0.05.
const T_TOTAL: f64 = 0.05;

/// Radiation power coefficient [W·keV^{-0.5}].
/// Calibrated so thermal quench completes in ~10 ms for ITER-class parameters.
const P_RAD_COEFF: f64 = 1e10;

/// Plasma inductance [H]. Python: 1e-6.
const L_PLASMA: f64 = 1e-6;

/// Disruption phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Assimilation,
    ThermalQuench,
    CurrentQuench,
}

/// SPI time-step snapshot.
#[derive(Debug, Clone)]
pub struct SPISnapshot {
    pub time: f64,
    pub w_th_mj: f64,
    pub ip_ma: f64,
    pub te_kev: f64,
    pub phase: Phase,
}

/// Shattered Pellet Injection simulator.
pub struct SPIMitigation {
    pub w_th: f64,  // [J]
    pub ip: f64,    // [A]
    pub te: f64,    // [keV]
    pub phase: Phase,
}

impl SPIMitigation {
    pub fn new(w_th_mj: f64, ip_ma: f64, te_kev: f64) -> Self {
        SPIMitigation {
            w_th: w_th_mj * 1e6,
            ip: ip_ma * 1e6,
            te: te_kev,
            phase: Phase::Assimilation,
        }
    }

    /// Run full SPI simulation. Returns time history.
    pub fn run(&mut self) -> Vec<SPISnapshot> {
        let n_steps = (T_TOTAL / DT) as usize;
        let mut history = Vec::with_capacity(n_steps);
        let mut t = 0.0;

        for _ in 0..n_steps {
            history.push(SPISnapshot {
                time: t,
                w_th_mj: self.w_th / 1e6,
                ip_ma: self.ip / 1e6,
                te_kev: self.te,
                phase: self.phase,
            });

            match self.phase {
                Phase::Assimilation => {
                    if t > T_MIX {
                        self.phase = Phase::ThermalQuench;
                    }
                }
                Phase::ThermalQuench => {
                    // Radiation cooling
                    let p_rad = P_RAD_COEFF * self.te.sqrt();
                    let dw = -p_rad * DT;
                    let w_old = self.w_th;
                    self.w_th = (self.w_th + dw).max(0.0);

                    // Temperature tracks thermal energy
                    if w_old > 0.0 {
                        self.te = (self.te * self.w_th / w_old).max(TE_FLOOR);
                    }

                    if self.te < TQ_THRESHOLD {
                        self.phase = Phase::CurrentQuench;
                    }
                }
                Phase::CurrentQuench => {
                    // Spitzer resistivity: η ∝ Te^{-3/2}
                    let r_plasma = 1e-6 / self.te.powf(1.5);
                    let di = -(r_plasma / L_PLASMA) * self.ip * DT;
                    self.ip = (self.ip + di).max(0.0);
                }
            }

            t += DT;
        }

        history
    }
}

impl Default for SPIMitigation {
    fn default() -> Self {
        Self::new(W_TH_MJ, IP_MA, TE_INIT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spi_thermal_quench() {
        let mut spi = SPIMitigation::default();
        let history = spi.run();
        // Should reach current quench phase
        let reached_cq = history.iter().any(|s| s.phase == Phase::CurrentQuench);
        assert!(reached_cq, "Should reach current quench phase");
    }

    #[test]
    fn test_spi_energy_decreases() {
        let mut spi = SPIMitigation::default();
        let history = spi.run();
        let first = history.first().unwrap().w_th_mj;
        let last = history.last().unwrap().w_th_mj;
        assert!(
            last < first,
            "Thermal energy should decrease: {first} → {last}"
        );
    }

    #[test]
    fn test_spi_current_decreases() {
        let mut spi = SPIMitigation::default();
        let history = spi.run();
        let first = history.first().unwrap().ip_ma;
        let last = history.last().unwrap().ip_ma;
        assert!(
            last < first,
            "Plasma current should decrease: {first} → {last}"
        );
    }

    #[test]
    fn test_spi_phases_sequential() {
        let mut spi = SPIMitigation::default();
        let history = spi.run();
        // Phases should go: Assimilation → ThermalQuench → CurrentQuench
        let mut saw_tq = false;
        let mut saw_cq = false;
        for s in &history {
            match s.phase {
                Phase::ThermalQuench => saw_tq = true,
                Phase::CurrentQuench => {
                    assert!(saw_tq, "CQ before TQ");
                    saw_cq = true;
                }
                _ => {}
            }
        }
        assert!(saw_cq, "Should reach CQ");
    }
}
