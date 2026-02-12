// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Sandpile
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! BTW sandpile model for self-organized criticality in fusion plasmas.
//!
//! Port of `sandpile_fusion_reactor.py`.
//! Models avalanche-like transport events in tokamak edge plasma
//! using a 1D cellular automaton with HJB-inspired controller.

/// Number of radial cells. Python: L=100.
const L: usize = 100;

/// Critical gradient threshold. Python: Z_CRIT_BASE=4.0.
const Z_CRIT_BASE: f64 = 4.0;

/// Maximum suppression factor from controller.
const MAX_SUPPRESSION: f64 = 0.5;

/// BTW sandpile state.
pub struct SandpileReactor {
    /// Height profile (analogous to temperature/pressure gradient).
    pub heights: Vec<f64>,
    /// Critical gradient (may vary with controller).
    pub z_crit: Vec<f64>,
    /// Total grains lost at edge (transport flux).
    pub total_flux: f64,
    /// Avalanche size history.
    pub avalanche_history: Vec<usize>,
    /// HJB suppression field.
    pub suppression: Vec<f64>,
}

impl SandpileReactor {
    pub fn new() -> Self {
        SandpileReactor {
            heights: vec![0.0; L],
            z_crit: vec![Z_CRIT_BASE; L],
            total_flux: 0.0,
            avalanche_history: Vec::new(),
            suppression: vec![0.0; L],
        }
    }

    /// Drive: add one grain to the core (index 0).
    pub fn drive(&mut self) {
        self.heights[0] += 1.0;
    }

    /// Relax: topple any site where gradient exceeds critical.
    /// Returns avalanche size (number of toppling events).
    pub fn relax(&mut self) -> usize {
        let mut avalanche_size = 0;
        let mut changed = true;

        while changed {
            changed = false;
            for i in 0..L - 1 {
                let gradient = self.heights[i] - self.heights[i + 1];
                let effective_crit = self.z_crit[i] * (1.0 + self.suppression[i]);

                if gradient > effective_crit {
                    // Topple: transfer from i to i+1
                    let transfer = 1.0 * (1.0 - self.suppression[i] * MAX_SUPPRESSION);
                    self.heights[i] -= transfer;
                    self.heights[i + 1] += transfer;
                    avalanche_size += 1;
                    changed = true;
                }
            }

            // Edge loss
            if self.heights[L - 1] > Z_CRIT_BASE {
                let lost = self.heights[L - 1] - Z_CRIT_BASE;
                self.total_flux += lost;
                self.heights[L - 1] = Z_CRIT_BASE;
            }
        }

        self.avalanche_history.push(avalanche_size);
        avalanche_size
    }

    /// HJB controller: heuristic policy.
    /// Maps (avalanche_size, core_temperature) → suppression action.
    pub fn update_controller(&mut self, avalanche_size: usize) {
        let core_temp = self.heights[0];
        let norm_aval = (avalanche_size as f64) / (L as f64);
        let norm_temp = core_temp / (Z_CRIT_BASE * L as f64);

        // Simple policy: increase suppression near core if avalanches are large
        for i in 0..L {
            let radial_factor = 1.0 - (i as f64) / (L as f64);
            self.suppression[i] = (norm_aval * 0.5 + norm_temp * 0.5) * radial_factor;
            self.suppression[i] = self.suppression[i].clamp(0.0, 1.0);
        }
    }

    /// Full step: drive + relax + control.
    pub fn step(&mut self) -> usize {
        self.drive();
        let aval = self.relax();
        self.update_controller(aval);
        aval
    }

    /// Run N steps, return avalanche sizes.
    pub fn run(&mut self, n_steps: usize) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            sizes.push(self.step());
        }
        sizes
    }
}

impl Default for SandpileReactor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandpile_creation() {
        let sp = SandpileReactor::new();
        assert_eq!(sp.heights.len(), L);
        assert_eq!(sp.z_crit[0], Z_CRIT_BASE);
    }

    #[test]
    fn test_sandpile_drive() {
        let mut sp = SandpileReactor::new();
        sp.drive();
        assert!((sp.heights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sandpile_avalanche_occurs() {
        let mut sp = SandpileReactor::new();
        // Drive enough to trigger avalanche
        let sizes = sp.run(1000);
        // At least some avalanches should have size > 0
        let nonzero: usize = sizes.iter().filter(|&&s| s > 0).count();
        assert!(nonzero > 0, "Should have some avalanches in 1000 steps");
    }

    #[test]
    fn test_avalanche_power_law_hint() {
        let mut sp = SandpileReactor::new();
        let sizes = sp.run(10000);
        // Power law means many small, few large
        let small: usize = sizes.iter().filter(|&&s| s <= 5).count();
        let large: usize = sizes.iter().filter(|&&s| s > 50).count();
        // Should have more small than large (rough power law check)
        assert!(small > large * 2, "Power law: small={small}, large={large}");
    }
}
