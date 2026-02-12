//! Balance of Plant: power flow chain for a fusion reactor.
//!
//! Port of `balance_of_plant.py`.
//! Models thermal-to-electric conversion and parasitic loads.

/// Rankine cycle efficiency. Python: 0.35.
const ETA_THERMAL: f64 = 0.35;

/// NBI/ECRH wall-plug efficiency. Python: 0.40.
const ETA_HEATING: f64 = 0.40;

/// Cryogenic plant load [MW]. Python: 30.
const P_CRYO: f64 = 30.0;

/// Vacuum pump load [MW]. Python: 10.
const P_PUMP: f64 = 10.0;

/// Misc BOP (control, HVAC) [MW]. Python: 15.
const P_BOP_MISC: f64 = 15.0;

/// Neutron power fraction. Python: 0.80.
const P_NEUTRON_FRAC: f64 = 0.80;

/// Alpha power fraction. Python: 0.20.
const P_ALPHA_FRAC: f64 = 0.20;

/// Blanket energy multiplication. Python: 1.15.
const M_BLANKET: f64 = 1.15;

/// Full power plant performance metrics.
#[derive(Debug, Clone)]
pub struct PlantMetrics {
    /// Input fusion power [MW].
    pub p_fusion: f64,
    /// Absorbed auxiliary heating [MW].
    pub p_aux_absorbed: f64,
    /// Total thermal power [MW].
    pub p_thermal: f64,
    /// Gross electric [MW].
    pub p_gross: f64,
    /// Total recirculating power [MW].
    pub p_recirc: f64,
    /// Net electric to grid [MW].
    pub p_net: f64,
    /// Plasma Q = P_fus / P_aux.
    pub q_plasma: f64,
    /// Engineering Q = P_gross / P_recirc.
    pub q_eng: f64,
    /// Heating wall-plug power [MW].
    pub p_heating_wallplug: f64,
}

/// Power plant model.
pub struct PowerPlantModel;

impl PowerPlantModel {
    pub fn new() -> Self {
        PowerPlantModel
    }

    /// Calculate full plant performance.
    pub fn calculate(&self, p_fusion_mw: f64, p_aux_absorbed_mw: f64) -> PlantMetrics {
        let p_neutron = P_NEUTRON_FRAC * p_fusion_mw;
        let p_alpha = P_ALPHA_FRAC * p_fusion_mw;

        // Blanket energy multiplication (exothermic Li-6 reactions)
        let p_thermal_blanket = p_neutron * M_BLANKET;

        // Total thermal: blanket + alpha (deposited in plasma → divertor) + aux
        let p_thermal = p_thermal_blanket + p_alpha + p_aux_absorbed_mw;

        // Gross electric
        let p_gross = p_thermal * ETA_THERMAL;

        // Heating wall-plug requirement
        let p_heating_wallplug = p_aux_absorbed_mw / ETA_HEATING;

        // Total recirculating
        let p_recirc = P_CRYO + P_PUMP + P_BOP_MISC + p_heating_wallplug;

        // Net to grid
        let p_net = p_gross - p_recirc;

        // Performance figures
        let q_plasma = if p_aux_absorbed_mw > 0.0 {
            p_fusion_mw / p_aux_absorbed_mw
        } else {
            f64::INFINITY
        };
        let q_eng = if p_recirc > 0.0 {
            p_gross / p_recirc
        } else {
            f64::INFINITY
        };

        PlantMetrics {
            p_fusion: p_fusion_mw,
            p_aux_absorbed: p_aux_absorbed_mw,
            p_thermal,
            p_gross,
            p_recirc,
            p_net,
            q_plasma,
            q_eng,
            p_heating_wallplug,
        }
    }
}

impl Default for PowerPlantModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_like() {
        let plant = PowerPlantModel::new();
        let m = plant.calculate(500.0, 50.0);
        assert!(
            (m.q_plasma - 10.0).abs() < 0.01,
            "Q_plasma should be 10: {}",
            m.q_plasma
        );
    }

    #[test]
    fn test_thermal_power_breakdown() {
        let plant = PowerPlantModel::new();
        let m = plant.calculate(500.0, 50.0);
        // P_neutron = 400, P_alpha = 100, P_blanket = 400*1.15 = 460
        // P_thermal = 460 + 100 + 50 = 610
        assert!(
            (m.p_thermal - 610.0).abs() < 0.1,
            "P_thermal should be 610: {}",
            m.p_thermal
        );
    }

    #[test]
    fn test_net_power_positive() {
        let plant = PowerPlantModel::new();
        let m = plant.calculate(500.0, 50.0);
        assert!(m.p_net > 0.0, "Net power should be positive: {}", m.p_net);
        assert!(m.q_eng > 1.0, "Q_eng should be > 1: {}", m.q_eng);
    }

    #[test]
    fn test_low_fusion_negative_net() {
        let plant = PowerPlantModel::new();
        let m = plant.calculate(100.0, 50.0);
        // 100 MW fusion with 50 MW heating → should be net-negative
        assert!(
            m.p_net < 0.0,
            "Low fusion power should give negative net: {}",
            m.p_net
        );
    }

    #[test]
    fn test_heating_wallplug() {
        let plant = PowerPlantModel::new();
        let m = plant.calculate(500.0, 50.0);
        // P_heating_wallplug = 50 / 0.40 = 125
        assert!(
            (m.p_heating_wallplug - 125.0).abs() < 0.1,
            "Heating wall-plug should be 125: {}",
            m.p_heating_wallplug
        );
    }
}
