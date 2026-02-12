// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Config
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
use serde::{Deserialize, Serialize};

/// Top-level reactor configuration.
/// Maps 1:1 to iter_config.json schema.
/// Must deserialize ALL 6 existing JSON config files without modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorConfig {
    pub reactor_name: String,
    pub grid_resolution: [usize; 2],
    pub dimensions: GridDimensions,
    pub physics: PhysicsParams,
    pub coils: Vec<CoilConfig>,
    pub solver: SolverConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridDimensions {
    #[serde(rename = "R_min")]
    pub r_min: f64,
    #[serde(rename = "R_max")]
    pub r_max: f64,
    #[serde(rename = "Z_min")]
    pub z_min: f64,
    #[serde(rename = "Z_max")]
    pub z_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParams {
    pub plasma_current_target: f64,
    pub vacuum_permeability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoilConfig {
    pub name: String,
    pub r: f64,
    pub z: f64,
    pub current: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub relaxation_factor: f64,
}

impl ReactorConfig {
    /// Load from JSON file. Must succeed for all 6 existing configs.
    pub fn from_file(path: &str) -> crate::error::FusionResult<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Create a Grid2D from this config's dimensions and resolution.
    pub fn create_grid(&self) -> crate::state::Grid2D {
        crate::state::Grid2D::new(
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.dimensions.r_min,
            self.dimensions.r_max,
            self.dimensions.z_min,
            self.dimensions.z_max,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Build path relative to the SCPN-Fusion-Core project root.
    /// CARGO_MANIFEST_DIR points to crates/fusion-types/ at compile time,
    /// so we go up 3 levels to reach SCPN-Fusion-Core/.
    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
    }

    fn config_path(relative: &str) -> String {
        project_root().join(relative).to_string_lossy().to_string()
    }

    #[test]
    fn test_load_iter_config() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        assert_eq!(cfg.reactor_name, "ITER-Like-Demo");
        assert_eq!(cfg.grid_resolution, [128, 128]);
        assert_eq!(cfg.coils.len(), 7);
        assert_eq!(cfg.coils[0].name, "PF1");
        assert!((cfg.coils[0].r - 3.0).abs() < 1e-10);
        assert!((cfg.coils[0].current - 8.0).abs() < 1e-10);
        assert_eq!(cfg.solver.max_iterations, 1000);
        assert!((cfg.solver.convergence_threshold - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn test_load_validated_config() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_validated_config.json"))
            .unwrap();
        assert_eq!(cfg.reactor_name, "ITER-Validated");
        assert_eq!(cfg.grid_resolution, [65, 65]);
        assert_eq!(cfg.coils.len(), 7);
    }

    #[test]
    fn test_load_default_config() {
        let cfg =
            ReactorConfig::from_file(&config_path("src/scpn_fusion/core/default_config.json"))
                .unwrap();
        assert_eq!(cfg.reactor_name, "SCPN-Standard-Model");
        assert_eq!(cfg.grid_resolution, [65, 65]);
    }

    #[test]
    fn test_load_all_six_configs() {
        let configs = [
            "iter_config.json",
            "validation/iter_validated_config.json",
            "validation/iter_genetic_config.json",
            "validation/iter_analytic_config.json",
            "validation/iter_force_balanced.json",
            "src/scpn_fusion/core/default_config.json",
        ];
        for relative in &configs {
            let path = config_path(relative);
            let result = ReactorConfig::from_file(&path);
            assert!(result.is_ok(), "Failed to load config: {}", path);
        }
    }

    #[test]
    fn test_roundtrip_serialization() {
        let cfg = ReactorConfig::from_file(&config_path("iter_config.json")).unwrap();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let cfg2: ReactorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.reactor_name, cfg2.reactor_name);
        assert_eq!(cfg.grid_resolution, cfg2.grid_resolution);
        assert_eq!(cfg.coils.len(), cfg2.coils.len());
    }
}
