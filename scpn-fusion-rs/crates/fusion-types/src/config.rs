// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Config
//! Governed reactor, grid, physics, coil, and solver configuration types.

use serde::{Deserialize, Serialize};

/// Maximum accepted serialized reactor-configuration size in bytes.
pub const MAX_CONFIG_BYTES: u64 = 10 * 1024 * 1024;

/// Top-level reactor configuration.
/// Maps 1:1 to iter_config.json schema.
/// Must deserialize ALL 6 existing JSON config files without modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorConfig {
    /// Human-readable reactor or scenario name.
    pub reactor_name: String,
    /// Grid resolution as `[nr, nz]`.
    pub grid_resolution: [usize; 2],
    /// Rectangular R-Z domain bounds.
    pub dimensions: GridDimensions,
    /// Plasma-current and permeability parameters.
    pub physics: PhysicsParams,
    /// External poloidal-field coil definitions.
    pub coils: Vec<CoilConfig>,
    /// Iterative solver controls.
    pub solver: SolverConfig,
}

/// Inclusive rectangular R-Z domain bounds in metres.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridDimensions {
    /// Minimum major radius in metres.
    #[serde(rename = "R_min")]
    pub r_min: f64,
    /// Maximum major radius in metres.
    #[serde(rename = "R_max")]
    pub r_max: f64,
    /// Minimum vertical coordinate in metres.
    #[serde(rename = "Z_min")]
    pub z_min: f64,
    /// Maximum vertical coordinate in metres.
    #[serde(rename = "Z_max")]
    pub z_max: f64,
}

/// Core physical parameters for equilibrium reconstruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParams {
    /// Target total toroidal plasma current in configuration units.
    pub plasma_current_target: f64,
    /// Vacuum permeability, either SI or the configured normalised value.
    pub vacuum_permeability: f64,
    /// Optional H-mode pedestal profile configuration.
    /// When absent, the solver uses L-mode linear profiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profiles: Option<ProfileConfig>,
}

/// H-mode pedestal profile parameters (optional in JSON config).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Profile mode: "l-mode" or "h-mode"
    pub mode: String,
    /// Pressure gradient pedestal parameters
    #[serde(default)]
    pub p_prime: PedestalParams,
    /// Poloidal current pedestal parameters
    #[serde(default)]
    pub ff_prime: PedestalParams,
}

/// Pedestal shape parameters for a single profile (p' or FF').
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedestalParams {
    /// Pedestal top location in normalized flux (default: 0.92)
    #[serde(default = "default_ped_top")]
    pub ped_top: f64,
    /// Pedestal width in normalized flux (default: 0.05)
    #[serde(default = "default_ped_width")]
    pub ped_width: f64,
    /// Pedestal height, relative (default: 1.0)
    #[serde(default = "default_ped_height")]
    pub ped_height: f64,
    /// Core peaking factor (default: 0.3)
    #[serde(default = "default_core_alpha")]
    pub core_alpha: f64,
}

fn default_ped_top() -> f64 {
    0.92
}
fn default_ped_width() -> f64 {
    0.05
}
fn default_ped_height() -> f64 {
    1.0
}
fn default_core_alpha() -> f64 {
    0.3
}

impl Default for PedestalParams {
    fn default() -> Self {
        PedestalParams {
            ped_top: default_ped_top(),
            ped_width: default_ped_width(),
            ped_height: default_ped_height(),
            core_alpha: default_core_alpha(),
        }
    }
}

/// One external poloidal-field coil.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoilConfig {
    /// Optional human-readable coil name.
    #[serde(default)]
    pub name: String,
    /// Coil-centre major radius in metres.
    pub r: f64,
    /// Coil-centre vertical coordinate in metres.
    pub z: f64,
    /// Coil current in configuration units.
    pub current: f64,
}

/// Iterative equilibrium-solver controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Maximum number of nonlinear solver iterations.
    pub max_iterations: usize,
    /// Residual threshold required for convergence.
    pub convergence_threshold: f64,
    /// Outer fixed-point relaxation factor.
    pub relaxation_factor: f64,
    /// Successive-over-relaxation factor, defaulting to `1.6`.
    #[serde(default = "default_sor_omega")]
    pub sor_omega: f64,
}

fn default_sor_omega() -> f64 {
    1.6
}

impl ReactorConfig {
    /// Loads a reactor configuration from a bounded JSON file.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::FusionError::Io`] when metadata or file reading
    /// fails, [`crate::error::FusionError::ConfigError`] when the file exceeds
    /// [`MAX_CONFIG_BYTES`], or [`crate::error::FusionError::Json`] when the
    /// serialized configuration is invalid.
    pub fn from_file(path: &str) -> crate::error::FusionResult<Self> {
        let metadata = std::fs::metadata(path)?;
        if metadata.len() > MAX_CONFIG_BYTES {
            return Err(crate::error::FusionError::ConfigError(format!(
                "configuration file exceeds {MAX_CONFIG_BYTES} byte limit: {path}"
            )));
        }
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Creates a [`crate::state::Grid2D`] from the configured domain and
    /// `[nr, nz]` resolution.
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
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_config.json")).unwrap();
        assert_eq!(cfg.reactor_name, "ITER-Like-Demo");
        assert_eq!(cfg.grid_resolution, [128, 128]);
        assert_eq!(cfg.coils.len(), 7);
        assert_eq!(cfg.coils[0].name, "PF1");
        assert!((cfg.coils[0].r - 3.5).abs() < 1e-10);
        assert!((cfg.coils[0].current - (-1.0)).abs() < 1e-10);
        assert_eq!(cfg.solver.max_iterations, 500);
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
        assert_eq!(cfg.grid_resolution, [129, 129]);
    }

    #[test]
    fn test_load_all_six_configs() {
        let configs = [
            "validation/iter_config.json",
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
    fn test_rejects_oversized_config_before_reading() {
        let path = std::env::temp_dir().join(format!(
            "scpn_oversized_config_{}_{}.json",
            std::process::id(),
            "reactor"
        ));
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_CONFIG_BYTES + 1).unwrap();

        let result = ReactorConfig::from_file(path.to_str().unwrap());

        std::fs::remove_file(&path).unwrap();
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("configuration file exceeds"));
    }

    #[test]
    fn test_roundtrip_serialization() {
        let cfg = ReactorConfig::from_file(&config_path("validation/iter_config.json")).unwrap();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let cfg2: ReactorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.reactor_name, cfg2.reactor_name);
        assert_eq!(cfg.grid_resolution, cfg2.grid_resolution);
        assert_eq!(cfg.coils.len(), cfg2.coils.len());
    }
}
