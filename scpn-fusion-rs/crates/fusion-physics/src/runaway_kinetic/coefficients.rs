// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Runaway Kinetic Operator Coefficients
//! Complete coefficient contract for the three-axis kinetic operator.

use super::grid::{validate_array, RunawayKineticGrid};

/// Flux, diffusion, avalanche, and external-source coefficients.
#[derive(Debug, Clone)]
pub struct RunawayKineticCoefficients {
    /// Radial-face advection.
    pub radial_advection: Vec<f64>,
    /// Momentum-face electric acceleration.
    pub momentum_electric_advection: Vec<f64>,
    /// Momentum-face collisional drag.
    pub momentum_collision_advection: Vec<f64>,
    /// Momentum-face synchrotron drag.
    pub momentum_synchrotron_advection: Vec<f64>,
    /// Momentum-face bremsstrahlung drag.
    pub momentum_bremsstrahlung_advection: Vec<f64>,
    /// Pitch-face electric advection.
    pub pitch_electric_advection: Vec<f64>,
    /// Pitch-face synchrotron advection.
    pub pitch_synchrotron_advection: Vec<f64>,
    /// Non-negative radial-face diffusion.
    pub radial_diffusion: Vec<f64>,
    /// Non-negative momentum-face diffusion.
    pub momentum_diffusion: Vec<f64>,
    /// Non-negative pitch-face diffusion.
    pub pitch_diffusion: Vec<f64>,
    /// Pitch-gradient coefficient on momentum faces.
    pub momentum_pitch_diffusion: Vec<f64>,
    /// Momentum-gradient coefficient on pitch faces.
    pub pitch_momentum_diffusion: Vec<f64>,
    /// Cell-centred kinetic avalanche source kernel.
    pub avalanche_source_kernel: Vec<f64>,
    /// Total electron density by radius in inverse cubic metres.
    pub total_electron_density_m3: Vec<f64>,
    /// Total-density avalanche rate by radius in inverse seconds.
    pub total_density_avalanche_rate_s_inv: Vec<f64>,
    /// External total-density source by radius.
    pub total_density_external_source_m3_s: Vec<f64>,
    /// Cell-centred external kinetic source.
    pub external_source: Vec<f64>,
}

impl RunawayKineticCoefficients {
    /// Validate every coefficient against the exact grid topology.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid: &RunawayKineticGrid,
        radial_advection: Vec<f64>,
        momentum_electric_advection: Vec<f64>,
        momentum_collision_advection: Vec<f64>,
        momentum_synchrotron_advection: Vec<f64>,
        momentum_bremsstrahlung_advection: Vec<f64>,
        pitch_electric_advection: Vec<f64>,
        pitch_synchrotron_advection: Vec<f64>,
        radial_diffusion: Vec<f64>,
        momentum_diffusion: Vec<f64>,
        pitch_diffusion: Vec<f64>,
        momentum_pitch_diffusion: Vec<f64>,
        pitch_momentum_diffusion: Vec<f64>,
        avalanche_source_kernel: Vec<f64>,
        total_electron_density_m3: Vec<f64>,
        total_density_avalanche_rate_s_inv: Vec<f64>,
        total_density_external_source_m3_s: Vec<f64>,
        external_source: Vec<f64>,
    ) -> Result<Self, String> {
        let radial_faces = (grid.nr() + 1) * grid.nxi() * grid.np();
        let momentum_faces = grid.nr() * grid.nxi() * (grid.np() + 1);
        let pitch_faces = grid.nr() * (grid.nxi() + 1) * grid.np();
        let cells = grid.cell_count();
        validate_array("radial_advection", &radial_advection, radial_faces, false)?;
        validate_array(
            "momentum_electric_advection",
            &momentum_electric_advection,
            momentum_faces,
            false,
        )?;
        validate_array(
            "momentum_collision_advection",
            &momentum_collision_advection,
            momentum_faces,
            false,
        )?;
        validate_array(
            "momentum_synchrotron_advection",
            &momentum_synchrotron_advection,
            momentum_faces,
            false,
        )?;
        validate_array(
            "momentum_bremsstrahlung_advection",
            &momentum_bremsstrahlung_advection,
            momentum_faces,
            false,
        )?;
        validate_array(
            "pitch_electric_advection",
            &pitch_electric_advection,
            pitch_faces,
            false,
        )?;
        validate_array(
            "pitch_synchrotron_advection",
            &pitch_synchrotron_advection,
            pitch_faces,
            false,
        )?;
        validate_array("radial_diffusion", &radial_diffusion, radial_faces, true)?;
        validate_array(
            "momentum_diffusion",
            &momentum_diffusion,
            momentum_faces,
            true,
        )?;
        validate_array("pitch_diffusion", &pitch_diffusion, pitch_faces, true)?;
        validate_array(
            "momentum_pitch_diffusion",
            &momentum_pitch_diffusion,
            momentum_faces,
            false,
        )?;
        validate_array(
            "pitch_momentum_diffusion",
            &pitch_momentum_diffusion,
            pitch_faces,
            false,
        )?;
        validate_array(
            "avalanche_source_kernel",
            &avalanche_source_kernel,
            cells,
            true,
        )?;
        validate_array(
            "total_electron_density_m3",
            &total_electron_density_m3,
            grid.nr(),
            true,
        )?;
        validate_array(
            "total_density_avalanche_rate_s_inv",
            &total_density_avalanche_rate_s_inv,
            grid.nr(),
            true,
        )?;
        validate_array(
            "total_density_external_source_m3_s",
            &total_density_external_source_m3_s,
            grid.nr(),
            false,
        )?;
        validate_array("external_source", &external_source, cells, false)?;
        Ok(Self {
            radial_advection,
            momentum_electric_advection,
            momentum_collision_advection,
            momentum_synchrotron_advection,
            momentum_bremsstrahlung_advection,
            pitch_electric_advection,
            pitch_synchrotron_advection,
            radial_diffusion,
            momentum_diffusion,
            pitch_diffusion,
            momentum_pitch_diffusion,
            pitch_momentum_diffusion,
            avalanche_source_kernel,
            total_electron_density_m3,
            total_density_avalanche_rate_s_inv,
            total_density_external_source_m3_s,
            external_source,
        })
    }

    /// Total momentum advection without hiding radiation components.
    pub fn momentum_advection(&self) -> Vec<f64> {
        self.momentum_electric_advection
            .iter()
            .zip(&self.momentum_collision_advection)
            .zip(&self.momentum_synchrotron_advection)
            .zip(&self.momentum_bremsstrahlung_advection)
            .map(|(((electric, collision), synchrotron), bremsstrahlung)| {
                electric + collision + synchrotron + bremsstrahlung
            })
            .collect()
    }

    /// Total pitch advection without hiding radiation components.
    pub fn pitch_advection(&self) -> Vec<f64> {
        self.pitch_electric_advection
            .iter()
            .zip(&self.pitch_synchrotron_advection)
            .map(|(electric, synchrotron)| electric + synchrotron)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> RunawayKineticGrid {
        RunawayKineticGrid::new(vec![0.0, 1.0], vec![-1.0, 0.0, 1.0], vec![0.0, 1.0, 2.0]).unwrap()
    }

    #[test]
    fn negative_physical_diffusion_is_rejected() {
        let grid = grid();
        let radial = (grid.nr() + 1) * grid.nxi() * grid.np();
        let momentum = grid.nr() * grid.nxi() * (grid.np() + 1);
        let pitch = grid.nr() * (grid.nxi() + 1) * grid.np();
        let cell = grid.cell_count();
        let result = RunawayKineticCoefficients::new(
            &grid,
            vec![0.0; radial],
            vec![0.0; momentum],
            vec![0.0; momentum],
            vec![0.0; momentum],
            vec![0.0; momentum],
            vec![0.0; pitch],
            vec![0.0; pitch],
            vec![-1.0; radial],
            vec![0.0; momentum],
            vec![0.0; pitch],
            vec![0.0; momentum],
            vec![0.0; pitch],
            vec![0.0; cell],
            vec![0.0; grid.nr()],
            vec![0.0; grid.nr()],
            vec![0.0; grid.nr()],
            vec![0.0; cell],
        );
        assert!(result.is_err());
    }
}
