// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Conservative Runaway Kinetic Grid
//! Validated radius-pitch-momentum finite-volume grids.

use std::f64::consts::PI;

/// Evolved tensor-product grid in radius-pitch-momentum order.
#[derive(Debug, Clone)]
pub struct RunawayKineticGrid {
    /// Radial cell faces in metres.
    pub radius_faces_m: Vec<f64>,
    /// Pitch-cosine cell faces.
    pub pitch_faces: Vec<f64>,
    /// Momentum cell faces in electron-rest-momentum units.
    pub momentum_faces_mc: Vec<f64>,
}

impl RunawayKineticGrid {
    /// Construct a grid after strict finite, bound, and monotonicity checks.
    pub fn new(
        radius_faces_m: Vec<f64>,
        pitch_faces: Vec<f64>,
        momentum_faces_mc: Vec<f64>,
    ) -> Result<Self, String> {
        validate_faces("radius_faces_m", &radius_faces_m, Some(0.0), None)?;
        validate_faces("pitch_faces", &pitch_faces, Some(-1.0), Some(1.0))?;
        validate_faces("momentum_faces_mc", &momentum_faces_mc, Some(0.0), None)?;
        Ok(Self {
            radius_faces_m,
            pitch_faces,
            momentum_faces_mc,
        })
    }

    /// Number of radial cells.
    #[inline]
    pub fn nr(&self) -> usize {
        self.radius_faces_m.len() - 1
    }

    /// Number of pitch cells.
    #[inline]
    pub fn nxi(&self) -> usize {
        self.pitch_faces.len() - 1
    }

    /// Number of momentum cells.
    #[inline]
    pub fn np(&self) -> usize {
        self.momentum_faces_mc.len() - 1
    }

    /// Total number of phase-space cells.
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.nr() * self.nxi() * self.np()
    }

    /// Flatten a radius-pitch-momentum cell index.
    #[inline]
    pub fn cell_index(&self, ir: usize, ixi: usize, ip: usize) -> usize {
        (ir * self.nxi() + ixi) * self.np() + ip
    }

    /// Flatten a radial-face index.
    #[inline]
    pub fn radial_face_index(&self, ir: usize, ixi: usize, ip: usize) -> usize {
        (ir * self.nxi() + ixi) * self.np() + ip
    }

    /// Flatten a pitch-face index.
    #[inline]
    pub fn pitch_face_index(&self, ir: usize, ixi: usize, ip: usize) -> usize {
        (ir * (self.nxi() + 1) + ixi) * self.np() + ip
    }

    /// Flatten a momentum-face index.
    #[inline]
    pub fn momentum_face_index(&self, ir: usize, ixi: usize, ip: usize) -> usize {
        (ir * self.nxi() + ixi) * (self.np() + 1) + ip
    }

    /// Radial cell centres.
    pub fn radius_m(&self) -> Vec<f64> {
        centres(&self.radius_faces_m)
    }

    /// Pitch cell centres.
    pub fn pitch(&self) -> Vec<f64> {
        centres(&self.pitch_faces)
    }

    /// Momentum cell centres.
    pub fn momentum_mc(&self) -> Vec<f64> {
        centres(&self.momentum_faces_mc)
    }

    /// Cylindrical radial shell measures `(r_hi^2-r_lo^2)/2`.
    pub fn radial_shell_measure_m2(&self) -> Vec<f64> {
        self.radius_faces_m
            .windows(2)
            .map(|face| 0.5 * (face[1].powi(2) - face[0].powi(2)))
            .collect()
    }

    /// Spherical momentum shell measures `(p_hi^3-p_lo^3)/3`.
    pub fn momentum_shell_measure(&self) -> Vec<f64> {
        self.momentum_faces_mc
            .windows(2)
            .map(|face| (face[1].powi(3) - face[0].powi(3)) / 3.0)
            .collect()
    }

    /// Axisymmetric cylindrical phase-space cell measures.
    pub fn phase_space_cell_measure(&self) -> Vec<f64> {
        let radial = self.radial_shell_measure_m2();
        let momentum = self.momentum_shell_measure();
        let mut result = vec![0.0; self.cell_count()];
        for (ir, radial_measure) in radial.iter().enumerate() {
            for ixi in 0..self.nxi() {
                let pitch_width = self.pitch_faces[ixi + 1] - self.pitch_faces[ixi];
                for (ip, momentum_measure) in momentum.iter().enumerate() {
                    result[self.cell_index(ir, ixi, ip)] =
                        radial_measure * pitch_width * 2.0 * PI * momentum_measure;
                }
            }
        }
        result
    }

    /// Validate one flattened finite state tensor.
    pub fn require_state(&self, name: &str, values: &[f64]) -> Result<Vec<f64>, String> {
        validate_array(name, values, self.cell_count(), false)?;
        Ok(values.to_vec())
    }
}

pub(crate) fn validate_array(
    name: &str,
    values: &[f64],
    expected: usize,
    nonnegative: bool,
) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!(
            "{name} must contain {expected} values, got {}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} contains a non-finite value"));
    }
    if nonnegative && values.iter().any(|value| *value < 0.0) {
        return Err(format!("{name} must be non-negative"));
    }
    Ok(())
}

fn validate_faces(
    name: &str,
    values: &[f64],
    lower: Option<f64>,
    upper: Option<f64>,
) -> Result<(), String> {
    if values.len() < 2 {
        return Err(format!("{name} must contain at least two faces"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} contains a non-finite coordinate"));
    }
    if let Some(bound) = lower {
        if values[0] < bound {
            return Err(format!("{name} starts below {bound}"));
        }
    }
    if let Some(bound) = upper {
        if values[values.len() - 1] > bound {
            return Err(format!("{name} ends above {bound}"));
        }
    }
    if values.windows(2).any(|face| face[1] <= face[0]) {
        return Err(format!("{name} must be strictly increasing"));
    }
    Ok(())
}

fn centres(faces: &[f64]) -> Vec<f64> {
    faces
        .windows(2)
        .map(|face| 0.5 * (face[0] + face[1]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_preserves_all_three_physical_axes() {
        let grid = RunawayKineticGrid::new(
            vec![0.0, 0.5, 1.0],
            vec![-1.0, 0.0, 1.0],
            vec![0.0, 1.0, 3.0],
        )
        .unwrap();
        assert_eq!((grid.nr(), grid.nxi(), grid.np()), (2, 2, 2));
        assert_eq!(grid.phase_space_cell_measure().len(), 8);
        assert!(grid
            .phase_space_cell_measure()
            .iter()
            .all(|value| *value > 0.0));
    }

    #[test]
    fn grid_rejects_projected_or_duplicate_axes() {
        assert!(
            RunawayKineticGrid::new(vec![0.0, 1.0], vec![-1.0, -1.0, 1.0], vec![0.0, 1.0],)
                .is_err()
        );
    }
}
