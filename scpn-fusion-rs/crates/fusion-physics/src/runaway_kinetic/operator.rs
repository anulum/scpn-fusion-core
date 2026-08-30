// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Conservative Runaway Kinetic Operator
//! Conservative finite-volume evolution on radius, momentum, and pitch.

use std::f64::consts::PI;

use super::coefficients::RunawayKineticCoefficients;
use super::grid::{validate_array, RunawayKineticGrid};

/// Cell and oriented-face measures used by conservative divergence.
#[derive(Debug, Clone)]
pub struct RunawayKineticGeometry {
    /// Phase-space cell measure.
    pub cell_measure: Vec<f64>,
    /// Momentum-pitch measure used for radially resolved density.
    pub density_cell_measure: Vec<f64>,
    /// Radial-face measure.
    pub radial_face_measure: Vec<f64>,
    /// Momentum-face measure.
    pub momentum_face_measure: Vec<f64>,
    /// Pitch-face measure.
    pub pitch_face_measure: Vec<f64>,
}

impl RunawayKineticGeometry {
    /// Construct exact tensor-product cylindrical phase-space measures.
    pub fn cylindrical(grid: &RunawayKineticGrid) -> Self {
        let radial_shell = grid.radial_shell_measure_m2();
        let momentum_shell = grid.momentum_shell_measure();
        let mut cell_measure = vec![0.0; grid.cell_count()];
        let mut density_cell_measure = vec![0.0; grid.cell_count()];
        let mut radial_face_measure = vec![0.0; (grid.nr() + 1) * grid.nxi() * grid.np()];
        let mut momentum_face_measure = vec![0.0; grid.nr() * grid.nxi() * (grid.np() + 1)];
        let mut pitch_face_measure = vec![0.0; grid.nr() * (grid.nxi() + 1) * grid.np()];

        for ir in 0..grid.nr() {
            for ixi in 0..grid.nxi() {
                let dxi = grid.pitch_faces[ixi + 1] - grid.pitch_faces[ixi];
                for (ip, momentum_measure) in momentum_shell.iter().enumerate() {
                    let density_measure = dxi * 2.0 * PI * momentum_measure;
                    let cell = grid.cell_index(ir, ixi, ip);
                    density_cell_measure[cell] = density_measure;
                    cell_measure[cell] = radial_shell[ir] * density_measure;
                }
                for ipf in 0..=grid.np() {
                    momentum_face_measure[grid.momentum_face_index(ir, ixi, ipf)] =
                        radial_shell[ir] * dxi * 2.0 * PI * grid.momentum_faces_mc[ipf].powi(2);
                }
            }
            for ixif in 0..=grid.nxi() {
                for ip in 0..grid.np() {
                    pitch_face_measure[grid.pitch_face_index(ir, ixif, ip)] =
                        radial_shell[ir] * 2.0 * PI * momentum_shell[ip];
                }
            }
        }
        for irf in 0..=grid.nr() {
            for ixi in 0..grid.nxi() {
                let dxi = grid.pitch_faces[ixi + 1] - grid.pitch_faces[ixi];
                for ip in 0..grid.np() {
                    radial_face_measure[grid.radial_face_index(irf, ixi, ip)] =
                        grid.radius_faces_m[irf] * dxi * 2.0 * PI * momentum_shell[ip];
                }
            }
        }
        Self {
            cell_measure,
            density_cell_measure,
            radial_face_measure,
            momentum_face_measure,
            pitch_face_measure,
        }
    }

    /// Validate independently supplied geometry against the exact grid topology.
    pub fn checked(
        grid: &RunawayKineticGrid,
        cell_measure: Vec<f64>,
        density_cell_measure: Vec<f64>,
        radial_face_measure: Vec<f64>,
        momentum_face_measure: Vec<f64>,
        pitch_face_measure: Vec<f64>,
    ) -> Result<Self, String> {
        validate_array("cell_measure", &cell_measure, grid.cell_count(), true)?;
        if cell_measure.iter().any(|value| *value <= 0.0) {
            return Err("cell_measure must be strictly positive".to_string());
        }
        validate_array(
            "density_cell_measure",
            &density_cell_measure,
            grid.cell_count(),
            true,
        )?;
        validate_array(
            "radial_face_measure",
            &radial_face_measure,
            (grid.nr() + 1) * grid.nxi() * grid.np(),
            true,
        )?;
        validate_array(
            "momentum_face_measure",
            &momentum_face_measure,
            grid.nr() * grid.nxi() * (grid.np() + 1),
            true,
        )?;
        validate_array(
            "pitch_face_measure",
            &pitch_face_measure,
            grid.nr() * (grid.nxi() + 1) * grid.np(),
            true,
        )?;
        Ok(Self {
            cell_measure,
            density_cell_measure,
            radial_face_measure,
            momentum_face_measure,
            pitch_face_measure,
        })
    }
}

/// Independently auditable contributions to the complete kinetic tendency.
#[derive(Debug, Clone)]
pub struct RunawayKineticTendencies {
    /// Radial advection and diffusion.
    pub radial_transport: Vec<f64>,
    /// Momentum and pitch electric acceleration.
    pub electric_acceleration: Vec<f64>,
    /// Momentum collisional drag and diffusion.
    pub collisional_drag_diffusion: Vec<f64>,
    /// Pitch diffusion.
    pub pitch_scattering: Vec<f64>,
    /// Momentum-pitch cross diffusion.
    pub cross_diffusion: Vec<f64>,
    /// Momentum and pitch synchrotron losses.
    pub synchrotron_loss: Vec<f64>,
    /// Momentum bremsstrahlung losses.
    pub bremsstrahlung_loss: Vec<f64>,
    /// Kinetic avalanche generation.
    pub avalanche_generation: Vec<f64>,
    /// External kinetic source.
    pub external_source: Vec<f64>,
    /// Momentum-pitch-integrated radial transport by radius.
    pub runaway_density_radial_transport_m3_s: Vec<f64>,
    /// Total-density avalanche generation by radius.
    pub runaway_density_avalanche_generation_m3_s: Vec<f64>,
    /// External total-density source by radius.
    pub runaway_density_external_source_m3_s: Vec<f64>,
    /// Complete total-density tendency by radius.
    pub runaway_density_tendency_m3_s: Vec<f64>,
}

impl RunawayKineticTendencies {
    /// Sum every declared kinetic contribution.
    pub fn total(&self) -> Vec<f64> {
        (0..self.radial_transport.len())
            .map(|i| {
                self.radial_transport[i]
                    + self.electric_acceleration[i]
                    + self.collisional_drag_diffusion[i]
                    + self.pitch_scattering[i]
                    + self.cross_diffusion[i]
                    + self.synchrotron_loss[i]
                    + self.bremsstrahlung_loss[i]
                    + self.avalanche_generation[i]
                    + self.external_source[i]
            })
            .collect()
    }
}

/// Complete conservative radius-momentum-pitch operator.
#[derive(Debug, Clone)]
pub struct RunawayKineticOperator {
    /// Physical grid.
    pub grid: RunawayKineticGrid,
    /// Complete coefficient bundle.
    pub coefficients: RunawayKineticCoefficients,
    /// Conservative geometry.
    pub geometry: RunawayKineticGeometry,
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    Radius,
    Pitch,
    Momentum,
}

impl RunawayKineticOperator {
    /// Construct with cylindrical geometry.
    pub fn new(grid: RunawayKineticGrid, coefficients: RunawayKineticCoefficients) -> Self {
        let geometry = RunawayKineticGeometry::cylindrical(&grid);
        Self {
            grid,
            coefficients,
            geometry,
        }
    }

    /// Construct with independently supplied geometry.
    pub fn with_geometry(
        grid: RunawayKineticGrid,
        coefficients: RunawayKineticCoefficients,
        geometry: RunawayKineticGeometry,
    ) -> Self {
        Self {
            grid,
            coefficients,
            geometry,
        }
    }

    fn axis_len(&self, axis: Axis) -> usize {
        match axis {
            Axis::Radius => self.grid.nr(),
            Axis::Pitch => self.grid.nxi(),
            Axis::Momentum => self.grid.np(),
        }
    }

    fn face_len(&self, axis: Axis) -> usize {
        match axis {
            Axis::Radius => (self.grid.nr() + 1) * self.grid.nxi() * self.grid.np(),
            Axis::Pitch => self.grid.nr() * (self.grid.nxi() + 1) * self.grid.np(),
            Axis::Momentum => self.grid.nr() * self.grid.nxi() * (self.grid.np() + 1),
        }
    }

    fn cell_at(&self, state: &[f64], axis: Axis, a: usize, b: usize, c: usize) -> f64 {
        match axis {
            Axis::Radius => state[self.grid.cell_index(c, a, b)],
            Axis::Pitch => state[self.grid.cell_index(a, c, b)],
            Axis::Momentum => state[self.grid.cell_index(a, b, c)],
        }
    }

    fn face_index(&self, axis: Axis, a: usize, b: usize, face: usize) -> usize {
        match axis {
            Axis::Radius => self.grid.radial_face_index(face, a, b),
            Axis::Pitch => self.grid.pitch_face_index(a, face, b),
            Axis::Momentum => self.grid.momentum_face_index(a, b, face),
        }
    }

    fn other_extents(&self, axis: Axis) -> (usize, usize) {
        match axis {
            Axis::Radius => (self.grid.nxi(), self.grid.np()),
            Axis::Pitch => (self.grid.nr(), self.grid.np()),
            Axis::Momentum => (self.grid.nr(), self.grid.nxi()),
        }
    }

    fn upwind_faces(
        &self,
        state: &[f64],
        advection: &[f64],
        axis: Axis,
        zero_low: bool,
        zero_high: bool,
    ) -> Vec<f64> {
        let mut face_state = vec![0.0; self.face_len(axis)];
        let n = self.axis_len(axis);
        let (na, nb) = self.other_extents(axis);
        for a in 0..na {
            for b in 0..nb {
                for face in 0..=n {
                    let index = self.face_index(axis, a, b, face);
                    face_state[index] = if face == 0 {
                        if !zero_low && advection[index] < 0.0 {
                            self.cell_at(state, axis, a, b, 0)
                        } else {
                            0.0
                        }
                    } else if face == n {
                        if !zero_high && advection[index] > 0.0 {
                            self.cell_at(state, axis, a, b, n - 1)
                        } else {
                            0.0
                        }
                    } else if advection[index] >= 0.0 {
                        self.cell_at(state, axis, a, b, face - 1)
                    } else {
                        self.cell_at(state, axis, a, b, face)
                    };
                }
            }
        }
        face_state
    }

    fn face_gradient(
        &self,
        state: &[f64],
        centres: &[f64],
        faces: &[f64],
        axis: Axis,
        zero_high: bool,
        high_boundary_distance: Option<f64>,
    ) -> Vec<f64> {
        let mut gradient = vec![0.0; self.face_len(axis)];
        let n = self.axis_len(axis);
        let (na, nb) = self.other_extents(axis);
        for a in 0..na {
            for b in 0..nb {
                for face in 1..n {
                    let index = self.face_index(axis, a, b, face);
                    gradient[index] = (self.cell_at(state, axis, a, b, face)
                        - self.cell_at(state, axis, a, b, face - 1))
                        / (centres[face] - centres[face - 1]);
                }
                if !zero_high {
                    let index = self.face_index(axis, a, b, n);
                    let distance = high_boundary_distance.unwrap_or(faces[n] - centres[n - 1]);
                    gradient[index] = -self.cell_at(state, axis, a, b, n - 1) / distance;
                }
            }
        }
        gradient
    }

    fn divergence(&self, flux: &[f64], face_measure: &[f64], axis: Axis) -> Vec<f64> {
        let mut result = vec![0.0; self.grid.cell_count()];
        for ir in 0..self.grid.nr() {
            for ixi in 0..self.grid.nxi() {
                for ip in 0..self.grid.np() {
                    let (a, b, cell_axis) = match axis {
                        Axis::Radius => (ixi, ip, ir),
                        Axis::Pitch => (ir, ip, ixi),
                        Axis::Momentum => (ir, ixi, ip),
                    };
                    let low = self.face_index(axis, a, b, cell_axis);
                    let high = self.face_index(axis, a, b, cell_axis + 1);
                    let cell = self.grid.cell_index(ir, ixi, ip);
                    result[cell] = -(flux[high] * face_measure[high]
                        - flux[low] * face_measure[low])
                        / self.geometry.cell_measure[cell];
                }
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn advection_tendency(
        &self,
        state: &[f64],
        advection: &[f64],
        face_measure: &[f64],
        axis: Axis,
        upwind_advection: Option<&[f64]>,
        zero_low: bool,
        zero_high: bool,
    ) -> Vec<f64> {
        let interpolation = upwind_advection.unwrap_or(advection);
        let faces = self.upwind_faces(state, interpolation, axis, zero_low, zero_high);
        let flux: Vec<f64> = advection
            .iter()
            .zip(faces)
            .map(|(speed, value)| speed * value)
            .collect();
        self.divergence(&flux, face_measure, axis)
    }

    #[allow(clippy::too_many_arguments)]
    fn diffusion_tendency(
        &self,
        state: &[f64],
        diffusion: &[f64],
        face_measure: &[f64],
        centres: &[f64],
        faces: &[f64],
        axis: Axis,
        zero_high: bool,
        high_boundary_distance: Option<f64>,
    ) -> Vec<f64> {
        let gradient = self.face_gradient(
            state,
            centres,
            faces,
            axis,
            zero_high,
            high_boundary_distance,
        );
        let flux: Vec<f64> = diffusion
            .iter()
            .zip(gradient)
            .map(|(coefficient, derivative)| -coefficient * derivative)
            .collect();
        self.divergence(&flux, face_measure, axis)
    }

    fn cell_gradient(&self, state: &[f64], axis: Axis, centres: &[f64]) -> Vec<f64> {
        let n = self.axis_len(axis);
        if n == 1 {
            return vec![0.0; self.grid.cell_count()];
        }
        let mut result = vec![0.0; self.grid.cell_count()];
        let (na, nb) = self.other_extents(axis);
        for a in 0..na {
            for b in 0..nb {
                for cell_axis in 0..n {
                    let value = if cell_axis == 0 {
                        (self.cell_at(state, axis, a, b, 1) - self.cell_at(state, axis, a, b, 0))
                            / (centres[1] - centres[0])
                    } else if cell_axis == n - 1 {
                        (self.cell_at(state, axis, a, b, n - 1)
                            - self.cell_at(state, axis, a, b, n - 2))
                            / (centres[n - 1] - centres[n - 2])
                    } else {
                        let h_left = centres[cell_axis] - centres[cell_axis - 1];
                        let h_right = centres[cell_axis + 1] - centres[cell_axis];
                        -h_right / (h_left * (h_left + h_right))
                            * self.cell_at(state, axis, a, b, cell_axis - 1)
                            + (h_right - h_left) / (h_left * h_right)
                                * self.cell_at(state, axis, a, b, cell_axis)
                            + h_left / (h_right * (h_left + h_right))
                                * self.cell_at(state, axis, a, b, cell_axis + 1)
                    };
                    let index = match axis {
                        Axis::Radius => self.grid.cell_index(cell_axis, a, b),
                        Axis::Pitch => self.grid.cell_index(a, cell_axis, b),
                        Axis::Momentum => self.grid.cell_index(a, b, cell_axis),
                    };
                    result[index] = value;
                }
            }
        }
        result
    }

    fn cross_gradient(&self, state: &[f64], face_axis: Axis, grad_axis: Axis) -> Vec<f64> {
        let centres = match grad_axis {
            Axis::Radius => self.grid.radius_m(),
            Axis::Pitch => self.grid.pitch(),
            Axis::Momentum => self.grid.momentum_mc(),
        };
        let cell_gradient = self.cell_gradient(state, grad_axis, &centres);
        let mut result = vec![0.0; self.face_len(face_axis)];
        let n = self.axis_len(face_axis);
        let (na, nb) = self.other_extents(face_axis);
        for a in 0..na {
            for b in 0..nb {
                for face in 0..=n {
                    let value = if face == 0 {
                        self.cell_at(&cell_gradient, face_axis, a, b, 0)
                    } else if face == n {
                        self.cell_at(&cell_gradient, face_axis, a, b, n - 1)
                    } else {
                        0.5 * (self.cell_at(&cell_gradient, face_axis, a, b, face - 1)
                            + self.cell_at(&cell_gradient, face_axis, a, b, face))
                    };
                    result[self.face_index(face_axis, a, b, face)] = value;
                }
            }
        }
        result
    }

    /// Evaluate every operator contribution for one finite distribution.
    pub fn evaluate(
        &self,
        distribution: &[f64],
        runaway_density_m3: Option<&[f64]>,
    ) -> Result<RunawayKineticTendencies, String> {
        let state = self.grid.require_state("distribution", distribution)?;
        let c = &self.coefficients;
        let g = &self.geometry;
        let radius = self.grid.radius_m();
        let pitch = self.grid.pitch();
        let momentum = self.grid.momentum_mc();
        let momentum_advection = c.momentum_advection();
        let pitch_advection = c.pitch_advection();
        let radial_distance = if self.grid.nr() > 1 {
            radius[self.grid.nr() - 1] - radius[self.grid.nr() - 2]
        } else {
            self.grid.radius_faces_m[1] - self.grid.radius_faces_m[0]
        };
        let momentum_distance = if self.grid.np() > 1 {
            momentum[self.grid.np() - 1] - momentum[self.grid.np() - 2]
        } else {
            self.grid.momentum_faces_mc[1] - self.grid.momentum_faces_mc[0]
        };

        let radial_advection = self.advection_tendency(
            &state,
            &c.radial_advection,
            &g.radial_face_measure,
            Axis::Radius,
            None,
            false,
            false,
        );
        let radial_diffusion = self.diffusion_tendency(
            &state,
            &c.radial_diffusion,
            &g.radial_face_measure,
            &radius,
            &self.grid.radius_faces_m,
            Axis::Radius,
            false,
            Some(radial_distance),
        );
        let radial_transport = add(&radial_advection, &radial_diffusion);

        let electric_momentum = self.advection_tendency(
            &state,
            &c.momentum_electric_advection,
            &g.momentum_face_measure,
            Axis::Momentum,
            Some(&momentum_advection),
            true,
            false,
        );
        let electric_pitch = self.advection_tendency(
            &state,
            &c.pitch_electric_advection,
            &g.pitch_face_measure,
            Axis::Pitch,
            Some(&pitch_advection),
            false,
            false,
        );
        let electric_acceleration = add(&electric_momentum, &electric_pitch);

        let collision_advection = self.advection_tendency(
            &state,
            &c.momentum_collision_advection,
            &g.momentum_face_measure,
            Axis::Momentum,
            Some(&momentum_advection),
            true,
            false,
        );
        let collision_diffusion = self.diffusion_tendency(
            &state,
            &c.momentum_diffusion,
            &g.momentum_face_measure,
            &momentum,
            &self.grid.momentum_faces_mc,
            Axis::Momentum,
            false,
            Some(momentum_distance),
        );
        let collisional_drag_diffusion = add(&collision_advection, &collision_diffusion);

        let pitch_scattering = self.diffusion_tendency(
            &state,
            &c.pitch_diffusion,
            &g.pitch_face_measure,
            &pitch,
            &self.grid.pitch_faces,
            Axis::Pitch,
            true,
            None,
        );

        let momentum_gradient = self.cross_gradient(&state, Axis::Momentum, Axis::Pitch);
        let pitch_gradient = self.cross_gradient(&state, Axis::Pitch, Axis::Momentum);
        let momentum_cross_flux: Vec<f64> = c
            .momentum_pitch_diffusion
            .iter()
            .zip(momentum_gradient)
            .map(|(coefficient, gradient)| -coefficient * gradient)
            .collect();
        let pitch_cross_flux: Vec<f64> = c
            .pitch_momentum_diffusion
            .iter()
            .zip(pitch_gradient)
            .map(|(coefficient, gradient)| -coefficient * gradient)
            .collect();
        let cross_diffusion = add(
            &self.divergence(
                &momentum_cross_flux,
                &g.momentum_face_measure,
                Axis::Momentum,
            ),
            &self.divergence(&pitch_cross_flux, &g.pitch_face_measure, Axis::Pitch),
        );

        let synchrotron_loss = add(
            &self.advection_tendency(
                &state,
                &c.momentum_synchrotron_advection,
                &g.momentum_face_measure,
                Axis::Momentum,
                Some(&momentum_advection),
                true,
                false,
            ),
            &self.advection_tendency(
                &state,
                &c.pitch_synchrotron_advection,
                &g.pitch_face_measure,
                Axis::Pitch,
                Some(&pitch_advection),
                false,
                false,
            ),
        );
        let bremsstrahlung_loss = self.advection_tendency(
            &state,
            &c.momentum_bremsstrahlung_advection,
            &g.momentum_face_measure,
            Axis::Momentum,
            Some(&momentum_advection),
            true,
            false,
        );

        let runaway_density = if let Some(values) = runaway_density_m3 {
            validate_array("runaway_density_m3", values, self.grid.nr(), true)?;
            values.to_vec()
        } else {
            self.integrated_density(&state)
        };
        let mut avalanche_generation = vec![0.0; self.grid.cell_count()];
        for (ir, density) in runaway_density.iter().enumerate() {
            for ixi in 0..self.grid.nxi() {
                for ip in 0..self.grid.np() {
                    let cell = self.grid.cell_index(ir, ixi, ip);
                    avalanche_generation[cell] =
                        c.avalanche_source_kernel[cell] * c.total_electron_density_m3[ir] * density;
                }
            }
        }

        let mut runaway_density_radial_transport_m3_s = vec![0.0; self.grid.nr()];
        for (ir, radial_rate) in runaway_density_radial_transport_m3_s.iter_mut().enumerate() {
            for ixi in 0..self.grid.nxi() {
                for ip in 0..self.grid.np() {
                    let cell = self.grid.cell_index(ir, ixi, ip);
                    *radial_rate += radial_transport[cell] * g.density_cell_measure[cell];
                }
            }
        }
        let runaway_density_avalanche_generation_m3_s: Vec<f64> = (0..self.grid.nr())
            .map(|ir| c.total_density_avalanche_rate_s_inv[ir] * runaway_density[ir])
            .collect();
        let runaway_density_tendency_m3_s: Vec<f64> = (0..self.grid.nr())
            .map(|ir| {
                runaway_density_radial_transport_m3_s[ir]
                    + runaway_density_avalanche_generation_m3_s[ir]
                    + c.total_density_external_source_m3_s[ir]
            })
            .collect();

        Ok(RunawayKineticTendencies {
            radial_transport,
            electric_acceleration,
            collisional_drag_diffusion,
            pitch_scattering,
            cross_diffusion,
            synchrotron_loss,
            bremsstrahlung_loss,
            avalanche_generation,
            external_source: c.external_source.clone(),
            runaway_density_radial_transport_m3_s,
            runaway_density_avalanche_generation_m3_s,
            runaway_density_external_source_m3_s: c.total_density_external_source_m3_s.clone(),
            runaway_density_tendency_m3_s,
        })
    }

    /// Integrate a kinetic state over momentum and pitch at every radius.
    pub fn integrated_density(&self, state: &[f64]) -> Vec<f64> {
        let mut density = vec![0.0; self.grid.nr()];
        for (ir, radial_density) in density.iter_mut().enumerate() {
            for ixi in 0..self.grid.nxi() {
                for ip in 0..self.grid.np() {
                    let cell = self.grid.cell_index(ir, ixi, ip);
                    *radial_density += state[cell] * self.geometry.density_cell_measure[cell];
                }
            }
        }
        density
    }
}

fn add(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}
