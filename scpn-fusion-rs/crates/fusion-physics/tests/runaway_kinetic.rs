// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Full Runaway Kinetic Integration Tests

use fusion_physics::runaway_kinetic::{
    RunawayKineticCoefficients, RunawayKineticGeometry, RunawayKineticGrid, RunawayKineticOperator,
    RunawayKineticSolver,
};

fn fixture() -> (RunawayKineticOperator, Vec<f64>, Vec<f64>) {
    let grid = RunawayKineticGrid::new(
        vec![0.0, 0.4, 1.0],
        vec![-1.0, -0.3, 0.4, 1.0],
        vec![0.0, 0.5, 1.5, 3.0, 6.0],
    )
    .unwrap();
    let radial_faces = (grid.nr() + 1) * grid.nxi() * grid.np();
    let momentum_faces = grid.nr() * grid.nxi() * (grid.np() + 1);
    let pitch_faces = grid.nr() * (grid.nxi() + 1) * grid.np();
    let cells = grid.cell_count();
    let coefficients = RunawayKineticCoefficients::new(
        &grid,
        (0..radial_faces)
            .map(|index| 1.0e-3 * (1.0 + index as f64 / radial_faces as f64))
            .collect(),
        vec![4.0e-2; momentum_faces],
        vec![-2.0e-2; momentum_faces],
        vec![-3.0e-3; momentum_faces],
        vec![-2.0e-3; momentum_faces],
        vec![5.0e-3; pitch_faces],
        vec![-1.0e-3; pitch_faces],
        vec![2.0e-5; radial_faces],
        vec![3.0e-5; momentum_faces],
        vec![4.0e-5; pitch_faces],
        vec![5.0e-6; momentum_faces],
        vec![7.0e-6; pitch_faces],
        vec![2.0e-25; cells],
        vec![1.0e19, 2.0e19],
        vec![2.0, 3.0],
        vec![1.0e7, 2.0e7],
        vec![3.0e6; cells],
    )
    .unwrap();
    let state: Vec<f64> = (0..cells)
        .map(|index| 1.0e10 * (1.0 + index as f64 / cells as f64))
        .collect();
    let density = vec![1.0e12, 2.0e12];
    (
        RunawayKineticOperator::new(grid, coefficients),
        state,
        density,
    )
}

fn zero_physics(operator: &mut RunawayKineticOperator) {
    operator.coefficients.radial_advection.fill(0.0);
    operator.coefficients.momentum_electric_advection.fill(0.0);
    operator.coefficients.momentum_collision_advection.fill(0.0);
    operator
        .coefficients
        .momentum_synchrotron_advection
        .fill(0.0);
    operator
        .coefficients
        .momentum_bremsstrahlung_advection
        .fill(0.0);
    operator.coefficients.pitch_electric_advection.fill(0.0);
    operator.coefficients.pitch_synchrotron_advection.fill(0.0);
    operator.coefficients.radial_diffusion.fill(0.0);
    operator.coefficients.momentum_diffusion.fill(0.0);
    operator.coefficients.pitch_diffusion.fill(0.0);
    operator.coefficients.momentum_pitch_diffusion.fill(0.0);
    operator.coefficients.pitch_momentum_diffusion.fill(0.0);
    operator.coefficients.avalanche_source_kernel.fill(0.0);
    operator
        .coefficients
        .total_density_avalanche_rate_s_inv
        .fill(0.0);
    operator
        .coefficients
        .total_density_external_source_m3_s
        .fill(0.0);
    operator.coefficients.external_source.fill(0.0);
}

#[test]
fn operator_reports_every_required_component_and_integrated_radial_transport() {
    let (operator, state, density) = fixture();
    let tendency = operator.evaluate(&state, Some(&density)).unwrap();
    for component in [
        &tendency.radial_transport,
        &tendency.electric_acceleration,
        &tendency.collisional_drag_diffusion,
        &tendency.pitch_scattering,
        &tendency.cross_diffusion,
        &tendency.synchrotron_loss,
        &tendency.bremsstrahlung_loss,
        &tendency.avalanche_generation,
        &tendency.external_source,
    ] {
        assert_eq!(component.len(), operator.grid.cell_count());
        assert!(component.iter().any(|value| value.abs() > 0.0));
    }
    for ir in 0..operator.grid.nr() {
        let mut integrated = 0.0;
        for ixi in 0..operator.grid.nxi() {
            for ip in 0..operator.grid.np() {
                let cell = operator.grid.cell_index(ir, ixi, ip);
                integrated +=
                    tendency.radial_transport[cell] * operator.geometry.density_cell_measure[cell];
            }
        }
        assert_eq!(
            integrated,
            tendency.runaway_density_radial_transport_m3_s[ir]
        );
        assert!(tendency.runaway_density_avalanche_generation_m3_s[ir] > 0.0);
    }
}

#[test]
fn solver_returns_unprojected_ssprk3_history_and_moments() {
    let (operator, state, density) = fixture();
    let cells = operator.grid.cell_count();
    let nr = operator.grid.nr();
    let solver = RunawayKineticSolver::new(operator, 2.5e-7, 1.0e-6).unwrap();
    let trajectory = solver
        .solve(&state, &[0.0, 5.0e-7, 1.0e-6], Some(&density))
        .unwrap();
    assert_eq!(trajectory.distribution.len(), 3 * cells);
    assert_eq!(trajectory.radial_transport.len(), 3 * cells);
    assert_eq!(trajectory.avalanche_generation.len(), 3 * cells);
    assert_eq!(trajectory.runaway_density_m3.len(), 3 * nr);
    assert_eq!(trajectory.moments.current_density_a_m2.len(), 3 * nr);
    assert_eq!(trajectory.internal_steps, 4);
    assert!(trajectory.minimum_distribution > 0.0);
}

#[test]
fn imported_geometry_rejects_negative_measures() {
    let (operator, _, _) = fixture();
    let grid = &operator.grid;
    let mut pitch_face_measure = operator.geometry.pitch_face_measure.clone();
    pitch_face_measure[0] = -1.0;
    let error = RunawayKineticGeometry::checked(
        grid,
        operator.geometry.cell_measure.clone(),
        operator.geometry.density_cell_measure.clone(),
        operator.geometry.radial_face_measure.clone(),
        operator.geometry.momentum_face_measure.clone(),
        pitch_face_measure,
    )
    .unwrap_err();
    assert_eq!(error, "pitch_face_measure must be non-negative");
}

#[test]
fn pitch_advection_closes_both_physical_pitch_boundaries() {
    let (mut operator, _, density) = fixture();
    zero_physics(&mut operator);
    operator.coefficients.pitch_electric_advection.fill(1.0);
    let pitch = operator.grid.pitch();
    let mut state = vec![0.0; operator.grid.cell_count()];
    for ir in 0..operator.grid.nr() {
        for (ixi, pitch_value) in pitch.iter().enumerate() {
            for ip in 0..operator.grid.np() {
                state[operator.grid.cell_index(ir, ixi, ip)] = 2.0 + pitch_value;
            }
        }
    }

    let electric = operator
        .evaluate(&state, Some(&density))
        .unwrap()
        .electric_acceleration;
    let integrated: f64 = electric
        .iter()
        .zip(&operator.geometry.cell_measure)
        .map(|(value, measure)| value * measure)
        .sum();

    assert!(electric.iter().any(|value| value.abs() > 0.0));
    assert!(integrated.abs() < 1.0e-12);
}

#[test]
fn solver_fails_closed_on_nonfinite_evolved_distribution() {
    let (mut operator, _, _) = fixture();
    zero_physics(&mut operator);
    operator.coefficients.external_source.fill(f64::MAX);
    let initial = vec![1.0; operator.grid.cell_count()];
    let density = vec![0.0; operator.grid.nr()];
    let solver = RunawayKineticSolver::new(operator, 1.0, 0.0).unwrap();

    let error = solver
        .solve(&initial, &[0.0, 1.0], Some(&density))
        .unwrap_err();

    assert_eq!(error, "kinetic evolution produced a non-finite state");
}

#[test]
fn solver_fails_closed_on_negative_evolved_runaway_density() {
    let (mut operator, _, _) = fixture();
    zero_physics(&mut operator);
    operator
        .coefficients
        .total_density_external_source_m3_s
        .fill(-100.0);
    let initial = vec![1.0; operator.grid.cell_count()];
    let density = vec![1.0; operator.grid.nr()];
    let solver = RunawayKineticSolver::new(operator, 0.1, 0.0).unwrap();

    let error = solver
        .solve(&initial, &[0.0, 0.1], Some(&density))
        .unwrap_err();

    assert_eq!(error, "runaway-density evolution produced an invalid state");
}
