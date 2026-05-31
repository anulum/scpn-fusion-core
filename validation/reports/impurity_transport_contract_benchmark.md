# Impurity Transport Contract Benchmark

This benchmark validates native trace impurity transport contracts.
It does not claim Aurora/STRAHL/JINTRAC collisional-operator parity.

## Metrics

- Actual particles: 9.790648e+17
- Expected particles: 9.790648e+17
- Relative conservation error: 1.307370e-16 (threshold 2.00e-02)
- Midradius pinch: -8.683411e+00 m/s
- Low radiated power: 3.921785e+00 MW
- High radiated power: 3.921785e+01 MW
- Edge density: 2.093337e+16 m^-3
- Charge-state count: 4
- Charge-state inventory error: 0.000000e+00 (threshold 1.00e-12)

## Aurora/STRAHL-style artifact contract

- Schema: `aurora-strahl-charge-state-artifact.v1`
- Coordinates: time_s, radius_m, charge_state
- Observables: charge_state_density_r_t, total_impurity_density_r_t, line_radiation_power_t, line_radiation_power_t_r_z, source_sink_matrix_t_r_z_z, total_impurity_inventory_t, ionisation_source_matrix, recombination_sink_matrix
- Parity status: `artifact_contract_only_not_public_aurora_strahl_parity`
- SHA-256: `e5804b23ced11d5a36db144ac8f298a7e8af0feebdb901e92a43c5ed09d890d1`
- Contract validation passed: `True`
- Same-case Aurora/STRAHL comparison ready: `False`
- Observable shapes: `{"charge_state_density_r_t": [3, 80, 4], "ionisation_source_matrix": [80, 4], "line_radiation_power_t": [3], "line_radiation_power_t_r_z": [3, 80, 4], "recombination_sink_matrix": [80, 4], "source_sink_matrix_t_r_z_z": [3, 80, 4, 4], "total_impurity_density_r_t": [3, 80], "total_impurity_inventory_t": [3]}`
- Required Aurora/STRAHL observables: `charge_state_density_r_t, total_impurity_density_r_t, line_radiation_power_t, line_radiation_power_t_r_z, source_sink_matrix_t_r_z_z, total_impurity_inventory_t`

## Invariants

- positivity: PASS
- edge_source_conservation: PASS
- inward_pinch_midradius: PASS
- radiation_monotonicity: PASS
- charge_state_artifact_contract: PASS
- charge_state_density_closure: PASS
- charge_state_particle_conservation: PASS
- source_sink_matrix_conservative: PASS
- line_radiation_power_finite: PASS

Overall: PASS
