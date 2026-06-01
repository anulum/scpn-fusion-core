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
- SHA-256: `4a250651cf21a5c36b3fe6221e8dc3761e2c1562f5da4810405752e13a1ac8d8`
- Contract validation passed: `True`
- Same-case Aurora/STRAHL comparison ready: `False`
- Observable shapes: `{"charge_state_density_r_t": [3, 80, 4], "ionisation_source_matrix": [80, 4], "line_radiation_power_t": [3], "line_radiation_power_t_r_z": [3, 80, 4], "recombination_sink_matrix": [80, 4], "source_sink_matrix_t_r_z_z": [3, 80, 4, 4], "total_impurity_density_r_t": [3, 80], "total_impurity_inventory_t": [3]}`
- Required Aurora/STRAHL observables: `charge_state_density_r_t, total_impurity_density_r_t, line_radiation_power_t, line_radiation_power_t_r_z, source_sink_matrix_t_r_z_z, total_impurity_inventory_t`

## Native impurity transport operator evidence

- Schema: `native-impurity-transport-operator-evidence.v1`
- Status: `blocked_native_charge_state_contract_not_full_aurora_strahl_transport_operator`
- Native artifact ready: `True`
- Charge-state radial transport operator ready: `False`
- Aurora/STRAHL same-case thresholds ready: `False`
- Density axes: `time_s, radius_m, charge_state`
- Density shape: `[3, 80, 4]`
- Source-sink shape: `[3, 80, 4, 4]`
- Line-radiation shape: `[3, 80, 4]`
- Operator terms present: `{"aurora_strahl_collisional_operator_parity": false, "charge_state_resolved_radial_transport": false, "charge_state_source_sink_matrix": true, "edge_source_particle_conservation": true, "external_adas_transport_coefficients": false, "line_radiation_power": true, "neoclassical_pinch": true, "same_case_aurora_strahl_transport_output": false, "total_impurity_inventory_closure": true, "trace_radial_transport": true}`
- Observable finiteness: `{"charge_state_density_r_t": true, "line_radiation_power_t": true, "line_radiation_power_t_r_z": true, "source_sink_matrix_t_r_z_z": true, "total_impurity_density_r_t": true, "total_impurity_inventory_t": true}`

## Native source/sink budget evidence

- Schema: `native-impurity-source-sink-budget-evidence.v1`
- Status: `native_artifact_source_sink_budget_only_not_aurora_strahl_operator_parity`
- Budget terms: `source_sink_matrix_t_r_z_z, ionisation_source_matrix, recombination_sink_matrix, line_radiation_power_t_r_z, total_impurity_inventory_t`
- Time count: `3`
- Radius count: `80`
- Charge-state count: `4`
- All budget terms finite: `True`
- Ionisation/recombination non-negative: `True`
- Source/sink transfer conservative: `True`
- Line radiation non-negative: `True`
- Aurora/STRAHL same-case budget ready: `False`
- Blocking requirements: `public Aurora or STRAHL radial transport output; charge-state-resolved radial transport operator on evolved density; external ADAS coefficient ingestion for transport parity; same-case line-radiation output from Aurora or STRAHL; same-case ionisation/recombination source-sink matrix output; native same-case solver-output comparison; distribution, radiation, and inventory threshold comparison against Aurora/STRAHL`

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
- native_impurity_transport_evidence_fail_closed: PASS
- native_source_sink_budget_evidence_fail_closed: PASS

Overall: PASS
