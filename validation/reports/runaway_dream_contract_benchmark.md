# Runaway DREAM-Style Contract Benchmark

This benchmark validates scalar runaway-density balance contracts compatible with DREAM fluid runs plus a native DREAM-style kinetic artifact contract.
It does not claim parity with DREAM's kinetic momentum-space distribution solver.

## Timing

- Repeats: 25
- Median balance wall time: 1.915102e-05 s
- Minimum balance wall time: 1.565111e-05 s
- Maximum balance wall time: 2.834306e-05 s

## Cases

| Case | Dreicer source [m^-3 s^-1] | Avalanche source [m^-3 s^-1] | Loss source [m^-3 s^-1] | Total source [m^-3 s^-1] | Runaway fraction | Growth time [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| subcritical_no_avalanche | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 1.000000e-08 | inf |
| supercritical_growth | 2.918729e+07 | 9.297121e+15 | 0.000000e+00 | 9.297121e+15 | 2.000000e-08 | 2.151204e-04 |
| mitigated_loss_accounting | 2.918729e+07 | 9.297121e+15 | 1.000000e+13 | 9.287121e+15 | 2.000000e-08 | 2.153520e-04 |

## Invariants

- subcritical_avalanche_zero: PASS
- supercritical_avalanche_positive: PASS
- loss_accounting_exact: PASS
- density_cap_enforced: PASS
- native_kinetic_artifact_contract: PASS
- native_kinetic_operator_evidence_fail_closed: PASS
- native_source_term_budget_evidence_fail_closed: PASS

## Native kinetic artifact contract

- Schema: `dream-kinetic-artifact.v1`
- SHA-256: `dbadcede66e42484ee48f5087c5608c1f631abaeb564d628d821a736aefe25e6`
- Parity status: `native_contract_only_not_dream_parity`
- Same-case DREAM comparison ready: `False`
- Contract validation passed: `True`
- Coordinate lengths: `{"momentum_mec": 48, "pitch_cosine": 5, "radius_m": 4, "time_s": 4}`
- Observable shapes: `{"avalanche_growth_rate_t": [4, 4], "bremsstrahlung_loss_power_t": [4, 4], "f_p_xi_t": [4, 4, 48, 5], "partial_screening_drag_t": [4, 4], "runaway_current_t": [4, 4], "synchrotron_loss_power_t": [4, 4]}`
- Required DREAM observables: `f_p_xi_t, runaway_current_t, avalanche_growth_rate_t, synchrotron_loss_power_t, partial_screening_drag_t, bremsstrahlung_loss_power_t`

## Native kinetic operator evidence

- Schema: `native-runaway-kinetic-operator-evidence.v1`
- Status: `blocked_native_projection_artifact_not_full_dream_operator`
- Native artifact ready: `True`
- Full momentum-pitch-radius operator ready: `False`
- DREAM same-case thresholds ready: `False`
- Evolved radius/pitch operator axes: `False`
- Distribution axes: `time_s, radius_m, momentum_mec, pitch_cosine`
- Distribution shape: `[4, 4, 48, 5]`
- Operator terms present: `{"avalanche_growth": true, "bremsstrahlung_radiation_loss_operator": false, "coupled_momentum_pitch_radius_operator": false, "dreicer_source": true, "full_pitch_angle_scattering_operator": false, "full_radial_transport_operator": false, "momentum_advection_drag": true, "momentum_diffusion": true, "partial_screening_dream_operator": false, "synchrotron_radiation_reaction": true}`
- Observable finiteness: `{"avalanche_growth_rate_t": true, "bremsstrahlung_loss_power_t": true, "f_p_xi_t": true, "partial_screening_drag_t": true, "runaway_current_t": true, "synchrotron_loss_power_t": true}`
- Observable non-negativity: `{"avalanche_growth_rate_t": true, "bremsstrahlung_loss_power_t": true, "f_p_xi_t": true, "partial_screening_drag_t": true, "runaway_current_t": true, "synchrotron_loss_power_t": true}`

## Native source-term budget evidence

- Schema: `native-runaway-source-term-budget-evidence.v1`
- Status: `native_artifact_observable_budget_only_not_dream_operator_parity`
- Budget terms: `avalanche_growth_rate_t, synchrotron_loss_power_t, partial_screening_drag_t, bremsstrahlung_loss_power_t`
- Time count: `4`
- Radius count: `4`
- Term finiteness: `{"avalanche_growth_rate_t": true, "bremsstrahlung_loss_power_t": true, "partial_screening_drag_t": true, "synchrotron_loss_power_t": true}`
- Term non-negativity: `{"avalanche_growth_rate_t": true, "bremsstrahlung_loss_power_t": true, "partial_screening_drag_t": true, "synchrotron_loss_power_t": true}`
- All observable budgets finite: `True`
- Non-negative loss/screening channels: `True`
- DREAM same-case budget ready: `False`
- Blocking requirements: `compiled DREAM iface/dreami same-case output; native coupled momentum-pitch-radius Fokker-Planck operator; radial transport operator on evolved radius grid; full pitch-angle scattering operator on evolved pitch grid; DREAM partial-screening operator parity; DREAM bremsstrahlung and synchrotron loss parity; distribution, current, and growth-rate threshold comparison against DREAM`

Overall: PASS
