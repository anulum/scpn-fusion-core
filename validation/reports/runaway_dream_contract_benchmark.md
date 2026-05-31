# Runaway DREAM-Style Contract Benchmark

This benchmark validates scalar runaway-density balance contracts compatible with DREAM fluid runs plus a native DREAM-style kinetic artifact contract.
It does not claim parity with DREAM's kinetic momentum-space distribution solver.

## Timing

- Repeats: 25
- Median balance wall time: 9.215903e-06 s
- Minimum balance wall time: 8.932082e-06 s
- Maximum balance wall time: 1.299602e-05 s

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

## Native kinetic artifact contract

- Schema: `dream-kinetic-artifact.v1`
- SHA-256: `dbadcede66e42484ee48f5087c5608c1f631abaeb564d628d821a736aefe25e6`
- Parity status: `native_contract_only_not_dream_parity`
- Same-case DREAM comparison ready: `False`
- Contract validation passed: `True`
- Coordinate lengths: `{"momentum_mec": 48, "pitch_cosine": 5, "radius_m": 4, "time_s": 4}`
- Observable shapes: `{"avalanche_growth_rate_t": [4, 4], "bremsstrahlung_loss_power_t": [4, 4], "f_p_xi_t": [4, 4, 48, 5], "partial_screening_drag_t": [4, 4], "runaway_current_t": [4, 4], "synchrotron_loss_power_t": [4, 4]}`
- Required DREAM observables: `f_p_xi_t, runaway_current_t, avalanche_growth_rate_t, synchrotron_loss_power_t, partial_screening_drag_t, bremsstrahlung_loss_power_t`

Overall: PASS
