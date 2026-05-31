# Runaway DREAM-Style Fluid Contract Benchmark

This benchmark validates scalar runaway-density balance contracts compatible with DREAM fluid runs.
It does not claim parity with DREAM's kinetic momentum-space distribution solver.

## Timing

- Repeats: 25
- Median balance wall time: 9.635929e-06 s
- Minimum balance wall time: 8.539064e-06 s
- Maximum balance wall time: 2.303103e-05 s

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

Overall: PASS
