# IDA fixed-reference operator residual

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `a01392e1b0280118aec51b44cf835466b6b1dc1fb3612cafec953fdc4d7a5814`
- Facility/control/PCS/safety/scientific admission: `false`

## Interior decomposition

| Component | relative L2 to reference scale | RMS | Linf |
|---|---:|---:|---:|
| exact_source_convention | 0.00423226961 | 0.0120046801 | 0.0428919674 |
| freegs_fourth_order_baseline | 5.07874553e-13 | 1.44056786e-12 | 1.09050546e-11 |
| second_order_operator | 0.000954149732 | 0.00270641131 | 0.0117831148 |
| vacuum_discretisation | 0.000129209452 | 0.00036649795 | 0.00296060052 |

## Wall decomposition

| Component | relative L2 to reference scale | RMS | Linf |
|---|---:|---:|---:|
| coil_vacuum_convention | 1.11893417e-15 | 1.45406595e-16 | 6.66133815e-16 |
| exact_source_convention | 0.00388056843 | 0.00050428368 | 0.00117647019 |
| plasma_response_quadrature | 2.22369055e-07 | 2.88970772e-08 | 1.06232469e-07 |

## Region guard

- Reference plasma support: `3271` interior points
- Coil filaments inside the rectangular domain: `216` / `216`
- Vacuum-operator L2 outside reference plasma support: `1`

- Dominant interior component: `exact_source_convention`
- Next interior target: `current_support_and_source_convention`
- Dominant wall component: `exact_source_convention`
- Next wall target: `wall_source_support_and_quadrature`

This is a fixed-reference engineering decomposition, not a validation or admission result.
