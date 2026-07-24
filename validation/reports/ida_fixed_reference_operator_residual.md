# IDA fixed-reference operator residual

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `a7ac591612e6b44b8b77aeaed2373ecec994887d7915dc568df00f8773e6c2b1`
- Facility/control/PCS/safety/scientific admission: `false`

## Interior decomposition

| Component | relative L2 to reference scale | RMS | Linf |
|---|---:|---:|---:|
| exact_source_convention | 0.00220105503 | 0.00624494928 | 0.0203176714 |
| freegs_fourth_order_baseline | 5.2009014e-13 | 1.47562714e-12 | 1.58335567e-11 |
| second_order_operator | 0.000954179891 | 0.00270724945 | 0.0117830828 |
| vacuum_discretisation | 0.000129208603 | 0.000366597454 | 0.00296064065 |

## Wall decomposition

| Component | relative L2 to reference scale | RMS | Linf |
|---|---:|---:|---:|
| coil_vacuum_convention | 1.17618659e-15 | 1.528699e-16 | 8.8817842e-16 |
| exact_source_convention | 0.00148989983 | 0.000193643457 | 0.000511639192 |
| plasma_response_quadrature | 2.29071245e-07 | 2.97725705e-08 | 1.12063633e-07 |

## Region guard

- Reference plasma support: `3269` interior points
- Coil filaments inside the rectangular domain: `216` / `216`
- Vacuum-operator L2 outside reference plasma support: `1`

- Dominant interior component: `exact_source_convention`
- Next interior target: `current_support_and_source_convention`
- Dominant wall component: `exact_source_convention`
- Next wall target: `wall_source_support_and_quadrature`

This is a fixed-reference engineering decomposition, not a validation or admission result.
