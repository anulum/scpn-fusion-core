# IDA fixed-point stability diagnostic

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `54cfbc43d5337fce5c1667888e21a51c9936189def728b5dc062b30649e85a41`
- Facility/control/PCS/safety/scientific admission: `false`

## Stationary-map forcing decomposition

| Component | relative L2 to terminal error | projection on terminal error | cosine |
|---|---:|---:|---:|
| native_operator_residual | 1.15844769 | 1.08527965 | 0.936839588 |
| boundary_anchor | 2.1355005e-07 | 4.25067392e-08 | 0.199048135 |
| source_mechanism | 0.00318965968 | 0.00190569536 | 0.597460403 |
| **total** | 1.16063159 | 1.08718539 | 0.936718769 |

## Local gains and frozen-map trajectory

- Terminal-error JVP gain: `0.276545852`
- Source-mechanism JVP gain: `0.866208313`
- Raw Picard map moves toward candidate: `true`
- Next ratcheting target: `native_operator_residual_reference_stationarity`

| Step | distance to reference / terminal | distance to candidate / terminal | projection |
|---:|---:|---:|---:|
| 0 | 0 | 1 | 0 |
| 1 | 1.16063159 | 0.415565757 | 1.08718539 |
| 2 | 1.09894646 | 0.274075724 | 1.0662829 |
| 3 | 1.08964699 | 0.261325465 | 1.05951978 |
| 4 | 1.08663013 | 0.268343556 | 1.05437838 |

This is a single-case engineering diagnostic, not experimental validation.
