# IDA fixed-point stability diagnostic

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `48c200a036ab6564ec6dccd732bb8bc20dff1beb3caf17d2e1dd4e4c1d955d14`
- Facility/control/PCS/safety/scientific admission: `false`

## Stationary-map forcing decomposition

| Component | relative L2 to terminal error | projection on terminal error | cosine |
|---|---:|---:|---:|
| native_operator_residual | 1.15816953 | 1.08506096 | 0.936875761 |
| boundary_anchor | 2.22683606e-07 | 5.689232e-08 | 0.255484995 |
| source_mechanism | 0.00318753013 | 0.0019052417 | 0.597717237 |
| **total** | 1.16035222 | 1.08696626 | 0.936755441 |

## Local gains and frozen-map trajectory

- Terminal-error JVP gain: `0.2763551`
- Source-mechanism JVP gain: `0.866282461`
- Raw Picard map moves toward candidate: `true`
- Next ratcheting target: `native_operator_residual_reference_stationarity`

| Step | distance to reference / terminal | distance to candidate / terminal | projection |
|---:|---:|---:|---:|
| 0 | 0 | 1 | 0 |
| 1 | 1.16035222 | 0.415312851 | 1.08696626 |
| 2 | 1.09872277 | 0.273887086 | 1.0660888 |
| 3 | 1.08945545 | 0.261185322 | 1.0593477 |
| 4 | 1.08646591 | 0.268244427 | 1.05422655 |

This is a single-case engineering diagnostic, not experimental validation.
