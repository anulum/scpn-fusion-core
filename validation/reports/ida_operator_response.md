# IDA operator-response decomposition

- Payload: `56eddd06af69d4726433aaed1aff43bb1f92af4def1a4efecbe5a7ab81de911e`
- Generated: `2026-07-24T15:50:00Z`
- Status: `diagnostic_complete_claims_blocked`
- Dominant response: `coil_vacuum_discretisation`
- Next ratchet: `coil_vacuum_grid_convergence`
- Solver physics changed: `false`

## Response-weighted components

| Component | Relative L2 to terminal error | Projection | Cosine |
|---|---:|---:|---:|
| `freegs_fourth_order_baseline` | 6.74763738523e-14 | -3.49733130365e-14 | -0.51830457151 |
| `native_second_order_stencil` | 0.000576692928129 | 2.65476724565e-05 | 0.0460343298168 |
| `coil_vacuum_discretisation` | 1.15825534094 | 1.08525310589 | 0.93697224397 |
| `exact_source_convention` | 0.00188265795294 | 0.000564395460896 | 0.299786511944 |

## Closure

- `exact_source_forcing_max_abs`: `0`
- `exact_source_response_max_abs_wb`: `5.51225731726e-14`
- `fixed_point_native_operator_max_abs_wb`: `8.780476346e-13`
- `native_operator_forcing_max_abs`: `0`
- `native_operator_response_max_abs_wb`: `2.12885264972e-14`

## Claim boundary

- control admission: `false`
- facility validation: `false`
- held out validation: `false`
- isolated latency admission: `false`
- pcs deployment: `false`
- safety admission: `false`
- scientific validation: `false`
