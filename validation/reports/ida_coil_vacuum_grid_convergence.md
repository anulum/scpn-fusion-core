# IDA coil-vacuum grid convergence

- Payload: `7f04e4cb4217d920f19eaecfbd7738b86aa9db93fa37894df3b4f68c7f211193`
- Generated: `2026-07-26T00:24:30Z`
- Status: `diagnostic_complete_claims_blocked`
- Routing: `mixed_source_and_vacuum_error`
- Solver physics changed: `false`

## Grid ladder

| Grid | Primary forcing L2 fraction | Weighted current error | Response closure [Wb] |
|---:|---:|---:|---:|
| 33 | 0.999975255644 | 0.0036768438623 | 2.43682053601e-12 |
| 65 | 0.999976590683 | 0.0012656610456 | 8.49404300693e-15 |
| 129 | 0.999951715191 | 0.00267098479801 | 6.17501926328e-14 |
| 257 | 0.999805924808 | 0.00449893038907 | 1.568000829e-13 |

## Source-free observed order

- `33_65_129`: `2.01203980513`
- `65_129_257`: `2.0025654602`

## Gates

- current recovery fine: `true`
- current recovery non increasing: `false`
- finest response stability: `false`
- source free observed order: `true`
- source localisation: `true`

## Claim boundary

- collaborator validation: `false`
- control admission: `false`
- experimental diiid validation: `false`
- facility validation: `false`
- held out validation: `false`
- isolated latency admission: `false`
- pcs deployment: `false`
- production physics admission: `false`
- safety admission: `false`
- scientific validation: `false`
