# DIII-D / IDA fixed-physical coil-vacuum response

- Schema: `scpn-fusion.ida-coil-vacuum-fixed-physical-response.v1`
- Payload: `739120849d5c887effae409e869e0cd17215a103052b634d8a44b1d5e16c0664`
- Upstream CVGC1: `7f04e4cb4217d920f19eaecfbd7738b86aa9db93fa37894df3b4f68c7f211193`
- Routing: `fixed_physical_source_and_vacuum_numerics_resolved`
- Production solver physics changed: `false`
- Scientific, facility, control, safety, PCS and held-out claims: `false`

| Grid | fixed-source L2 fraction | fixed current error | source-free response / total | response closure [Wb] |
|---:|---:|---:|---:|---:|
| 33 | 0.9999806 | 0.0043221762 | 0.0029745409 | 2.5365619e-12 |
| 65 | 0.99999953 | 0.00058242855 | 0.00034447275 | 1.6668304e-15 |
| 129 | 0.99999999 | 0.00014385938 | 9.7647711e-05 | 1.8034245e-14 |
| 257 | 1 | 3.5501998e-05 | 2.4710163e-05 | 1.8962306e-15 |

## Observed orders

- `33_65_129` forcing `2.0120398`, response `3.3969232`
- `65_129_257` forcing `2.0025655`, response `1.7994469`

## Admission boundary

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

CVGC1 remains immutable; its failed relative-response gate is not rewritten.
