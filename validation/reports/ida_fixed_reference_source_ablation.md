# IDA fixed-reference current-source ablation

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `893938b7dd74ba466004c5d3e042fa2281fc64ed4ca26a3e51d724c878072170`
- Facility/control/PCS/safety/scientific admission: `false`

## Measured isolation

| Source path | current TV distance | centroid ΔR (m) | centroid ΔZ (m) |
|---|---:|---:|---:|
| Fixed reference + exact samples | 0.00407335236 | -0.00166188166 | -0.00138317702 |
| Fixed reference + compact B-spline | 0.00407335236 | -0.00166188166 | -0.00138317702 |
| Self-consistent candidate | 0.251353197 | -0.0420389347 | 0.0584266588 |

- Maximum profile-fit relative L2 error: `5.80227936e-16`
- Self-consistent / exact-fixed TV ratio: `61.7067159`
- Next ratcheting target: `self_consistent_equilibrium_geometry_and_flux_normalisation`

This routes engineering work only; it is not a physical-validation or admission result.
