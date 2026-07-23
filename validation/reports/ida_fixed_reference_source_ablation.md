# IDA fixed-reference current-source ablation

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `a0dd4593e0fa45bc7c36080eff172639cf002f2899e9c253412277587523f8a4`
- Facility/control/PCS/safety/scientific admission: `false`

## Measured isolation

| Source path | current TV distance | centroid ΔR (m) | centroid ΔZ (m) |
|---|---:|---:|---:|
| Fixed reference + exact samples | 0.00407197811 | -0.00166066968 | -0.00138463247 |
| Fixed reference + compact B-spline | 0.00407197811 | -0.00166066968 | -0.00138463247 |
| Self-consistent candidate | 0.251353196 | -0.0420389348 | 0.0584266563 |
| Candidate ψ + reference boundary | 0.220638753 | -0.0338318405 | 0.0585186926 |
| Candidate ψ + reference axis and boundary | 0.220629204 | -0.0338255492 | 0.0585189792 |

- Maximum profile-fit relative L2 error: `8.64220477e-16`
- Self-consistent / exact-fixed TV ratio: `61.7275411`
- Anchor routing: `candidate_flux_geometry_primary_boundary_anchor_secondary_axis_anchor_excluded`
- Next ratcheting target: `self_consistent_equilibrium_geometry_and_flux_normalisation`

This routes engineering work only; it is not a physical-validation or admission result.
