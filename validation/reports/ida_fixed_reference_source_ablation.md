# IDA fixed-reference current-source ablation

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `e84c92dacb0f42b812a82d35796e6a94d8c8a15f72d431579ef71fbc43d941f4`
- Facility/control/PCS/safety/scientific admission: `false`

## Measured isolation

| Source path | current TV distance | centroid ΔR (m) | centroid ΔZ (m) |
|---|---:|---:|---:|
| Fixed reference + exact samples | 0.00120254894 | 0.000611573235 | 0.000152949977 |
| Fixed reference + compact B-spline | 0.00120254894 | 0.000611573235 | 0.000152949977 |
| Self-consistent candidate | 0.251326266 | -0.0418928001 | 0.0584236274 |
| Candidate ψ + reference boundary | 0.220637959 | -0.0337057636 | 0.0585128395 |
| Candidate ψ + reference axis and boundary | 0.220616982 | -0.0336907041 | 0.0585134399 |

- Maximum profile-fit relative L2 error: `8.64220477e-16`
- Self-consistent / exact-fixed TV ratio: `208.994625`
- Anchor routing: `candidate_flux_geometry_primary_boundary_anchor_secondary_axis_anchor_excluded`
- Next ratcheting target: `self_consistent_equilibrium_geometry_and_flux_normalisation`

This routes engineering work only; it is not a physical-validation or admission result.
