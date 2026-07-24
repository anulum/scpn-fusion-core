# IDA fixed-reference current-source ablation

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `89ce2d234ec978e1401fbf18b73cd4712bd19fd9eb33b3cecea0da6e0c539b29`
- Facility/control/PCS/safety/scientific admission: `false`

## Measured isolation

| Source path | current TV distance | centroid ΔR (m) | centroid ΔZ (m) |
|---|---:|---:|---:|
| Fixed reference + exact samples | 0.00120196228 | 0.000611297882 | 0.000152757986 |
| Fixed reference + compact B-spline | 0.00120196228 | 0.000611297882 | 0.000152757986 |
| Self-consistent candidate | 0.251331578 | -0.0418157144 | 0.0584499029 |
| Candidate ψ + reference boundary | 0.220640756 | -0.0336279967 | 0.058539036 |
| Candidate ψ + reference axis and boundary | 0.220619787 | -0.0336129488 | 0.0585396365 |

- Maximum profile-fit relative L2 error: `5.80227936e-16`
- Self-consistent / exact-fixed TV ratio: `209.101053`
- Anchor routing: `candidate_flux_geometry_primary_boundary_anchor_secondary_axis_anchor_excluded`
- Next ratcheting target: `self_consistent_equilibrium_geometry_and_flux_normalisation`

This routes engineering work only; it is not a physical-validation or admission result.
