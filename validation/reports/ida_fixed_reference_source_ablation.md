# IDA fixed-reference current-source ablation

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `cd6905267248745248e63b630c08a55943e512aa0f53b1a5f85bdb367bf5409a`
- Facility/control/PCS/safety/scientific admission: `false`

## Measured isolation

| Source path | current TV distance | centroid ΔR (m) | centroid ΔZ (m) |
|---|---:|---:|---:|
| Fixed reference + exact samples | 0.00407335236 | -0.00166188166 | -0.00138317702 |
| Fixed reference + compact B-spline | 0.00407335236 | -0.00166188166 | -0.00138317702 |
| Self-consistent candidate | 0.251353197 | -0.0420389347 | 0.0584266588 |
| Candidate ψ + reference boundary | 0.220641525 | -0.0337540614 | 0.0585448872 |
| Candidate ψ + reference axis and boundary | 0.220631967 | -0.0337477782 | 0.0585451733 |

- Maximum profile-fit relative L2 error: `5.80227936e-16`
- Self-consistent / exact-fixed TV ratio: `61.7067159`
- Anchor routing: `candidate_flux_geometry_primary_boundary_anchor_secondary_axis_anchor_excluded`
- Next ratcheting target: `self_consistent_equilibrium_geometry_and_flux_normalisation`

This routes engineering work only; it is not a physical-validation or admission result.
