# IDA fixed-reference source-mechanism decomposition

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `787af5175bc148572f265df6e2aa0e51f7c0585d6d2a55c4dd0ef8cb3081461c`
- Facility/control/PCS/safety/scientific admission: `false`

## Fixed-reference current fields

| Construction | rectangular current (A) | relative L2 | TV | outside support |
|---|---:|---:|---:|---:|
| freegs_hard_romberg | -1533631.79141 | 0 | 0 | 0 |
| freegs_hard_rectangular_normalised | -1533632 | 1.36008949e-07 | 2.17822397e-17 | 0 |
| fusion_smooth_unscaled | -1534085.12136 | 0.00212341639 | 0.00120254894 | 6.18280936e-07 |
| fusion_smooth_rectangular_normalised | -1533632 | 0.00212342437 | 0.00120254894 | 6.18280936e-07 |

## Sequential mechanism vectors

| Component | current relative L2 | interior-source relative L2 | wall relative L2 |
|---|---:|---:|---:|
| hard_rectangular_normalisation | 1.36008949e-07 | 1.36008949e-07 | 1.36008949e-07 |
| smooth_cutoff | 0.00212340722 | 0.00225026568 | 0.00103448731 |
| smooth_ip_normalisation | 0.000295412063 | 0.000295519151 | 0.000295671089 |

- Dominant current component: `smooth_cutoff`
- Dominant interior-source component: `smooth_cutoff`
- Dominant wall-response component: `smooth_cutoff`
- Next ratcheting target: `self_consistent_equilibrium_geometry_and_boundary_response`

This is a fixed-reference engineering decomposition, not a validation or admission result.
