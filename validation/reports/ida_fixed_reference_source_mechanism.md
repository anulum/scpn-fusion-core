# IDA fixed-reference source-mechanism decomposition

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `46865342c0daa5d75cae7403fc4e927694eea2b3e200e90c67ef5f5262a38fe0`
- Facility/control/PCS/safety/scientific admission: `false`

## Fixed-reference current fields

| Construction | rectangular current (A) | relative L2 | TV | outside support |
|---|---:|---:|---:|---:|
| freegs_hard_romberg | -1533631.7912 | 0 | 0 | 0 |
| freegs_hard_rectangular_normalised | -1533632 | 1.36148024e-07 | 2.58654552e-17 | 0 |
| fusion_smooth_unscaled | -1534084.62762 | 0.00212234359 | 0.00120196228 | 6.15988685e-07 |
| fusion_smooth_rectangular_normalised | -1533632 | 0.00212234695 | 0.00120196228 | 6.15988685e-07 |

## Sequential mechanism vectors

| Component | current relative L2 | interior-source relative L2 | wall relative L2 |
|---|---:|---:|---:|
| hard_rectangular_normalisation | 1.36148024e-07 | 1.36148024e-07 | 1.36148024e-07 |
| smooth_cutoff | 0.00212233442 | 0.00224911298 | 0.00103385007 |
| smooth_ip_normalisation | 0.000295090227 | 0.000295197151 | 0.000295348822 |

- Dominant current component: `smooth_cutoff`
- Dominant interior-source component: `smooth_cutoff`
- Dominant wall-response component: `smooth_cutoff`
- Next ratcheting target: `self_consistent_equilibrium_geometry_and_boundary_response`

This is a fixed-reference engineering decomposition, not a validation or admission result.
