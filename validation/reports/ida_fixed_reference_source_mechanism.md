# IDA fixed-reference source-mechanism decomposition

- Status: `diagnostic_complete_claims_blocked`
- Payload SHA-256: `8cb5d9d72b7081a8bb5ddf1c05dfc66850178fde11d68c91455c8cb97c6c1914`
- Facility/control/PCS/safety/scientific admission: `false`

## Fixed-reference current fields

| Construction | rectangular current (A) | relative L2 | TV | outside support |
|---|---:|---:|---:|---:|
| freegs_hard_romberg | -1533631.79141 | 0 | 0 | 0 |
| freegs_hard_rectangular_normalised | -1533632 | 1.36008949e-07 | 2.17822397e-17 | 0 |
| fusion_smooth_unscaled | -1534085.1225 | 0.00212341639 | 0.00120254936 | 6.19023978e-07 |
| fusion_smooth_rectangular_normalised | -1533632 | 0.00212342443 | 0.00120254936 | 6.19023978e-07 |

## Sequential mechanism vectors

| Component | current relative L2 | interior-source relative L2 | wall relative L2 |
|---|---:|---:|---:|
| hard_rectangular_normalisation | 1.36008949e-07 | 1.36008949e-07 | 1.36008949e-07 |
| smooth_cutoff | 0.00212340722 | 0.00225026568 | 0.00103448758 |
| smooth_ip_normalisation | 0.000295412806 | 0.000295519894 | 0.000295671832 |

- Dominant current component: `smooth_cutoff`
- Dominant interior-source component: `smooth_cutoff`
- Dominant wall-response component: `smooth_cutoff`
- Next ratcheting target: `soft_axis_connected_support_topology_and_unclipped_lcfs_distance`

This is a fixed-reference engineering decomposition, not a validation or admission result.
