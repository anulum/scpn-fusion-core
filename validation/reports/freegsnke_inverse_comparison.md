# FreeGSNKE inverse-equilibrium comparison

Status: **PASS**

This is bounded research evidence. It is not facility, PCS, safety, or control admission.

## Results

| Check | Result |
|---|---:|
| FreeGSNKE active-current max error | 0.00152679 A |
| FreeGSNKE total-psi max error | 6.31419e-08 Wb |
| Magnetic-axis error | 6.41978e-08 m |
| Worst primary X-point error | 1.65949e-08 m |
| SCPN vs FreeGSNKE vacuum-psi max error inside limiter | 1.13798e-15 Wb |
| SCPN current-gradient relative L2 error | 1.37241e-12 |
| Sampled pprime/FFprime source relative L2 error | 3.48738e-08 |
| Gauge-shifted source relative L2 error | 3.68502e-14 |
| Selected COCOS-3 source adapter | identity |
| Production smooth total-psi psi_N RMSE | 0.00354214 |
| Production smooth current-source relative L2 | 0.00747369 |
| Production smooth nonlinear residual | 1.0496e-07 |
| Frozen-topology total-psi psi_N RMSE | 0.000377735 |
| Worst total-psi gradient relative error | 7.847e-05 |

## Gates

- PASS — `active_current_regression`
- PASS — `current_limits`
- PASS — `passive_currents_zero`
- PASS — `scpn_coil_gradient`
- PASS — `scpn_freegsnke_vacuum_parity_inside_limiter`
- PASS — `scpn_freegsnke_profile_source_bridge`
- PASS — `scpn_freegsnke_total_psi_same_case`
- PASS — `topology_regression`
- PASS — `total_psi_regression`

## Claim boundary

The profile-source translation and self-consistent SCPN production-smooth total-psi solve are admitted for this one pinned case, including explicit coil/plasma field decomposition, topology, residual, current-support, and implicit-gradient gates. Shot-disjoint, facility, PCS, safety, control, latency, and real-time claims remain explicitly unadmitted.

Payload SHA-256: `019fdaee778ed9429da9c997262cfb115b0c0dde7db89f536644a366555b6a87`
