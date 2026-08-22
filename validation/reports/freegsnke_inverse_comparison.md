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
| SCPN vs FreeGSNKE vacuum-psi max error inside limiter | 1.11022e-16 Wb |
| SCPN current-gradient relative L2 error | 4.60171e-13 |
| Sampled pprime/FFprime source relative L2 error | 3.48738e-08 |
| Gauge-shifted source relative L2 error | 3.68492e-14 |
| Selected COCOS-3 source adapter | identity |

## Gates

- PASS — `active_current_regression`
- PASS — `current_limits`
- PASS — `passive_currents_zero`
- PASS — `scpn_coil_gradient`
- PASS — `scpn_freegsnke_vacuum_parity_inside_limiter`
- PASS — `scpn_freegsnke_profile_source_bridge`
- PASS — `topology_regression`
- PASS — `total_psi_regression`

## Claim boundary

The FreeGSNKE profile-source translation into SCPN sampled `pprime`/`FFprime` is frozen for this case, including exact LCFS support, gauge invariance, and identity COCOS-3 flux-per-radian scaling. A self-consistent SCPN solve has not yet demonstrated full total-psi cross-solver parity, so that broader claim remains explicitly unadmitted.

Payload SHA-256: `a39d6522a2ce9362c94ff80455753be155c8c4828af5fa5fe90671e77fdd0667`
