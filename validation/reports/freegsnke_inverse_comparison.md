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

## Gates

- PASS — `active_current_regression`
- PASS — `current_limits`
- PASS — `passive_currents_zero`
- PASS — `scpn_coil_gradient`
- PASS — `scpn_freegsnke_vacuum_parity_inside_limiter`
- PASS — `topology_regression`
- PASS — `total_psi_regression`

## Claim boundary

The full FreeGSNKE profile-source normalisation/gauge translation into SCPN sampled `pprime`/`FFprime` is not frozen, so full total-psi cross-solver parity remains explicitly unadmitted.

Payload SHA-256: `1df12993cf2022815633e4de19cbcae10ec980837ab233b65d7aeb7212ebbc23`
