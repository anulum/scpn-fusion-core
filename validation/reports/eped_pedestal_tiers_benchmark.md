<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# EPED Pedestal Tier Benchmark (documented divergence)

- Status: `documented_divergence_no_parity_claim`
- Reference: digitised EPED1 DIII-D Ip scan (https://fusion.gat.com/pubs-ext/APS11/Snydervgs.pdf)
- Assumed geometry: R0=1.67 m, a=0.67 m (not stated on the slide)
- Density scan: [4.0, 6.0, 8.0] ×1e19 m⁻³

| Ip [MA] | EPED1 p_ped [kPa] | measured [kPa] | fast tier best [kPa] | ratio | PB-KBM best [kPa] | PB-KBM verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 4.5 | 4.0 | 2.6 | 0.57 | 2.6 | `collapsed_salpha_first_stability` |
| 1.0 | 13.0 | 12.5 | 2.6 | 0.20 | 2.6 | `collapsed_salpha_first_stability` |
| 1.5 | 12.0 | 12.5 | 2.6 | 0.21 | 2.6 | `collapsed_salpha_first_stability` |

Named blockers for any quantitative EPED parity claim:

- `shaped_geometry_miller_ballooning_required_for_second_stability_access`
- `n_ped_and_geometry_not_published_on_reference_slide`
