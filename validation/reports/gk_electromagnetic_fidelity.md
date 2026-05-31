# GK Electromagnetic Fidelity Gate

Separate electrostatic and electromagnetic nonlinear GK gate. Compact A_parallel/B_parallel diagnostics are local readiness evidence only, not full Vlasov-Maxwell parity.

- Schema: `gk-electromagnetic-fidelity.v1`
- Status: `blocked_missing_full_vlasov_maxwell_field_solve`
- Compact EM contract ready: `True`
- External EM parity comparison ready: `False`

## Gate rows

| Gate | EM enabled | Time history | Compact closure | Full Vlasov-Maxwell parity |
|---|:---:|:---:|:---:|:---:|
| electrostatic_gate | `False` | `True` | `False` | `False` |
| electromagnetic_gate | `True` | `True` | `True` | `False` |

## Omitted physics

- Faraday induction equation for evolving B
- displacement-current Ampere-Maxwell evolution
- self-consistent inductive parallel electric field evolution
- external same-deck electromagnetic GENE/CGYRO/GS2 output parity

## Missing full-fidelity requirements

- full Faraday/displacement-current Maxwell field evolution
- same-deck electromagnetic GENE/CGYRO/GS2 output artifacts
- native electromagnetic phi/A_parallel/B_parallel same-case parity thresholds
- grid-convergence evidence for electromagnetic field-energy histories
