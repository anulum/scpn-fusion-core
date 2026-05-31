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

## Compact-EM grid convergence evidence

- Schema: `gk-electromagnetic-grid-convergence.v1`
- Status: `accepted_local_compact_em_grid_convergence`
- Grid convergence ready: `True`
- Max relative total-energy drift: `5.494182e-03`
- Relative energy tolerance: `5.000000e-01`

| Case | Grid | Field-energy closure | Compact closure | Relative total-energy drift |
|---|---|:---:|:---:|---:|
| compact_em_4x4x8 | `4x4x8x5x4` | `True` | `True` | 3.881947e-03 |
| compact_em_6x6x10 | `6x6x10x5x4` | `True` | `True` | 4.761628e-03 |
| compact_em_8x8x12 | `8x8x12x5x4` | `True` | `True` | 5.494182e-03 |

## Omitted physics

- Faraday induction equation for evolving B
- displacement-current Ampere-Maxwell evolution
- self-consistent inductive parallel electric field evolution
- external same-deck electromagnetic GENE/CGYRO/GS2 output parity

## Maxwell evolution contract

- Native field-evolution mode: `compact_algebraic_Apar_Bpar_closure`
- Full Vlasov-Maxwell parity ready: `False`
- Blocking equation ids: `faraday_induction`, `ampere_maxwell_displacement_current`, `inductive_parallel_electric_field`

| Equation id | Implemented | Compact closure | Native status |
|---|:---:|:---:|---|
| faraday_induction | `False` | `False` | missing_time_evolved_magnetic_field |
| ampere_maxwell_displacement_current | `False` | `False` | missing_displacement_current_evolution |
| inductive_parallel_electric_field | `False` | `False` | missing_self_consistent_inductive_parallel_e_field |
| compact_parallel_ampere_closure | `True` | `True` | implemented_as_algebraic_closure_not_maxwell_evolution |
| compact_perpendicular_pressure_balance_closure | `True` | `True` | implemented_as_algebraic_closure_not_maxwell_evolution |

## Missing full-fidelity requirements

- full Faraday/displacement-current Maxwell field evolution
- same-deck electromagnetic GENE/CGYRO/GS2 output artifacts
- native electromagnetic phi/A_parallel/B_parallel same-case parity thresholds
- same-deck external electromagnetic grid-convergence evidence
