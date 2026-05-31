# GK Electromagnetic Fidelity Gate

Separate electrostatic and electromagnetic nonlinear GK gate. Compact A_parallel/B_parallel diagnostics are local readiness evidence only, not full Vlasov-Maxwell parity.

- Schema: `gk-electromagnetic-fidelity.v1`
- Status: `blocked_missing_external_em_parity_outputs`
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

## Local Maxwell evolution evidence

- Schema: `gk-maxwell-evolution.v1`
- Status: `accepted_local_source_free_maxwell_evolution`
- Faraday induction supported: `True`
- Ampere-Maxwell displacement current supported: `True`
- Inductive parallel electric field supported: `True`
- Self-consistent kinetic current supported: `False`
- Max relative total-field-energy drift: `5.090959e-16`
- Max Faraday residual: `0.000000e+00`
- Max Ampere-Maxwell residual: `0.000000e+00`
- Max inductive parallel electric-field residual: `0.000000e+00`

## Omitted physics

- self-consistent kinetic current coupling in the nonlinear 5D Vlasov-Maxwell loop
- external same-deck electromagnetic GENE/CGYRO/GS2 output parity

## Maxwell evolution contract

- Native field-evolution mode: `local_spectral_maxwell_evolution`
- Full Vlasov-Maxwell parity ready: `False`
- Blocking equation ids: `self_consistent_kinetic_current_coupling`, `same_deck_external_em_parity`

| Equation id | Implemented | Compact closure | Native status |
|---|:---:|:---:|---|
| faraday_induction | `True` | `False` | implemented_as_local_source_free_spectral_field_evolution |
| ampere_maxwell_displacement_current | `True` | `False` | implemented_as_local_source_free_spectral_field_evolution |
| inductive_parallel_electric_field | `True` | `False` | implemented_as_local_source_free_spectral_field_evolution |
| self_consistent_kinetic_current_coupling | `False` | `False` | missing_self_consistent_5d_kinetic_current_coupling |
| same_deck_external_em_parity | `False` | `False` | missing_external_same_deck_em_outputs_and_thresholds |
| compact_parallel_ampere_closure | `True` | `True` | implemented_as_algebraic_closure_not_maxwell_evolution |
| compact_perpendicular_pressure_balance_closure | `True` | `True` | implemented_as_algebraic_closure_not_maxwell_evolution |

## Missing full-fidelity requirements

- self-consistent kinetic current coupling in the nonlinear 5D Vlasov-Maxwell loop
- same-deck electromagnetic GENE/CGYRO/GS2 output artifacts
- native electromagnetic phi/A_parallel/B_parallel same-case parity thresholds
- same-deck external electromagnetic grid-convergence evidence
