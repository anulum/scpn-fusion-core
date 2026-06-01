# GK Electromagnetic Fidelity Gate

Separate electrostatic and electromagnetic nonlinear GK gate. Compact A_parallel/B_parallel diagnostics are local readiness evidence only, not full Vlasov-Maxwell parity.

- Schema: `gk-electromagnetic-fidelity.v1`
- Status: `blocked_missing_external_em_parity_outputs`
- Compact EM contract ready: `True`
- Electrostatic GK ready: `True`
- Compact EM ready: `True`
- Source-free Maxwell ready: `True`
- Sourced Maxwell ready: `False`
- Full Vlasov-Maxwell ready: `False`
- External EM parity comparison ready: `False`
- External EM solver-family completeness ready: `False`

## Gate rows

| Gate | EM enabled | Time history | Compact closure | Full Vlasov-Maxwell parity |
|---|:---:|:---:|:---:|:---:|
| electrostatic_gate | `False` | `True` | `False` | `False` |
| electromagnetic_gate | `True` | `True` | `True` | `False` |

## Electromagnetic evidence gate matrix

| Surface | Ready | Blockers |
|---|:---:|---|
| `electrostatic_gk_gate_separation` | `True` | - |
| `compact_A_parallel_B_parallel_closure` | `True` | - |
| `source_free_faraday_induction` | `True` | - |
| `source_free_ampere_maxwell_displacement_current` | `True` | - |
| `source_free_inductive_parallel_electric_field` | `True` | - |
| `magnetic_divergence_constraint` | `True` | - |
| `electromagnetic_energy_invariant_diagnostics` | `True` | - |
| `native_em_same_case_thresholds` | `True` | - |
| `sourced_kinetic_current_maxwell_coupling` | `False` | pending_5d_kinetic_current_moment_coupling |
| `external_em_gene_cgyro_gs2_parity` | `False` | missing_same_deck_external_em_gene_cgyro_gs2_outputs |
| `external_em_grid_convergence` | `False` | missing_same_deck_external_em_grid_convergence |

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
- Magnetic divergence constraint supported: `True`
- Max relative total-field-energy drift: `5.090959e-16`
- Max Faraday residual: `0.000000e+00`
- Max Ampere-Maxwell residual: `0.000000e+00`
- Max inductive parallel electric-field residual: `0.000000e+00`
- Max magnetic divergence residual: `2.028048e-16`

## Native EM same-case threshold evidence

- Schema: `gk-native-em-same-case-thresholds.v1`
- Status: `accepted_native_em_same_case_thresholds`
- Benchmark case id: `native_em_replay_4x4x8_seed419`
- Reference kind: `native_deterministic_replay_not_external_parity`
- Same-case thresholds ready: `True`

| Observable | Shape | Max absolute error | Absolute tolerance | Max relative error | Relative tolerance | Pass |
|---|---|---:|---:|---:|---:|:---:|
| electromagnetic_apar_energy | `4` | 0.000000e+00 | 1.000000e-18 | 0.000000e+00 | 1.000000e-15 | `True` |
| electromagnetic_bpar_energy | `4` | 0.000000e+00 | 1.000000e-18 | 0.000000e+00 | 1.000000e-15 | `True` |
| electromagnetic_phi_energy | `4` | 0.000000e+00 | 1.000000e-18 | 0.000000e+00 | 1.000000e-15 | `True` |
| electromagnetic_total_field_energy | `4` | 0.000000e+00 | 1.000000e-18 | 0.000000e+00 | 1.000000e-15 | `True` |

## Omitted physics

- self-consistent kinetic current coupling in the nonlinear 5D Vlasov-Maxwell loop
- external same-deck electromagnetic GENE/CGYRO/GS2 output parity

## Sourced Maxwell contract

- Schema: `gk-sourced-maxwell-contract.v1`
- Status: `blocked_sourced_maxwell_requires_5d_current_moments`
- Current status: `blocked_pending_5d_kinetic_current_moment_coupling`
- Sourced Maxwell ready: `False`

Required inputs:
- `phi(kx, ky, t)`
- `A_parallel(kx, ky, t)`
- `B_parallel(kx, ky, t)`
- `J_parallel(kx, ky, t)`
- `rho_charge(kx, ky, t)`
- `continuity_residual(kx, ky, t)`

Readiness criteria:
- J_parallel(kx, ky, t) derived from the evolved 5D distribution
- charge/current continuity residual history
- sourced Ampere-Maxwell residual history
- sourced Faraday residual history
- sourced electromagnetic energy exchange diagnostic
- same-case native threshold rows for sourced fields

## External EM parity evidence

- Schema: `gk-electromagnetic-external-parity-evidence.v1`
- Status: `blocked_missing_same_deck_external_em_outputs`
- Same-deck group ready: `False`
- Solver-family completeness ready: `False`

| Solver | Reference output | Required observables | Native comparison | Native thresholds |
|---|:---:|:---:|:---:|:---:|
| GENE | `False` | `False` | `False` | `False` |
| CGYRO | `False` | `False` | `False` | `False` |
| GS2 | `False` | `False` | `False` | `False` |

## Maxwell evolution contract

- Native field-evolution mode: `local_spectral_maxwell_evolution`
- Full Vlasov-Maxwell parity ready: `False`
- Blocking equation ids: `self_consistent_kinetic_current_coupling`, `same_deck_external_em_parity`

| Equation id | Implemented | Compact closure | Native status |
|---|:---:|:---:|---|
| faraday_induction | `True` | `False` | implemented_as_local_source_free_spectral_field_evolution |
| ampere_maxwell_displacement_current | `True` | `False` | implemented_as_local_source_free_spectral_field_evolution |
| inductive_parallel_electric_field | `True` | `False` | implemented_as_local_source_free_spectral_field_evolution |
| magnetic_divergence_constraint | `True` | `False` | implemented_as_local_source_free_spectral_field_evolution |
| self_consistent_kinetic_current_coupling | `False` | `False` | missing_self_consistent_5d_kinetic_current_coupling |
| same_deck_external_em_parity | `False` | `False` | missing_external_same_deck_em_outputs_and_thresholds |
| compact_parallel_ampere_closure | `True` | `True` | implemented_as_algebraic_closure_not_maxwell_evolution |
| compact_perpendicular_pressure_balance_closure | `True` | `True` | implemented_as_algebraic_closure_not_maxwell_evolution |

## Missing full-fidelity requirements

- self-consistent kinetic current coupling in the nonlinear 5D Vlasov-Maxwell loop
- same-deck electromagnetic GENE/CGYRO/GS2 output artifacts
- external electromagnetic phi/A_parallel/B_parallel same-case parity thresholds
- same-deck external electromagnetic grid-convergence evidence
