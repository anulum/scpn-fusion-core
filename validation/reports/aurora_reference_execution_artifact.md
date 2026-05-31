# Aurora Reference Execution Artifact

Public Aurora/Open-ADAS argon fractional-abundance execution artifact. This is a partial atomic-physics output, not full transport parity.

- Schema: `aurora-reference-execution-artifact.v1`
- Status: `blocked_partial_public_atomic_artifact_not_transport_parity`
- Artifact generated: `True`
- Reference output ready: `True`
- Same-case comparison ready: `False`
- Accepted full-fidelity ready: `False`

## Required Aurora/STRAHL output contract

- Schema: `aurora-strahl-output-contract.v1`
- Coordinate axes: `time_s, radius_m, charge_state`
- Observables: `charge_state_density_r_t, total_impurity_density_r_t, line_radiation_power_t, line_radiation_power_t_r_z, source_sink_matrix_t_r_z_z, total_impurity_inventory_t`

## Next action

Run a public Aurora or STRAHL radial transport case with line radiation, source/sink matrices, inventory closure, and native same-case comparison.

## Artifact

- Artifact: `validation/reference_data/full_fidelity_public_artifacts/aurora_argon_fractional_abundance_public.npz`
- Metadata: `validation/reference_data/full_fidelity_public_artifacts/aurora_argon_fractional_abundance_public.metadata.json`
- SHA-256: `a2b5b42d333609884976c70788fa94c22f4322d1d758094e186517916aaa93fd`
- Solver comparison ready: `False`
