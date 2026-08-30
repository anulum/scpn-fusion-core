# Real TORAX runtime contract

Overall gate: **PASS**.

## Gates

- `schemas_exact`: PASS
- `all_runs_complete`: PASS
- `deterministic_projection`: PASS
- `deterministic_complete_sidecar_content`: PASS
- `deterministic_review_envelope`: PASS
- `refinement_converged`: PASS
- `complete_inventory`: PASS
- `backend_scalars_retained`: PASS
- `inferred_scalars_omitted_from_typed_projection`: PASS
- `clock_exact`: PASS
- `sidecar_custody_verified`: PASS
- `source_totals_finite`: PASS
- `state_budgets_finite`: PASS

## Fixed-timestep refinement

- `electron_density` relative L2: `0.000255524754507`
- `electron_temperature` relative L2: `0.000896212145609`
- `ion_temperature` relative L2: `0.00133874793953`
- `poloidal_flux` relative L2: `9.70515601354e-05`

The public runtime executed real TORAX 1.4.3 through the isolated CLI. 
The typed projection contains only Ti, Te, ne, poloidal flux, source totals, 
state budgets, and numerical status. The checksummed NetCDF DataTree sidecar 
retains every backend variable. No actuation, experimental-validation, full-physics 
equivalence, or portable-performance claim is made.
