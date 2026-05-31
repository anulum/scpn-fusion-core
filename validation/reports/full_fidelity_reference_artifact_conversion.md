# Full-Fidelity Reference Artifact Conversion

Conversion of cached public upstream outputs into tracked artifacts. Partial outputs remain outside full-fidelity acceptance until required observables and solver-output comparisons are present.

- Schema: `full-fidelity-reference-artifact-conversion.v1`
- Status: `partial_public_outputs_converted_not_full_fidelity`
- Accepted full-fidelity artifacts: `0`
- Partial public output artifacts: `3`
- Conversion modes: `external_cache_conversion`
- Reference manifest updated: `False`

## Converted public output artifacts

| Artifact | Surface | Family | Accepted | Comparison ready | Missing required observables | Path |
| --- | --- | --- | ---: | ---: | --- | --- |
| dream_avalanche_public_raw | runaway_electrons | DREAM | False | False | f_p_xi_t, runaway_current_t, avalanche_growth_rate_t, synchrotron_loss_power_t, partial_screening_drag_t | `validation/reference_data/full_fidelity_public_artifacts/dream_avalanche_public_raw.npz` |
| freegsnke_static_inverse_baseline_public | free_boundary_equilibrium | FreeGSNKE | False | False | strict_FreeGS_or_FreeGSNKE_coil_current_sidecar, boundary_contour, limiter_contour, native_psi_comparison, axis_or_xpoint_metadata | `validation/reference_data/full_fidelity_public_artifacts/freegsnke_static_inverse_baseline_public.npz` |
| freegsnke_mastu_current_sidecars_public | free_boundary_equilibrium | FreeGSNKE | False | False | boundary_contour, limiter_contour, axis_or_xpoint_metadata, same_case_native_psi_comparison | `validation/reference_data/full_fidelity_public_artifacts/freegsnke_mastu_current_sidecars_public.json` |

## Blocking sources

- native_nonlinear_gyrokinetics (GENE/CGYRO/GS2): cached public sources contain input decks, docs, GYRO linear outputs, or restart files, but no complete public nonlinear output artifact with the required heat-flux, zonal-flow, saturation, and electromagnetic field observables
- runaway_electrons (DREAM): DREAM avalanche HDF5 data was converted as a partial raw output artifact, but it does not contain the required f_p_xi_t, runaway_current_t, synchrotron_loss_power_t, and partial_screening_drag_t observables under the current manifest
- impurity_transport (Aurora/STRAHL): Aurora cache contains examples and docs, but no redistributed Aurora/STRAHL output artifact with charge-state density, total density, radiation, ionisation, and recombination matrices
- free_boundary_equilibrium (FreeGS/FreeGSNKE): FreeGSNKE baselines and current sidecars were converted as partial raw artifacts, but strict FreeGS parity still needs boundary/limiter metadata, axis/X-point data, and native psi comparison for the same public case
