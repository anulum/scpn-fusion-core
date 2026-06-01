# FreeGS Public Example Reconstruction

This benchmark attempts same-case reconstruction from cached public FreeGS
examples. It accepts only the vacuum Green-function convention check and keeps
strict free-boundary parity blocked until nonlinear same-case native-vs-FreeGS
`psi(R,Z)` comparison evidence exists.

- Schema: `freegs-public-example-reconstruction-report.v1`
- Status: `blocked_public_freegs_native_same_case_compared_missing_strict_threshold_grid_convergence_coil_sidecars`
- Backend available: `True`
- FreeGS version: `0.8.2`
- Case count: `2`
- Vacuum comparison pass: `True`
- External nonlinear output ready: `True`
- Native same-case comparison ready: `True`
- Strict threshold acceptance ready: `False`
- Grid convergence ready: `False`
- Coil/vacuum sidecar ready: `False`
- Geometry containment ready: `True`
- Boundary-containment metric ready: `True`
- Failed strict threshold checks: `6`
- Accepted full fidelity: `False`
- Artifact: `validation/reference_data/full_fidelity_public_artifacts/freegs_public_example_reconstruction_attempt.json`

| Case | Machine | Vacuum NRMSE | Vacuum pass | Native psi_N RMSE | Axis error [m] | Nonlinear status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | 2.353629e-16 | True | 2.291811e-01 | 4.310348e-02 | `external_backend_solved_native_same_case_profile_source_compared_fail_closed` |
| freegs_16_diiid_public_example | DIIID | 5.750194e-16 | True | 6.656438e-02 | 2.250000e-01 | `external_backend_solved_native_same_case_profile_source_compared_fail_closed` |

## Geometry containment evidence

- Schema: `strict-free-boundary-geometry-containment.v1`
- Status: `accepted_local_geometry_containment_evidence`
- Source points inside grid: `True`
- Axis containment metric ready: `True`
- Boundary-containment metric ready: `True`

| Case | Source points | X-points | Isoflux endpoints | Source inside | Axis inside | Boundary metric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| freegs_01_test_tokamak_freeboundary | 4 | 2 | 2 | True | True | True |
| freegs_16_diiid_public_example | 4 | 2 | 2 | True | True | True |

## Grid-convergence evidence

- Schema: `strict-free-boundary-grid-convergence-evidence.v1`
- Status: `blocked_public_freegs_single_resolution_grid_evidence`
- Required resolution count: `3`
- Grid convergence ready: `False`

| Case | Machine | Observed resolutions | Missing count | Ready | Blocking reason |
| --- | --- | --- | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | `65x65` | 2 | False | public_example_has_single_resolution |
| freegs_16_diiid_public_example | DIIID | `65x65` | 2 | False | public_example_has_single_resolution |

## Strict parity threshold checks

### `freegs_01_test_tokamak_freeboundary`

| Metric | Value | Comparator | Limit | Pass |
| --- | ---: | --- | ---: | ---: |
| psi_n_rmse | 0.229181105368877 | <= | 0.05 | False |
| axis_error_m | 0.043103481950417945 | <= | 0.025 | False |
| current_closure_relative_error | 4.7578550031175836e-06 | <= | 0.05 | True |
| boundary_max_abs_error_wb | 0.0 | <= | 1e-10 | True |
| xpoint_psi_n_error_max | 0.2709460979429926 | <= | 0.05 | False |
| boundary_containment_fraction | 1.0 | >= | 1.0 | True |
| q_profile_sanity_status | pass_finite_signed_q_profile | == | pass_finite_signed_q_profile | True |

### `freegs_16_diiid_public_example`

| Metric | Value | Comparator | Limit | Pass |
| --- | ---: | --- | ---: | ---: |
| psi_n_rmse | 0.06656438258503716 | <= | 0.05 | False |
| axis_error_m | 0.2250000000000001 | <= | 0.025 | False |
| current_closure_relative_error | 1.0581304436254637e-05 | <= | 0.05 | True |
| boundary_max_abs_error_wb | 0.0 | <= | 1e-10 | True |
| xpoint_psi_n_error_max | 0.06964126692247141 | <= | 0.05 | False |
| boundary_containment_fraction | 1.0 | >= | 1.0 | True |
| q_profile_sanity_status | pass_finite_signed_q_profile | == | pass_finite_signed_q_profile | True |


## Missing full-fidelity requirements

- strict native-vs-FreeGS psi_N RMSE/current/axis/X-point/boundary threshold acceptance
- grid convergence across public example resolutions
- coil/vacuum reconstruction linked to public machine current sidecars
