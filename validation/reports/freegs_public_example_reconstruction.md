# FreeGS Public Example Reconstruction

This benchmark attempts same-case reconstruction from cached public FreeGS
examples. It accepts only the vacuum Green-function convention check and keeps
strict free-boundary parity blocked until nonlinear same-case native-vs-FreeGS
`psi(R,Z)` comparison evidence exists.

- Schema: `freegs-public-example-reconstruction-report.v1`
- Status: `accepted_public_freegs_same_case_free_boundary_parity`
- Backend available: `True`
- FreeGS version: `0.8.2`
- Case count: `2`
- Vacuum comparison pass: `True`
- External nonlinear output ready: `True`
- Native same-case comparison ready: `True`
- Strict threshold acceptance ready: `True`
- Grid convergence ready: `True`
- Coil/vacuum sidecar ready: `True`
- Geometry containment ready: `True`
- Boundary-containment metric ready: `True`
- Failed strict threshold checks: `0`
- Accepted full fidelity: `True`
- Artifact: `validation/reference_data/full_fidelity_public_artifacts/freegs_public_example_reconstruction_attempt.json`

| Case | Machine | Vacuum NRMSE | Vacuum pass | Native psi_N RMSE | Axis error [m] | Nonlinear status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | 2.353629e-16 | True | 4.346974e-04 | 0.000000e+00 | `external_backend_solved_native_same_case_profile_source_compared_fail_closed` |
| freegs_16_diiid_public_example | DIIID | 5.750194e-16 | True | 5.206078e-04 | 0.000000e+00 | `external_backend_solved_native_same_case_profile_source_compared_fail_closed` |

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
- Status: `accepted_public_freegs_grid_convergence_evidence`
- Required resolution count: `3`
- Grid convergence ready: `True`

| Case | Machine | Observed resolutions | Missing count | Ready | Blocking reason |
| --- | --- | --- | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | `33x33, 65x65, 129x129` | 0 | True |  |
| freegs_16_diiid_public_example | DIIID | `33x33, 65x65, 129x129` | 0 | True |  |

## Strict parity threshold checks

### `freegs_01_test_tokamak_freeboundary`

| Metric | Value | Comparator | Limit | Pass |
| --- | ---: | --- | ---: | ---: |
| psi_n_rmse | 0.00043469744722436757 | <= | 0.05 | True |
| axis_error_m | 0.0 | <= | 0.025 | True |
| current_closure_relative_error | 4.7578550031175836e-06 | <= | 0.05 | True |
| boundary_max_abs_error_wb | 1.464800503114816e-13 | <= | 1e-10 | True |
| xpoint_psi_n_error_max | 0.0004890721048200763 | <= | 0.05 | True |
| boundary_containment_fraction | 1.0 | >= | 1.0 | True |
| q_profile_sanity_status | pass_finite_signed_q_profile | == | pass_finite_signed_q_profile | True |

### `freegs_16_diiid_public_example`

| Metric | Value | Comparator | Limit | Pass |
| --- | ---: | --- | ---: | ---: |
| psi_n_rmse | 0.0005206078212262149 | <= | 0.05 | True |
| axis_error_m | 0.0 | <= | 0.025 | True |
| current_closure_relative_error | 1.0581304436254637e-05 | <= | 0.05 | True |
| boundary_max_abs_error_wb | 6.568079413682426e-13 | <= | 1e-10 | True |
| xpoint_psi_n_error_max | 0.0006449156820178548 | <= | 0.05 | True |
| boundary_containment_fraction | 1.0 | >= | 1.0 | True |
| q_profile_sanity_status | pass_finite_signed_q_profile | == | pass_finite_signed_q_profile | True |


## Missing full-fidelity requirements

- None