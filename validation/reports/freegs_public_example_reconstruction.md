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
- Accepted full fidelity: `False`
- Artifact: `validation/reference_data/full_fidelity_public_artifacts/freegs_public_example_reconstruction_attempt.json`

| Case | Machine | Vacuum NRMSE | Vacuum pass | Native psi_N RMSE | Axis error [m] | Nonlinear status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | 2.353629e-16 | True | 2.291811e-01 | 4.310348e-02 | `external_backend_solved_native_same_case_profile_source_compared_fail_closed` |
| freegs_16_diiid_public_example | DIIID | 5.750194e-16 | True | 6.656438e-02 | 2.250000e-01 | `external_backend_solved_native_same_case_profile_source_compared_fail_closed` |

## Missing full-fidelity requirements

- strict native-vs-FreeGS psi_N RMSE/current/axis/X-point/boundary threshold acceptance
- grid convergence across public example resolutions
- coil/vacuum reconstruction linked to public machine current sidecars
