# FreeGS Public Example Reconstruction

This benchmark attempts same-case reconstruction from cached public FreeGS
examples. It accepts only the vacuum Green-function convention check and keeps
strict free-boundary parity blocked until nonlinear same-case native-vs-FreeGS
`psi(R,Z)` comparison evidence exists.

- Schema: `freegs-public-example-reconstruction-report.v1`
- Status: `blocked_public_freegs_vacuum_matched_missing_nonlinear_same_case_psi`
- Backend available: `True`
- FreeGS version: `0.8.2`
- Case count: `2`
- Vacuum comparison pass: `True`
- Accepted full fidelity: `False`
- Artifact: `validation/reference_data/full_fidelity_public_artifacts/freegs_public_example_reconstruction_attempt.json`

| Case | Machine | Vacuum NRMSE | Vacuum pass | Nonlinear status |
| --- | --- | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | 2.353629e-16 | True | `failed_external_backend_solve` |
| freegs_16_diiid_public_example | DIIID | 5.750194e-16 | True | `failed_external_backend_solve` |

## Missing full-fidelity requirements

- converged FreeGS/FreeGSNKE public nonlinear equilibrium output
- native same-case free-boundary profile-source reconstruction
- psi_N RMSE threshold against external output
- axis/X-point/boundary containment and q-profile thresholds
- grid convergence across public example resolutions
