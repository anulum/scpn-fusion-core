# IDA free-boundary same-case evidence

- Status: `blocked_same_case_evidence`
- Schema: `scpn-fusion.ida-same-case-evidence.v2`
- Payload SHA-256: `7dda4c095acfed4603246e474d13c1516fd458834bb1a5ee60aa8665f9fef939`
- Facility/control/PCS/safety claims: `false`

## Cases

| Case | Role | Grid | ψ_N RMSE | current relative error | warm p95 (ms) |
|---|---|---:|---:|---:|---:|
| freegs_01_test_tokamak_freeboundary | development | 65×65 | 0.469381 | 1.45519e-16 | 12.2332 |
| freegs_16_diiid_public_example | evaluation_candidate | 129×129 | 0.210997 | 0 | 23.3701 |

## Blockers

- `collaborator_solver_reference_not_bound`
- `evaluation_threshold_failed:latency`
- `evaluation_threshold_failed:psi_n_rmse`
- `execution_preceding_selection_lock_missing`
- `facility_validation_not_bound`
- `isolated_latency_evidence_missing`
- `pcs_and_safety_programmes_not_bound`
- `statistically_held_out_case_missing`
