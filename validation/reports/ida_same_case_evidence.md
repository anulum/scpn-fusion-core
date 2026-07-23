# IDA free-boundary same-case evidence

- Status: `blocked_same_case_evidence`
- Schema: `scpn-fusion.ida-same-case-evidence.v2`
- Payload SHA-256: `641fc90fb1fccfd6a8de9d37e26ec66256f6edb6af9f9f84c4bc4143e89474ee`
- Facility/control/PCS/safety claims: `false`

## Cases

| Case | Role | Grid | ψ_N RMSE | current relative error | warm p95 (ms) |
|---|---|---:|---:|---:|---:|
| freegs_01_test_tokamak_freeboundary | development | 65×65 | 0.469381 | 0 | 10.415 |
| freegs_16_diiid_public_example | evaluation_candidate | 129×129 | 0.210997 | 0 | 23.4362 |

## Blockers

- `collaborator_solver_reference_not_bound`
- `evaluation_threshold_failed:latency`
- `evaluation_threshold_failed:psi_n_rmse`
- `execution_preceding_selection_lock_missing`
- `facility_validation_not_bound`
- `isolated_latency_evidence_missing`
- `pcs_and_safety_programmes_not_bound`
- `statistically_held_out_case_missing`
