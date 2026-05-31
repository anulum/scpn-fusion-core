# Production Decomposition Contract

Deterministic radial/toroidal decomposition contract for production-scale 5D nonlinear GK scheduling. This is not distributed runtime scaling evidence.

- Schema: `production-decomposition-contract.v1`
- Status: `blocked_contract_ready_missing_distributed_runtime_scaling`
- Contract pass: `True`
- Production-scale ready: `False`

| Case | R x T | Parts | Ranks | Imbalance | Halo overhead |
|---|---:|---:|---:|---:|---:|
| medium_64x32_4x2 | 64 x 32 | 4 x 2 | 8 | 1.000000 | 1.162109 |
| production_256x128_8x4 | 256 x 128 | 8 x 4 | 32 | 1.000000 | 1.104126 |

## Missing requirements

- MPI or multi-GPU execution path over the declared rank tiles
- halo exchange implementation and correctness tests
- large-grid cluster/GPU wall-time scaling report
- same-physics convergence evidence across decomposition shapes
