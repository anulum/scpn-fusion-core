# Production Decomposition Contract

Deterministic radial/toroidal decomposition contract for production-scale 5D nonlinear GK scheduling. This is not distributed runtime scaling evidence.

- Schema: `production-decomposition-contract.v1`
- Status: `blocked_contract_ready_missing_distributed_runtime_scaling`
- Contract pass: `True`
- Halo exchange pass: `True`
- Decomposition invariant pass: `True`
- Production-scale ready: `False`
- Python: `3.12.3`
- CPU count: `12`

| Case | R x T | Parts | Ranks | Imbalance | Halo overhead |
|---|---:|---:|---:|---:|---:|
| medium_64x32_4x2 | 64 x 32 | 4 x 2 | 8 | 1.000000 | 1.162109 |
| production_256x128_8x4 | 256 x 128 | 8 x 4 | 32 | 1.000000 | 1.104126 |

## Local CPU halo/invariant benchmark

| Case | Owned phase cells | Elapsed s | Cells/s | Halo | Reconstruction L_inf | Inventory rel | Free-energy rel |
|---|---:|---:|---:|:---:|---:|---:|---:|
| local_cpu_64x32_4x2 | 524288 | 3.408579e-02 | 1.538143e+07 | `True` | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

## Reproducible commands

- `python validation/benchmark_production_decomposition_contract.py`
- `python -m pytest tests/test_gk_domain_decomposition.py -q`

## Missing requirements

- MPI or multi-GPU execution path over the declared rank tiles
- large-grid cluster/GPU wall-time scaling report
- same-physics convergence evidence across distributed decomposition shapes
- hardware-specific multi-rank throughput and efficiency thresholds
