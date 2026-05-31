# Production Decomposition Contract

Deterministic radial/toroidal decomposition contract for production-scale 5D nonlinear GK scheduling with executable local rank-tile evidence. This is not distributed MPI or multi-GPU scaling evidence.

- Schema: `production-decomposition-contract.v1`
- Status: `blocked_local_decomposition_ready_missing_distributed_runtime_scaling`
- Contract pass: `True`
- Communication contract ready: `True`
- Local decomposed execution pass: `True`
- Halo exchange pass: `True`
- Decomposition invariant pass: `True`
- Same-physics decomposition shape pass: `True`
- Production-scale ready: `False`
- Python: `3.12.3`
- CPU count: `12`

| Case | R x T | Parts | Ranks | Imbalance | Halo overhead |
|---|---:|---:|---:|---:|---:|
| medium_64x32_4x2 | 64 x 32 | 4 x 2 | 8 | 1.000000 | 1.162109 |
| production_256x128_8x4 | 256 x 128 | 8 x 4 | 32 | 1.000000 | 1.104126 |

## Rank communication contract

| Rank | Neighbours | Halo face shapes ready |
|---:|---|:---:|
| 0 | radial_lower=None, radial_upper=4, toroidal_lower=None, toroidal_upper=1 | `True` |
| 1 | radial_lower=None, radial_upper=5, toroidal_lower=0, toroidal_upper=2 | `True` |
| 2 | radial_lower=None, radial_upper=6, toroidal_lower=1, toroidal_upper=3 | `True` |
| 3 | radial_lower=None, radial_upper=7, toroidal_lower=2, toroidal_upper=None | `True` |
| 4 | radial_lower=0, radial_upper=8, toroidal_lower=None, toroidal_upper=5 | `True` |
| 5 | radial_lower=1, radial_upper=9, toroidal_lower=4, toroidal_upper=6 | `True` |
| 6 | radial_lower=2, radial_upper=10, toroidal_lower=5, toroidal_upper=7 | `True` |
| 7 | radial_lower=3, radial_upper=11, toroidal_lower=6, toroidal_upper=None | `True` |
| 8 | radial_lower=4, radial_upper=12, toroidal_lower=None, toroidal_upper=9 | `True` |
| 9 | radial_lower=5, radial_upper=13, toroidal_lower=8, toroidal_upper=10 | `True` |
| 10 | radial_lower=6, radial_upper=14, toroidal_lower=9, toroidal_upper=11 | `True` |
| 11 | radial_lower=7, radial_upper=15, toroidal_lower=10, toroidal_upper=None | `True` |
| 12 | radial_lower=8, radial_upper=16, toroidal_lower=None, toroidal_upper=13 | `True` |
| 13 | radial_lower=9, radial_upper=17, toroidal_lower=12, toroidal_upper=14 | `True` |
| 14 | radial_lower=10, radial_upper=18, toroidal_lower=13, toroidal_upper=15 | `True` |
| 15 | radial_lower=11, radial_upper=19, toroidal_lower=14, toroidal_upper=None | `True` |
| 16 | radial_lower=12, radial_upper=20, toroidal_lower=None, toroidal_upper=17 | `True` |
| 17 | radial_lower=13, radial_upper=21, toroidal_lower=16, toroidal_upper=18 | `True` |
| 18 | radial_lower=14, radial_upper=22, toroidal_lower=17, toroidal_upper=19 | `True` |
| 19 | radial_lower=15, radial_upper=23, toroidal_lower=18, toroidal_upper=None | `True` |
| 20 | radial_lower=16, radial_upper=24, toroidal_lower=None, toroidal_upper=21 | `True` |
| 21 | radial_lower=17, radial_upper=25, toroidal_lower=20, toroidal_upper=22 | `True` |
| 22 | radial_lower=18, radial_upper=26, toroidal_lower=21, toroidal_upper=23 | `True` |
| 23 | radial_lower=19, radial_upper=27, toroidal_lower=22, toroidal_upper=None | `True` |
| 24 | radial_lower=20, radial_upper=28, toroidal_lower=None, toroidal_upper=25 | `True` |
| 25 | radial_lower=21, radial_upper=29, toroidal_lower=24, toroidal_upper=26 | `True` |
| 26 | radial_lower=22, radial_upper=30, toroidal_lower=25, toroidal_upper=27 | `True` |
| 27 | radial_lower=23, radial_upper=31, toroidal_lower=26, toroidal_upper=None | `True` |
| 28 | radial_lower=24, radial_upper=None, toroidal_lower=None, toroidal_upper=29 | `True` |
| 29 | radial_lower=25, radial_upper=None, toroidal_lower=28, toroidal_upper=30 | `True` |
| 30 | radial_lower=26, radial_upper=None, toroidal_lower=29, toroidal_upper=31 | `True` |
| 31 | radial_lower=27, radial_upper=None, toroidal_lower=30, toroidal_upper=None | `True` |

## Local CPU halo/invariant benchmark

| Case | Ranks | Owned phase cells | Elapsed s | Cells/s | Local execution | Halo | Reconstruction L_inf | Inventory rel | Free-energy rel |
|---|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|
| local_cpu_64x32_4x2 | 8 | 524288 | 1.062676e-01 | 4.933659e+06 | `True` | `True` | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| local_cpu_64x32_8x1 | 8 | 524288 | 5.155375e-02 | 1.016974e+07 | `True` | `True` | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

## Reproducible commands

- `python validation/benchmark_production_decomposition_contract.py`
- `python -m pytest tests/test_gk_domain_decomposition.py -q`

## Missing requirements

- MPI or multi-GPU distributed execution path over the declared rank tiles
- large-grid cluster/GPU wall-time scaling report
- same-physics convergence evidence across distributed decomposition shapes
- hardware-specific multi-rank throughput and efficiency thresholds
