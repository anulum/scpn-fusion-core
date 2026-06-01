# Production Decomposition Contract

Deterministic radial/toroidal decomposition contract for production-scale 5D nonlinear GK scheduling with executable local rank-tile evidence. This is not distributed MPI or multi-GPU scaling evidence.

- Schema: `production-decomposition-contract.v1`
- Status: `blocked_local_decomposition_ready_missing_distributed_runtime_scaling`
- Contract pass: `True`
- Communication contract ready: `True`
- Halo-face integrity pass: `True`
- Local decomposed execution pass: `True`
- Halo exchange pass: `True`
- Decomposition invariant pass: `True`
- Parallel-moment invariant pass: `True`
- Reciprocal neighbour graph pass: `True`
- Same-physics decomposition shape pass: `True`
- Production-scale ready: `False`
- Python: `3.12.3`
- CPU count: `12`

| Case | R x T | Parts | Ranks | Imbalance | Halo overhead |
|---|---:|---:|---:|---:|---:|
| medium_64x32_4x2 | 64 x 32 | 4 x 2 | 8 | 1.000000 | 1.162109 |
| production_256x128_8x4 | 256 x 128 | 8 x 4 | 32 | 1.000000 | 1.104126 |

## Local serial halo-face integrity

- Schema: `production-decomposition-halo-face-integrity.v1`
- Status: `accepted_local_serial_halo_face_integrity`
- Case: `local_cpu_64x32_4x2`
- Checked faces: `20`
- Max halo-face L_inf error: `0.000000e+00`
- Distributed runtime halo exchange ready: `False`

| Rank | Face | Neighbour | Shape | L_inf | Pass |
|---:|---|---:|---|---:|:---:|
| 0 | radial_upper | 2 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 0 | toroidal_upper | 1 | `[16, 1, 8, 8, 4]` | 0.000000e+00 | `True` |
| 1 | radial_upper | 3 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 1 | toroidal_lower | 0 | `[16, 1, 8, 8, 4]` | 0.000000e+00 | `True` |
| 2 | radial_lower | 0 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 2 | radial_upper | 4 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 2 | toroidal_upper | 3 | `[16, 1, 8, 8, 4]` | 0.000000e+00 | `True` |
| 3 | radial_lower | 1 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 3 | radial_upper | 5 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 3 | toroidal_lower | 2 | `[16, 1, 8, 8, 4]` | 0.000000e+00 | `True` |
| 4 | radial_lower | 2 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 4 | radial_upper | 6 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 4 | toroidal_upper | 5 | `[16, 1, 8, 8, 4]` | 0.000000e+00 | `True` |
| 5 | radial_lower | 3 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 5 | radial_upper | 7 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 5 | toroidal_lower | 4 | `[16, 1, 8, 8, 4]` | 0.000000e+00 | `True` |
| 6 | radial_lower | 4 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 6 | toroidal_upper | 7 | `[16, 1, 8, 8, 4]` | 0.000000e+00 | `True` |
| 7 | radial_lower | 5 | `[1, 16, 8, 8, 4]` | 0.000000e+00 | `True` |
| 7 | toroidal_lower | 6 | `[16, 1, 8, 8, 4]` | 0.000000e+00 | `True` |

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

## Distributed communication volume evidence

- Schema: `production-decomposition-communication-volume.v1`
- Status: `blocked_missing_distributed_runtime_execution`
- Distributed runtime ready: `False`
- Halo dtype: `float64`
- Communicating faces: `104`
- Total halo-exchange bytes per step: `436207616`
- Max rank halo-exchange bytes per step: `16777216`

| Rank | Bytes/step | Communicating faces |
|---:|---:|---:|
| 0 | 8388608 | 2 |
| 1 | 12582912 | 3 |
| 2 | 12582912 | 3 |
| 3 | 8388608 | 2 |
| 4 | 12582912 | 3 |
| 5 | 16777216 | 4 |
| 6 | 16777216 | 4 |
| 7 | 12582912 | 3 |
| 8 | 12582912 | 3 |
| 9 | 16777216 | 4 |
| 10 | 16777216 | 4 |
| 11 | 12582912 | 3 |
| 12 | 12582912 | 3 |
| 13 | 16777216 | 4 |
| 14 | 16777216 | 4 |
| 15 | 12582912 | 3 |
| 16 | 12582912 | 3 |
| 17 | 16777216 | 4 |
| 18 | 16777216 | 4 |
| 19 | 12582912 | 3 |
| 20 | 12582912 | 3 |
| 21 | 16777216 | 4 |
| 22 | 16777216 | 4 |
| 23 | 12582912 | 3 |
| 24 | 12582912 | 3 |
| 25 | 16777216 | 4 |
| 26 | 16777216 | 4 |
| 27 | 12582912 | 3 |
| 28 | 8388608 | 2 |
| 29 | 12582912 | 3 |
| 30 | 12582912 | 3 |
| 31 | 8388608 | 2 |

## Reciprocal neighbour graph evidence

- Schema: `production-decomposition-reciprocal-neighbour-graph.v1`
- Status: `accepted_local_reciprocal_neighbour_graph`
- Reciprocal neighbour graph pass: `True`
- Directed links: `104`
- Undirected links: `52`
- Mismatched links: `0`
- Max payload byte asymmetry: `0`

| Rank | Face | Neighbour | Opposite face | Payload bytes | Reciprocal bytes | Pass |
|---:|---|---:|---|---:|---:|:---:|
| 0 | radial_upper | 4 | radial_lower | 4194304 | 4194304 | `True` |
| 0 | toroidal_upper | 1 | toroidal_lower | 4194304 | 4194304 | `True` |
| 1 | radial_upper | 5 | radial_lower | 4194304 | 4194304 | `True` |
| 1 | toroidal_lower | 0 | toroidal_upper | 4194304 | 4194304 | `True` |
| 1 | toroidal_upper | 2 | toroidal_lower | 4194304 | 4194304 | `True` |
| 2 | radial_upper | 6 | radial_lower | 4194304 | 4194304 | `True` |
| 2 | toroidal_lower | 1 | toroidal_upper | 4194304 | 4194304 | `True` |
| 2 | toroidal_upper | 3 | toroidal_lower | 4194304 | 4194304 | `True` |
| 3 | radial_upper | 7 | radial_lower | 4194304 | 4194304 | `True` |
| 3 | toroidal_lower | 2 | toroidal_upper | 4194304 | 4194304 | `True` |
| 4 | radial_lower | 0 | radial_upper | 4194304 | 4194304 | `True` |
| 4 | radial_upper | 8 | radial_lower | 4194304 | 4194304 | `True` |
| 4 | toroidal_upper | 5 | toroidal_lower | 4194304 | 4194304 | `True` |
| 5 | radial_lower | 1 | radial_upper | 4194304 | 4194304 | `True` |
| 5 | radial_upper | 9 | radial_lower | 4194304 | 4194304 | `True` |
| 5 | toroidal_lower | 4 | toroidal_upper | 4194304 | 4194304 | `True` |
| 5 | toroidal_upper | 6 | toroidal_lower | 4194304 | 4194304 | `True` |
| 6 | radial_lower | 2 | radial_upper | 4194304 | 4194304 | `True` |
| 6 | radial_upper | 10 | radial_lower | 4194304 | 4194304 | `True` |
| 6 | toroidal_lower | 5 | toroidal_upper | 4194304 | 4194304 | `True` |
| 6 | toroidal_upper | 7 | toroidal_lower | 4194304 | 4194304 | `True` |
| 7 | radial_lower | 3 | radial_upper | 4194304 | 4194304 | `True` |
| 7 | radial_upper | 11 | radial_lower | 4194304 | 4194304 | `True` |
| 7 | toroidal_lower | 6 | toroidal_upper | 4194304 | 4194304 | `True` |
| 8 | radial_lower | 4 | radial_upper | 4194304 | 4194304 | `True` |
| 8 | radial_upper | 12 | radial_lower | 4194304 | 4194304 | `True` |
| 8 | toroidal_upper | 9 | toroidal_lower | 4194304 | 4194304 | `True` |
| 9 | radial_lower | 5 | radial_upper | 4194304 | 4194304 | `True` |
| 9 | radial_upper | 13 | radial_lower | 4194304 | 4194304 | `True` |
| 9 | toroidal_lower | 8 | toroidal_upper | 4194304 | 4194304 | `True` |
| 9 | toroidal_upper | 10 | toroidal_lower | 4194304 | 4194304 | `True` |
| 10 | radial_lower | 6 | radial_upper | 4194304 | 4194304 | `True` |
| 10 | radial_upper | 14 | radial_lower | 4194304 | 4194304 | `True` |
| 10 | toroidal_lower | 9 | toroidal_upper | 4194304 | 4194304 | `True` |
| 10 | toroidal_upper | 11 | toroidal_lower | 4194304 | 4194304 | `True` |
| 11 | radial_lower | 7 | radial_upper | 4194304 | 4194304 | `True` |
| 11 | radial_upper | 15 | radial_lower | 4194304 | 4194304 | `True` |
| 11 | toroidal_lower | 10 | toroidal_upper | 4194304 | 4194304 | `True` |
| 12 | radial_lower | 8 | radial_upper | 4194304 | 4194304 | `True` |
| 12 | radial_upper | 16 | radial_lower | 4194304 | 4194304 | `True` |
| 12 | toroidal_upper | 13 | toroidal_lower | 4194304 | 4194304 | `True` |
| 13 | radial_lower | 9 | radial_upper | 4194304 | 4194304 | `True` |
| 13 | radial_upper | 17 | radial_lower | 4194304 | 4194304 | `True` |
| 13 | toroidal_lower | 12 | toroidal_upper | 4194304 | 4194304 | `True` |
| 13 | toroidal_upper | 14 | toroidal_lower | 4194304 | 4194304 | `True` |
| 14 | radial_lower | 10 | radial_upper | 4194304 | 4194304 | `True` |
| 14 | radial_upper | 18 | radial_lower | 4194304 | 4194304 | `True` |
| 14 | toroidal_lower | 13 | toroidal_upper | 4194304 | 4194304 | `True` |
| 14 | toroidal_upper | 15 | toroidal_lower | 4194304 | 4194304 | `True` |
| 15 | radial_lower | 11 | radial_upper | 4194304 | 4194304 | `True` |
| 15 | radial_upper | 19 | radial_lower | 4194304 | 4194304 | `True` |
| 15 | toroidal_lower | 14 | toroidal_upper | 4194304 | 4194304 | `True` |
| 16 | radial_lower | 12 | radial_upper | 4194304 | 4194304 | `True` |
| 16 | radial_upper | 20 | radial_lower | 4194304 | 4194304 | `True` |
| 16 | toroidal_upper | 17 | toroidal_lower | 4194304 | 4194304 | `True` |
| 17 | radial_lower | 13 | radial_upper | 4194304 | 4194304 | `True` |
| 17 | radial_upper | 21 | radial_lower | 4194304 | 4194304 | `True` |
| 17 | toroidal_lower | 16 | toroidal_upper | 4194304 | 4194304 | `True` |
| 17 | toroidal_upper | 18 | toroidal_lower | 4194304 | 4194304 | `True` |
| 18 | radial_lower | 14 | radial_upper | 4194304 | 4194304 | `True` |
| 18 | radial_upper | 22 | radial_lower | 4194304 | 4194304 | `True` |
| 18 | toroidal_lower | 17 | toroidal_upper | 4194304 | 4194304 | `True` |
| 18 | toroidal_upper | 19 | toroidal_lower | 4194304 | 4194304 | `True` |
| 19 | radial_lower | 15 | radial_upper | 4194304 | 4194304 | `True` |
| 19 | radial_upper | 23 | radial_lower | 4194304 | 4194304 | `True` |
| 19 | toroidal_lower | 18 | toroidal_upper | 4194304 | 4194304 | `True` |
| 20 | radial_lower | 16 | radial_upper | 4194304 | 4194304 | `True` |
| 20 | radial_upper | 24 | radial_lower | 4194304 | 4194304 | `True` |
| 20 | toroidal_upper | 21 | toroidal_lower | 4194304 | 4194304 | `True` |
| 21 | radial_lower | 17 | radial_upper | 4194304 | 4194304 | `True` |
| 21 | radial_upper | 25 | radial_lower | 4194304 | 4194304 | `True` |
| 21 | toroidal_lower | 20 | toroidal_upper | 4194304 | 4194304 | `True` |
| 21 | toroidal_upper | 22 | toroidal_lower | 4194304 | 4194304 | `True` |
| 22 | radial_lower | 18 | radial_upper | 4194304 | 4194304 | `True` |
| 22 | radial_upper | 26 | radial_lower | 4194304 | 4194304 | `True` |
| 22 | toroidal_lower | 21 | toroidal_upper | 4194304 | 4194304 | `True` |
| 22 | toroidal_upper | 23 | toroidal_lower | 4194304 | 4194304 | `True` |
| 23 | radial_lower | 19 | radial_upper | 4194304 | 4194304 | `True` |
| 23 | radial_upper | 27 | radial_lower | 4194304 | 4194304 | `True` |
| 23 | toroidal_lower | 22 | toroidal_upper | 4194304 | 4194304 | `True` |
| 24 | radial_lower | 20 | radial_upper | 4194304 | 4194304 | `True` |
| 24 | radial_upper | 28 | radial_lower | 4194304 | 4194304 | `True` |
| 24 | toroidal_upper | 25 | toroidal_lower | 4194304 | 4194304 | `True` |
| 25 | radial_lower | 21 | radial_upper | 4194304 | 4194304 | `True` |
| 25 | radial_upper | 29 | radial_lower | 4194304 | 4194304 | `True` |
| 25 | toroidal_lower | 24 | toroidal_upper | 4194304 | 4194304 | `True` |
| 25 | toroidal_upper | 26 | toroidal_lower | 4194304 | 4194304 | `True` |
| 26 | radial_lower | 22 | radial_upper | 4194304 | 4194304 | `True` |
| 26 | radial_upper | 30 | radial_lower | 4194304 | 4194304 | `True` |
| 26 | toroidal_lower | 25 | toroidal_upper | 4194304 | 4194304 | `True` |
| 26 | toroidal_upper | 27 | toroidal_lower | 4194304 | 4194304 | `True` |
| 27 | radial_lower | 23 | radial_upper | 4194304 | 4194304 | `True` |
| 27 | radial_upper | 31 | radial_lower | 4194304 | 4194304 | `True` |
| 27 | toroidal_lower | 26 | toroidal_upper | 4194304 | 4194304 | `True` |
| 28 | radial_lower | 24 | radial_upper | 4194304 | 4194304 | `True` |
| 28 | toroidal_upper | 29 | toroidal_lower | 4194304 | 4194304 | `True` |
| 29 | radial_lower | 25 | radial_upper | 4194304 | 4194304 | `True` |
| 29 | toroidal_lower | 28 | toroidal_upper | 4194304 | 4194304 | `True` |
| 29 | toroidal_upper | 30 | toroidal_lower | 4194304 | 4194304 | `True` |
| 30 | radial_lower | 26 | radial_upper | 4194304 | 4194304 | `True` |
| 30 | toroidal_lower | 29 | toroidal_upper | 4194304 | 4194304 | `True` |
| 30 | toroidal_upper | 31 | toroidal_lower | 4194304 | 4194304 | `True` |
| 31 | radial_lower | 27 | radial_upper | 4194304 | 4194304 | `True` |
| 31 | toroidal_lower | 30 | toroidal_upper | 4194304 | 4194304 | `True` |

## Distributed scaling gate

- Schema: `production-decomposition-distributed-scaling-gate.v1`
- Status: `blocked_missing_distributed_scaling_measurements`
- Distributed scaling ready: `False`
- Measured run count: `0`
- Required rank counts: `[1, 2, 4, 8, 16, 32]`
- Minimum parallel efficiency threshold: `0.70`
- Minimum weak-scaling efficiency threshold: `0.80`
- Estimated halo bytes per step: `436207616`
- Blocking reason: MPI or multi-GPU distributed runtime measurements are required before production-scale decomposition can be accepted.

Required measurements:
- wall_time_s by rank count for the same physics deck
- parallel efficiency relative to the one-rank baseline
- weak-scaling efficiency at fixed owned phase cells per rank
- hardware metadata for CPU, accelerator, interconnect, and driver stack
- decomposition-invariant physics checks for every distributed run

## Local CPU halo/invariant benchmark

| Case | Ranks | Owned phase cells | Elapsed s | Cells/s | Local execution | Halo | Reconstruction L_inf | Inventory rel | Free-energy rel |
|---|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|
| local_cpu_64x32_4x2 | 8 | 524288 | 5.259111e-02 | 9.969138e+06 | `True` | `True` | 0.000000e+00 | 0.000000e+00 | 1.665333e-16 |
| local_cpu_64x32_8x1 | 8 | 524288 | 1.491587e-01 | 3.514968e+06 | `True` | `True` | 0.000000e+00 | 0.000000e+00 | 1.665333e-16 |
| local_cpu_64x32_2x4 | 8 | 524288 | 1.786316e-01 | 2.935024e+06 | `True` | `True` | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

## Same-physics decomposition-shape convergence

- Schema: `production-decomposition-shape-convergence.v1`
- Status: `accepted_local_same_physics_shape_convergence`
- Shape convergence pass: `True`
- Reference case: `local_cpu_64x32_4x2`
- Max inventory relative deviation: `0.000000e+00`
- Max free-energy relative deviation: `3.330666e-16`
- Max parallel-moment relative deviation: `0.000000e+00`
- Relative reduction tolerance: `1.000000e-12`

| Case | Ranks | Owned phase cells | Cells/s | Inventory rel dev | Free-energy rel dev | Parallel-moment rel dev | Reconstruction L_inf | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| local_cpu_64x32_4x2 | 8 | 524288 | 9.969138e+06 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | `True` |
| local_cpu_64x32_8x1 | 8 | 524288 | 3.514968e+06 | 0.000000e+00 | 3.330666e-16 | 0.000000e+00 | 0.000000e+00 | `True` |
| local_cpu_64x32_2x4 | 8 | 524288 | 2.935024e+06 | 0.000000e+00 | 1.665333e-16 | 0.000000e+00 | 0.000000e+00 | `True` |

## Reproducible commands

- `python validation/benchmark_production_decomposition_contract.py`
- `python -m pytest tests/test_gk_domain_decomposition.py -q`

## Missing requirements

- MPI or multi-GPU distributed execution path over the declared rank tiles
- large-grid cluster/GPU wall-time scaling report
- same-physics convergence evidence across distributed MPI/multi-GPU decomposition shapes
- hardware-specific multi-rank throughput and efficiency thresholds
- accepted distributed scaling gate over required rank counts
