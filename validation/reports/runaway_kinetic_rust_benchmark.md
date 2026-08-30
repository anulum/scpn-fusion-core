# Full runaway kinetic Rust benchmark

- Grid: `[8, 16, 32]` in radius-pitch-momentum order
- Requested outputs: `3`
- Internal SSPRK3 steps: `4`
- NumPy median: `0.014342346 s`
- Rust median: `0.006977463 s`
- Rust speedup on this host/problem: `2.056x`
- Maximum component relative L2 error: `5.272e-14`
- Host load averages: `[4.15478515625, 4.3359375, 5.40966796875]`
- Source revision: `0e40075d505daa2250accb0c287a69856ba61fd3`
- Scientific projection SHA-256: `cbd218dbe4f15b4735f5e5d740c11b71f7b30fc72479a312d91d056268941cd6`
- Compiled extension SHA-256: `e800a355b2045611027d1b7a7d08b10b4b8ad9db8533ff42cd41b6b6f22de7b4`

The timing is host-conditioned: a pinned DREAM reference run was active concurrently.
Rust is selected for the explicit production kernel because it preserves all outputs
and has a measured advantage here. Python remains the orchestration/DREAM/HDF5 tier.
Julia is gated on a future measured need for stiff implicit, sparse nonlinear, adjoint,
or differentiable solves. Go remains appropriate for service/orchestration surfaces,
not this tensor kernel.
