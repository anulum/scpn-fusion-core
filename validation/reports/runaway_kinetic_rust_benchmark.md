# Full runaway kinetic Rust benchmark

- Grid: `[8, 16, 32]` in radius-pitch-momentum order
- Requested outputs: `3`
- Internal SSPRK3 steps: `4`
- NumPy median: `0.012024127 s`
- Rust median: `0.005894273 s`
- Rust speedup on this host/problem: `2.040x`
- Maximum component relative L2 error: `5.311e-14`
- Host load averages: `[4.28076171875, 4.919921875, 5.06201171875]`
- Source revision: `4590a3270a0263a2d86b7beb13af68525a26a6b9`
- Scientific projection SHA-256: `7f6482aa79dbc3f49113fa5cc4d898a19a9846ab39834f01ea0450ed069c9c7b`
- Compiled extension SHA-256: `89d46d7c10eecdee4a5c317c74fe472eeec5692b17cb50b011711bd27d17c335`

The timing is host-conditioned: a pinned DREAM reference run was active concurrently.
Rust is selected for the explicit production kernel because it preserves all outputs
and has a measured advantage here. Python remains the orchestration/DREAM/HDF5 tier.
Julia is gated on a future measured need for stiff implicit, sparse nonlinear, adjoint,
or differentiable solves. Go remains appropriate for service/orchestration surfaces,
not this tensor kernel.
