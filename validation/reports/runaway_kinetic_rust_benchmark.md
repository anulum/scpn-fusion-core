# Full runaway kinetic Rust benchmark

- Grid: `[8, 16, 32]` in radius-pitch-momentum order
- Requested outputs: `3`
- Internal SSPRK3 steps: `4`
- NumPy median: `0.015892060 s`
- Rust median: `0.007313352 s`
- Rust speedup on this host/problem: `2.173x`
- Maximum component relative L2 error: `5.311e-14`
- Host load averages: `[6.72265625, 7.11669921875, 7.89404296875]`
- Source revision: `ec8f6e24dd8104df7b43efa62d32dc39bce64253`
- Scientific projection SHA-256: `247670c844b74fdcbb2b1a6a88e40a6273646dff80e17aad7214c7e3633c06a0`
- Compiled extension SHA-256: `89d46d7c10eecdee4a5c317c74fe472eeec5692b17cb50b011711bd27d17c335`

The timing is host-conditioned: a pinned DREAM reference run was active concurrently.
Rust is selected for the explicit production kernel because it preserves all outputs
and has a measured advantage here. Python remains the orchestration/DREAM/HDF5 tier.
Julia is gated on a future measured need for stiff implicit, sparse nonlinear, adjoint,
or differentiable solves. Go remains appropriate for service/orchestration surfaces,
not this tensor kernel.
