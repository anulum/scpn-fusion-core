# Full runaway kinetic Rust benchmark

- Grid: `[8, 16, 32]` in radius-pitch-momentum order
- Requested outputs: `3`
- Internal SSPRK3 steps: `4`
- NumPy median: `0.018411425 s`
- Rust median: `0.009335502 s`
- Rust speedup on this host/problem: `1.972x`
- Maximum component relative L2 error: `5.311e-14`
- Host load averages: `[8.51025390625, 7.76708984375, 8.7255859375]`
- Source revision: `e64bcd3d0bafaabcfc8b057881112379cea3a89d`
- Scientific projection SHA-256: `e71f76467da9c8cc17de9e4053de8ba8ebacd1fc0b3f8f43be9bf86a380b2b73`
- Compiled extension SHA-256: `89d46d7c10eecdee4a5c317c74fe472eeec5692b17cb50b011711bd27d17c335`

The timing is host-conditioned: a pinned DREAM reference run was active concurrently.
Rust is selected for the explicit production kernel because it preserves all outputs
and has a measured advantage here. Python remains the orchestration/DREAM/HDF5 tier.
Julia is gated on a future measured need for stiff implicit, sparse nonlinear, adjoint,
or differentiable solves. Go remains appropriate for service/orchestration surfaces,
not this tensor kernel.
