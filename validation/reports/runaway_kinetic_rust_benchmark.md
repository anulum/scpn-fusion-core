# Full runaway kinetic Rust benchmark

- Grid: `[8, 16, 32]` in radius-pitch-momentum order
- Requested outputs: `3`
- Internal SSPRK3 steps: `4`
- NumPy median: `0.015769141 s`
- Rust median: `0.007061332 s`
- Rust speedup on this host/problem: `2.233x`
- Maximum component relative L2 error: `5.357e-14`
- Host load averages: `[8.44775390625, 11.7080078125, 16.29736328125]`
- Source revision: `598beb237e784a4abc38b71e7245fbf235a56d97`
- Scientific projection SHA-256: `8e32f9ef25badc789dfb596c6628ed1bf752bc32dc00396a384b2dcc25972a59`
- Compiled extension SHA-256: `35cdfc022687955c5eda281b72a2d795ac25e557f618bac3f02a59b53b6da9af`

The timing is host-conditioned: a pinned DREAM reference run was active concurrently.
Rust is selected for the explicit production kernel because it preserves all outputs
and has a measured advantage here. Python remains the orchestration/DREAM/HDF5 tier.
Julia is gated on a future measured need for stiff implicit, sparse nonlinear, adjoint,
or differentiable solves. Go remains appropriate for service/orchestration surfaces,
not this tensor kernel.
