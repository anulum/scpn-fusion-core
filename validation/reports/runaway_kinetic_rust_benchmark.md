# Full runaway kinetic Rust benchmark

- Grid: `[8, 16, 32]` in radius-pitch-momentum order
- Requested outputs: `3`
- Internal SSPRK3 steps: `4`
- NumPy median: `0.012564712 s`
- Rust median: `0.005990588 s`
- Rust speedup on this host/problem: `2.097x`
- Maximum component relative L2 error: `5.357e-14`
- Host load averages: `[7.560546875, 14.55517578125, 14.5869140625]`
- Source revision: `98b6d4d2cfde1e8cf22129fa90e779b0291f8149`
- Scientific projection SHA-256: `a3a4539fab7339e5da090aaa2fcd9c1281f6e87831b81f2cf2fa6bb0cd8ecffd`
- Compiled extension SHA-256: `35cdfc022687955c5eda281b72a2d795ac25e557f618bac3f02a59b53b6da9af`

The timing is host-conditioned: Local shared host; no concurrent workload was declared. Timings are evidence for this host and problem only.
Rust is selected for the explicit production kernel because it preserves all outputs
and has a measured advantage here. Python remains the orchestration/DREAM/HDF5 tier.
Julia is gated on a future measured need for stiff implicit, sparse nonlinear, adjoint,
or differentiable solves. Go remains appropriate for service/orchestration surfaces,
not this tensor kernel.
