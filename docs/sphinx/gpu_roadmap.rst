SCPN GPU Roadmap
================

The GPU acceleration program is tracked in ``docs/GPU_ACCELERATION_ROADMAP.md``.
This page summarizes the execution sequence and acceptance criteria for quick
reference in the Sphinx docs set.

Scope
-----

- Target path: hybrid CPU/GPU execution for Grad-Shafranov + transport kernels
- Explicitly non-speculative: implementation-first roadmap only
- Tracking issue: GitHub ``#17``

Execution Phases
----------------

1. ``wgpu`` SOR kernel (red-black stencil + deterministic CPU fallback)
2. GPU-backed GMRES preconditioning (CUDA/ROCm adapters with CPU fallback)
3. Full multigrid on device (smooth/restrict/prolong + coupled nonlinear path)

Performance Targets
-------------------

- Phase 1:
  - 65x65: 2x to 4x
  - 257x257: 5x to 12x
- Phase 2:
  - 2x to 6x speedup on linear-algebra-heavy inverse solves
- Phase 3:
  - under 1 ms for control-loop grids
  - 10x to 30x speedup for 257x257+ workloads

Acceptance Gates
----------------

- Correctness: residual behavior matches CPU reference within configured tolerance
- Performance: measured speedups meet declared minimum floors
- Operations: runtime capability detection + automatic CPU fallback

Reference
---------

- Full plan: ``docs/GPU_ACCELERATION_ROADMAP.md``
