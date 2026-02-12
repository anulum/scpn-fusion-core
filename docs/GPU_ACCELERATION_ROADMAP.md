<!--
─────────────────────────────────────────────────────────────────────
SCPN Fusion Core — GPU Acceleration Roadmap
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
─────────────────────────────────────────────────────────────────────
-->

# GPU Acceleration Roadmap

Tracking issue: GitHub `#17` (GPU backend initiative)

## Scope

This roadmap defines a staged migration path from CPU-only solvers to hybrid CPU/GPU execution for the Grad-Shafranov and transport stack. It is intentionally implementation-focused and excludes speculative kernels.

## Baseline (2026-02-12)

- CPU reference path: Rust SOR/Chebyshev + kernel Picard iterations
- Typical grid sizes:
  - control-loop: 65x65
  - offline validation: 129x129 to 257x257
- Current bottleneck classes:
  - stencil sweeps (elliptic/SOR kernels)
  - Krylov preconditioning and dense linear algebra blocks
  - multilevel smooth/restrict/prolong operations

## Phase 1: `wgpu` SOR Kernel (Fastest Practical Win)

### Deliverables

- `fusion-math` compute-shader stencil kernel for red-black SOR
- Host orchestration in Rust with deterministic CPU fallback
- Validation harness: CPU vs GPU residual agreement on canonical grids

### Why first

- Stencil kernels map directly to GPU threadgroups
- Minimal algorithmic restructuring compared with existing CPU code
- `wgpu` provides NVIDIA/AMD/Intel/Apple portability

### Targets

- 65x65: 2x to 4x speedup
- 257x257: 5x to 12x speedup
- Numerical drift: max relative residual delta <= 1e-6 vs CPU reference

### Memory envelope

- State arrays (`psi`, `source`, scratch): 3 to 5 arrays
- 257x257 f64: approximately 2.7 MB total device memory footprint
- Optional f32 path for control-loop latency mode (guarded by tolerance checks)

## Phase 2: cuBLAS/rocBLAS-Backed GMRES Preconditioning

### Deliverables

- Device-backed dense blocks for Krylov orthogonalization/preconditioning
- Backend abstraction:
  - CUDA path (`cuBLAS`)
  - ROCm path (`rocBLAS`)
  - CPU fallback for unsupported nodes
- Benchmark suite for iteration count and wall-clock tradeoff

### Targets

- 2x to 6x speedup for linear-algebra-heavy inverse solves
- <5% change in iteration count vs CPU path
- Robust failover to CPU on driver/runtime mismatch

### Memory envelope

- Arnoldi basis + Hessenberg buffers dominate memory
- For 512-vector Krylov subspace (f64): approximately 0.5 to 1.0 GB budget

## Phase 3: Full Multigrid on GPU

### Deliverables

- GPU kernels for smooth/restrict/prolong across hierarchy levels
- Device-resident V-cycle orchestration (minimize host/device transfers)
- Coupled nonlinear solve path for kernel iteration

### Targets

- End-to-end equilibrium solve under 1 ms for control-loop sized grids
- 10x to 30x speedup on 257x257+ workloads relative to CPU baseline
- Maintain convergence parity with CPU V-cycle reference (residual slope match)

### Memory envelope

- Multi-level hierarchy storage:
  - finest + coarse levels + staging buffers
  - approximately 1.3x to 1.8x finest-grid storage
- 513x513 hierarchy (f64): approximately 40 to 80 MB total

## Hardware Targets

- Development floor:
  - NVIDIA RTX 3060 (12 GB)
  - AMD RX 6700 XT (12 GB)
  - Apple M-series integrated GPU (Metal via `wgpu`)
- CI/perf floor:
  - one CUDA runner
  - one non-CUDA `wgpu` runner (AMD/Intel/Metal)
- Production/HPC stretch:
  - NVIDIA A100/H100
  - AMD MI250/MI300

## Validation and Acceptance Gates

- Gate A: correctness
  - GPU residual curves overlap CPU baseline within tolerance
  - Deterministic replay for fixed seeds and inputs
- Gate B: performance
  - Speedup >= declared floor on reference hardware
  - Host-device copy overhead <20% of total solve time
- Gate C: operational safety
  - Runtime capability detection
  - automatic CPU fallback
  - telemetry for backend, precision, and kernel timings

## Risk Register

- Divergent precision behavior (`f32` vs `f64`)
- Driver fragmentation across vendors
- Kernel launch overhead dominating small grids
- CI portability constraints (GPU runners not always available)

## Execution Order

1. Phase 1 `wgpu` SOR prototype and parity tests
2. Phase 1 integration into benchmark harness
3. Phase 2 GPU linear algebra backend adapter
4. Phase 3 multigrid hierarchy kernels and nonlinear coupling
