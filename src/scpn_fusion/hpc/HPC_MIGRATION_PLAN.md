# SCPN Migration Plan: From Workstation to Cluster/HPC

This roadmap tracks migration of SCPN core solves from single-node execution to
distributed domain-decomposed execution while preserving deterministic behavior.

## Phase 1: Hybrid Baseline (Current)
- Python orchestrates experiments and validation workflows.
- Rust crates (`fusion-core`, `fusion-control`) host compute-heavy kernels.
- Deterministic test/benchmark lanes remain the compatibility gate.

## Phase 2: MPI Domain-Decomposition Scaffolding (Delivered)
Implemented in `scpn-fusion-rs/crates/fusion-core/src/mpi_domain.rs`:
- `decompose_z(...)` for deterministic Z-slab partitioning.
- Halo utilities: `pack_halo_rows(...)`, `apply_halo_rows(...)`.
- Serial-equivalent exchange path: `serial_halo_exchange(...)`.
- Merge helpers: `split_with_halo(...)`, `stitch_without_halo(...)`.
- Equivalence metric helper: `l2_norm_delta(...)`.

This phase provides MPI-ready data movement semantics without forcing an MPI
runtime dependency in default builds.

## Phase 3: Cluster Runtime Enablement (Next)
- Add optional MPI backend wiring (e.g. `rsmpi`) behind a feature flag.
- Replace serial halo exchange calls with communicator-backed exchange.
- Add deterministic multi-rank smoke tests comparing stitched result to serial.

## Phase 4: Accelerator and In-Situ Tooling
- GPU offload for sparse/dense hot loops once distributed path is stable.
- In-situ diagnostics/visualization to reduce checkpoint IO overhead.

---
**Status (2026-02-14):** Phase 2 scaffolding delivered and validated via unit tests.
**Next Step:** Wire optional communicator-backed exchange path under feature-gated MPI runtime lane.
