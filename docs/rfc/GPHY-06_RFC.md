<!--
SCPN Fusion Core — RFC GPHY-06
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# RFC GPHY-06 — Runtime Regime Kernel Specialization (Reduced JIT Lane)

Status: Draft for review

## 1. Task Identity

- Task ID: `GPHY-06`
- Title: Add reduced runtime regime specialization and hot-swap kernel cache in `fusion-core`
- Owner: Unassigned
- Proposed sprint: Phase 2, Sprint S2

## 2. Problem Statement

Fixed generic solver paths cannot represent regime-specific runtime specialization. This task adds a deterministic reduced JIT lane with explicit compile/cache/hot-swap semantics so control code can switch kernels by detected plasma regime without rebuilding binaries.

## 3. Target Paths

- Rust module(s):
  - `scpn-fusion-rs/crates/fusion-core/src/jit.rs`
  - `scpn-fusion-rs/crates/fusion-core/src/lib.rs`
- Validation path:
  - unit tests in `jit.rs`

## 4. Data and Dependency Readiness

- External dataset(s): none.
- License and usage confirmation: synthetic deterministic routing and actuation scope only.
- External dependencies: none (no external LLVM/inkwell in this reduced lane).
- Offline fallback path: if no active compiled kernel is available, execution is identity/stable.

## 5. Metrics and Acceptance Criteria

- Primary metrics:
  - deterministic regime detection from reduced observation bundle,
  - compile cache reuse for unchanged `(regime, spec)`,
  - hot-swap response change across regime transitions.
- Minimum acceptable threshold:
  - repeated compile requests for same `(regime, spec)` do not increment compile counter,
  - regime transitions produce different response dynamics,
  - no-active-kernel path remains bounded and identity-safe.
- Runtime budget:
  - `cargo test -p fusion-core` and `cargo clippy -p fusion-core --all-targets --all-features -- -D warnings` within normal CI budget.

## 6. Regression and Safety Plan

- CI jobs to update:
  - none required beyond existing Rust crate gates.
- Property/invariant tests:
  - cache hit reuse,
  - compile-event monotonicity,
  - active-regime correctness after hot-swap.
- Failure rollback strategy:
  - additive module; no default kernel path replacement.

## 7. Delivery Plan

- Milestone 1:
  - implement `PlasmaRegime`, `RegimeObservation`, and routing heuristic.
- Milestone 2:
  - implement `RuntimeKernelJit` compile/cache/activate API.
- Milestone 3:
  - add deterministic tests for cache reuse, hot-swap behavior, and identity fallback.

## 8. Go/No-Go Checklist

- [x] Path mapping reviewed
- [x] Data/license approved (synthetic deterministic scope)
- [x] Metrics protocol approved
- [x] CI cost estimated
- [ ] Owner committed
