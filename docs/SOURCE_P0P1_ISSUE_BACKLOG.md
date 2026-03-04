# Source P0/P1 Issue Backlog

- Generated at: `2026-03-04T03:07:20.163415+00:00`
- Generator: `tools/generate_source_p0p1_issue_backlog.py`
- Scope: source files only (`src/scpn_fusion/**`) with P0/P1 severity

## Summary

| Metric | Value |
|---|---:|
| Source issue seeds | 7 |
| P0 seeds | 7 |
| P1 seeds | 0 |
| Domains represented | 2 |

## Marker Distribution

| Marker | Count |
|---|---:|
| `MONOLITH` | 6 |
| `FALLBACK_DENSITY` | 1 |

## Domain Distribution

| Domain | Count |
|---|---:|
| `core_physics` | 6 |
| `control` | 1 |

## Auto-generated Issue Seeds

_Each section below is ready to open as a GitHub issue with owner hints and closure criteria._

### 1. [P0] Harden `src/scpn_fusion/core/fno_training.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 2. [P0] Harden `src/scpn_fusion/core/fusion_kernel.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 3. [P0] Harden `src/scpn_fusion/core/fusion_kernel_newton_solver.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 4. [P0] Harden `src/scpn_fusion/core/neural_equilibrium.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 5. [P0] Harden `src/scpn_fusion/core/neural_transport.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 6. [P0] Harden `src/scpn_fusion/core/tglf_interface.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `core_physics`
- **Owner Hint**: Core Physics WG
- **Priority Score**: `101`
- **Markers**: `MONOLITH`
- **Trigger Lines**: `1`

**Proposed Actions**
- Split module into focused subcomponents and lock interface contracts.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 70.0% (tools/coverage_guard.py).
- [ ] At least one high-risk function path is extracted behind a unit-tested helper or submodule boundary.

### 7. [P0] Harden `src/scpn_fusion/control/disruption_predictor.py`

- **Labels**: `hardening`, `underdeveloped`, `p0`, `control`
- **Owner Hint**: Control WG
- **Priority Score**: `96`
- **Markers**: `FALLBACK_DENSITY`
- **Trigger Lines**: `1`

**Proposed Actions**
- Reduce fallback concentration and enforce strict-backend parity checks.

**Acceptance Checklist**
- [ ] Add or tighten regression tests for this module path and update coverage baselines.
- [ ] Update claim/evidence references if behavior or metrics change.

**Closure Metrics**
- [ ] Module no longer appears in docs/SOURCE_P0P1_ISSUE_BACKLOG after register regeneration.
- [ ] File line coverage in release lane is >= 78.0% (tools/coverage_guard.py).
