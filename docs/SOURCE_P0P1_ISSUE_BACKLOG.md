# Source P0/P1 Issue Backlog

- Generated at: `2026-03-04T13:04:19.248729+00:00`
- Generator: `tools/generate_source_p0p1_issue_backlog.py`
- Scope: source files only (`src/scpn_fusion/**`) with P0/P1 severity

## Summary

| Metric | Value |
|---|---:|
| Source issue seeds | 1 |
| P0 seeds | 1 |
| P1 seeds | 0 |
| Domains represented | 1 |

## Marker Distribution

| Marker | Count |
|---|---:|
| `FALLBACK_DENSITY` | 1 |

## Domain Distribution

| Domain | Count |
|---|---:|
| `control` | 1 |

## Auto-generated Issue Seeds

_Each section below is ready to open as a GitHub issue with owner hints and closure criteria._

### 1. [P0] Harden `src/scpn_fusion/control/disruption_predictor.py`

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
