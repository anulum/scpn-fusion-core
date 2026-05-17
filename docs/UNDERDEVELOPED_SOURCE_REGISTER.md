# Underdeveloped Register

- Generated at: `2026-05-17T22:58:18.727560+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 7 |
| P0 + P1 entries | 0 |
| Source-domain entries | 7 |
| Source-domain P0 + P1 entries | 0 |
| Docs-claims entries | 0 |
| Domains affected | 2 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 7 |

## Domain Distribution

| Key | Count |
|---|---:|
| `control` | 4 |
| `core_physics` | 3 |

## Top Priority Backlog (Top 7)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_supervisory_types.py:22` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_tracking_control.py:171` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback currents are not configured.") |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_tracking_shot.py:276` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | f"lag={max_abs_actuator_lag:.3e} \| fallback={fallback_active} \| " |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:13` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | rejection can ramp the coil set toward explicit safe fallback currents instead |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/disruption_sequence.py:286` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | seed = 1e10 # Fallback |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gk_tglf.py:204` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | _logger.warning("TGLF binary not found, returning fallback") |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/jax_gs_solver.py:24` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | NumPy fallback provided when JAX is unavailable. |

## Full Register (Top 7)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_supervisory_types.py:22` | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_tracking_control.py:171` | raise ValueError("fallback currents are not configured.") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_tracking_shot.py:276` | f"lag={max_abs_actuator_lag:.3e} \| fallback={fallback_active} \| " |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:13` | rejection can ramp the coil set toward explicit safe fallback currents instead |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/disruption_sequence.py:286` | seed = 1e10 # Fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gk_tglf.py:204` | _logger.warning("TGLF binary not found, returning fallback") |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/jax_gs_solver.py:24` | NumPy fallback provided when JAX is unavailable. |
