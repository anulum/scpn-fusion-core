# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Test Fixtures
"""Deterministic fixtures for coil-vacuum grid-convergence tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any


from validation import ida_coil_vacuum_grid_contract as contract


def _digest(label: str) -> str:
    """Return a deterministic lowercase SHA-256 fixture value."""
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest() -> dict[str, Any]:
    """Return an internally consistent 18-parent/216-filament manifest."""
    parents: list[dict[str, Any]] = []
    for parent_index in range(18):
        name = f"FC{parent_index + 1}"
        filaments = [
            {
                "effective_current_a_turns": float(parent_index + 1) / 12.0,
                "filament_id": f"{name}:{filament_index:03d}",
                "filament_index": filament_index,
                "parent_index": parent_index,
                "parent_name": name,
                "r_m": 1.0 + 0.01 * filament_index,
                "weight": 1.0 / 12.0,
                "z_m": -0.5 + 0.01 * parent_index,
            }
            for filament_index in range(12)
        ]
        parents.append(
            {
                "coil_type": "ShapedCoil",
                "current_a": float(parent_index + 1),
                "effective_current_a_turns": float(parent_index + 1),
                "filament_count": 12,
                "filaments": filaments,
                "name": name,
                "parent_index": parent_index,
                "turns": 1.0,
            }
        )
    unsigned: dict[str, Any] = {
        "filament_count": 216,
        "parent_count": 18,
        "parents": parents,
    }
    return {
        **unsigned,
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                unsigned,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def _field_metric(label: str) -> dict[str, Any]:
    """Return one complete deterministic field-metric fixture."""
    return {
        "area_weighted_l2": 1.0,
        "area_weighted_rms": 0.1,
        "field_sha256": _digest(label),
        "l2": 10.0,
        "linf": 0.5,
    }


def _comparison_metric(*, relative_l2: float = 0.01) -> dict[str, float]:
    """Return one complete exact-restriction comparison fixture."""
    return {
        "area_weighted_l2": 0.1,
        "area_weighted_rms": 0.01,
        "cosine": 0.999,
        "linf": 0.02,
        "projection": 1.0,
        "relative_l2": relative_l2,
    }


def _recovery_metric(*, expected: float, recovered: float) -> dict[str, float]:
    """Return one internally consistent signed recovery fixture."""
    absolute = abs(recovered - expected)
    return {
        "absolute_error_a_turns": absolute,
        "expected_a_turns": expected,
        "recovered_a_turns": recovered,
        "relative_error": absolute / max(abs(expected), 1.0e-30),
        "signed_error_a_turns": recovered - expected,
    }


def _mask_summary(*, count: int, resolution: int, cell_area: float) -> dict[str, Any]:
    """Return one internally consistent mask summary fixture."""
    return {
        "area_m2": count * cell_area,
        "point_count": count,
        "point_fraction": count / (resolution * resolution),
    }


def _grid(
    resolution: int,
    *,
    recovery_error: float,
    localisation: float = 0.99,
) -> dict[str, Any]:
    """Return one structurally complete measured-grid fixture."""
    d_r = (contract.R_BOUNDS_M[1] - contract.R_BOUNDS_M[0]) / (resolution - 1)
    d_z = (contract.Z_BOUNDS_M[1] - contract.Z_BOUNDS_M[0]) / (resolution - 1)
    cell_area = d_r * d_z
    parent_rows = []
    expected_sum = 0.0
    primary_sum = 0.0
    for parent_index in range(18):
        name = f"FC{parent_index + 1}"
        expected = float(parent_index + 1)
        primary = expected * (1.0 - recovery_error)
        expected_sum += expected
        primary_sum += primary
        parent_rows.append(
            {
                "name": name,
                "regions": {
                    "fixed_physical": _recovery_metric(
                        expected=expected,
                        recovered=primary,
                    ),
                    "full_interior": _recovery_metric(
                        expected=expected,
                        recovered=expected,
                    ),
                    "primary": _recovery_metric(
                        expected=expected,
                        recovered=primary,
                    ),
                    "source_free": _recovery_metric(
                        expected=expected,
                        recovered=0.0,
                    ),
                },
            }
        )
    return {
        "current_recovery": {
            "aggregate": {
                "fixed_physical_a_turns": primary_sum,
                "full_interior_a_turns": expected_sum,
                "primary_a_turns": primary_sum,
                "source_free_a_turns": 0.0,
                "target_absolute_sum_a_turns": expected_sum,
                "target_net_a_turns": expected_sum,
            },
            "parents": parent_rows,
            "weighted_fixed_physical_error": recovery_error,
            "weighted_primary_error": recovery_error,
        },
        "forcing_partition": {
            "closure_max_abs": 0.0,
            "primary_l2_fraction": localisation,
            "sensitivity_l2_fraction": {
                "rho_h_le_1": max(0.0, localisation - 0.1),
                "rho_h_le_2": localisation,
                "rho_h_le_4": min(1.0, localisation + 0.005),
            },
            "source": _field_metric(f"forcing-source-{resolution}"),
            "source_free": _field_metric(f"forcing-source-free-{resolution}"),
            "total": _field_metric(f"forcing-total-{resolution}"),
        },
        "grid": {
            "cell_area_m2": cell_area,
            "d_r_m": d_r,
            "d_z_m": d_z,
            "filament_phase_sha256": _digest(f"phase-{resolution}"),
            "minimum_filament_to_node_distance_m": 0.01,
            "nearest_distance_max_m": 1.0,
            "nearest_distance_mean_m": 0.1,
            "r_bounds_m": list(contract.R_BOUNDS_M),
            "z_bounds_m": list(contract.Z_BOUNDS_M),
        },
        "masks": {
            "fixed_physical": _mask_summary(
                count=40,
                resolution=resolution,
                cell_area=cell_area,
            ),
            "fixed_physical_radius_m": contract.FIXED_PHYSICAL_RADIUS_M,
            "plasma_support": _mask_summary(
                count=50,
                resolution=resolution,
                cell_area=cell_area,
            ),
            "primary_fixed_overlap_point_count": 20,
            "rho_h_le_1": _mask_summary(
                count=10,
                resolution=resolution,
                cell_area=cell_area,
            ),
            "rho_h_le_2": _mask_summary(
                count=20,
                resolution=resolution,
                cell_area=cell_area,
            ),
            "rho_h_le_4": _mask_summary(
                count=30,
                resolution=resolution,
                cell_area=cell_area,
            ),
        },
        "resolution": resolution,
        "response_partition": {
            "closure_max_abs_wb": 1.0e-14,
            "source": _field_metric(f"response-source-{resolution}"),
            "source_free": _field_metric(f"response-source-free-{resolution}"),
            "total": _field_metric(f"response-total-{resolution}"),
        },
        "timings_ms": {
            "freegs_field": 1.0,
            "inverse_source": 2.0,
            "inverse_source_free": 2.0,
            "inverse_total": 2.0,
            "native_field": 1.0,
        },
        "vacuum_fields": {
            "freegs": _field_metric(f"vacuum-freegs-{resolution}"),
            "native": _field_metric(f"vacuum-native-{resolution}"),
            "source_free_parity": {
                "max_abs_wb": 1.0e-14,
                "nrmse": 1.0e-14,
                "passes": True,
                "point_count": 100,
            },
        },
    }


def _report(
    *,
    recovery_errors: tuple[float, float, float, float] = (0.04, 0.03, 0.02, 0.01),
    localisation: float = 0.99,
    observed_order: float = 2.0,
    response_relative_l2: float = 0.01,
) -> dict[str, Any]:
    """Build one fully signed report through the production contract."""
    source_artifacts: dict[str, dict[str, Any]] = {
        name: {"path": path, "sha256": _digest(name)}
        for name, path in contract.SOURCE_PATHS.items()
    }
    source_artifacts.update(
        {
            name: {
                "path": f"python-package://freegs/{name}.py",
                "sha256": _digest(name),
            }
            for name in contract.RUNTIME_SOURCE_NAMES
        }
    )
    source_artifacts["repository"] = {
        "git_commit": "a" * 40,
        "path": ".",
        "worktree_clean": True,
    }
    return contract.build_report(
        generated_at="2026-07-25T19:30:00Z",
        environment={
            "backend": "cpu",
            "x64_enabled": True,
        },
        source_artifacts=source_artifacts,
        bindings={
            name: {
                "path": contract.EXPECTED_BINDING_PATHS[name],
                "payload_sha256": payload,
            }
            for name, payload in contract.EXPECTED_PAYLOADS.items()
        },
        anchor={
            "forcing_sha256": contract.EXPECTED_ANCHOR_FORCING_SHA256,
            "response_closure_max_abs_wb": 2.1288526497187377e-14,
            "response_sha256": contract.EXPECTED_ANCHOR_RESPONSE_SHA256,
        },
        coil_manifest=_manifest(),
        grids=[
            _grid(
                resolution,
                recovery_error=error,
                localisation=localisation,
            )
            for resolution, error in zip(
                contract.GRID_RESOLUTIONS,
                recovery_errors,
                strict=True,
            )
        ],
        convergence={
            "finest_source_free_response": _comparison_metric(
                relative_l2=response_relative_l2,
            ),
            "pairwise": {
                pair_name: {
                    region_name: {
                        "forcing": _comparison_metric(),
                        "response": _comparison_metric(),
                    }
                    for region_name in (
                        "fixed_physical_source_free",
                        "full",
                        "plasma_support",
                        "source_footprint",
                    )
                }
                for pair_name in ("33_65", "65_129", "129_257")
            },
            "source_free_forcing_order": {
                name: {
                    "coarse_to_medium_rms": 4.0e-4,
                    "medium_to_fine_rms": 1.0e-4,
                    "observed_order": observed_order,
                }
                for name in contract.GRID_TRIPLES
            },
        },
    )


def _resign(report: dict[str, Any]) -> None:
    """Refresh the canonical payload digest after an intentional mutation."""
    report["payload_sha256"] = contract._payload_sha256(report)


def _resign_manifest(report: dict[str, Any]) -> None:
    """Refresh both manifest and enclosing report digests after mutation."""
    manifest = report["coil_manifest"]
    unsigned = {name: item for name, item in manifest.items() if name != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    _resign(report)
