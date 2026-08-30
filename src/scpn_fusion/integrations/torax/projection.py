# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Projection
"""Complete DataTree custody and typed critical TORAX projections."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np
import numpy.typing as npt

from .contracts import ToraxArtifact, ToraxProjection, ToraxRunRequest
from .serialization import canonical_json_bytes, canonical_sha256, file_sha256, write_json_atomic

FloatArray = npt.NDArray[np.float64]
MANIFEST_SCHEMA = "scpn-fusion-core.torax-datatree-manifest.v1"

_PROFILE_NAMES = {
    "ion_temperature": ("T_i", "keV"),
    "electron_temperature": ("T_e", "keV"),
    "electron_density": ("n_e", "m^-3"),
    "poloidal_flux": ("psi", "Wb/rad"),
}
_KNOWN_UNITS = {
    "time": "s",
    "rho_norm": "1",
    "rho_face_norm": "1",
    "rho_cell_norm": "1",
    "T_i": "keV",
    "T_e": "keV",
    "n_e": "m^-3",
    "psi": "Wb/rad",
    "p_generic_heat_i": "W/m^3",
    "p_generic_heat_e": "W/m^3",
    "s_generic_particle": "s^-1 m^-3",
    "j_generic_current": "A/m^2",
    "ei_exchange": "W/m^3",
    "outer_solver_iterations": "1",
    "inner_solver_iterations": "1",
    "sawtooth_crash": "1",
    "sim_error": "1",
    "sim_status": "1",
}


def write_complete_sidecar(data_tree: Any, path: Path) -> None:
    """Atomically persist every TORAX DataTree group and variable as NetCDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".nc.tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        data_tree.to_netcdf(temporary)
        if temporary.stat().st_size <= 0:
            raise ValueError("TORAX produced an empty NetCDF sidecar")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_manifest(data_tree: Any, sidecar_path: Path) -> dict[str, object]:
    """Enumerate and hash every DataTree group, coordinate, and data variable."""
    groups: list[dict[str, object]] = []
    total_variables = 0
    for group_path in data_tree.groups:
        dataset = data_tree[group_path].dataset
        variables: list[dict[str, object]] = []
        for name in sorted(dataset.variables):
            variable = dataset.variables[name]
            values = np.asarray(variable.values)
            unit = variable.attrs.get("units", _KNOWN_UNITS.get(name))
            unit_text = None if unit is None else str(unit)
            variables.append(
                {
                    "name": name,
                    "kind": "coordinate" if name in dataset.coords else "data_variable",
                    "dimensions": list(variable.dims),
                    "shape": list(values.shape),
                    "dtype": values.dtype.str,
                    "unit": unit_text,
                    "unit_status": "declared_or_canonical"
                    if unit_text is not None
                    else "not_declared_by_torax_1.4.3",
                    "values_sha256": _array_sha256(values),
                    "attributes_sha256": canonical_sha256(_json_attributes(variable.attrs)),
                }
            )
        total_variables += len(variables)
        groups.append(
            {
                "path": group_path,
                "dimensions": {name: int(size) for name, size in sorted(dataset.sizes.items())},
                "coordinates": sorted(str(name) for name in dataset.coords),
                "data_variables": sorted(str(name) for name in dataset.data_vars),
                "attributes_sha256": canonical_sha256(_json_attributes(dataset.attrs)),
                "variables": variables,
            }
        )
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "sidecar": {
            "sha256": file_sha256(sidecar_path),
            "bytes": sidecar_path.stat().st_size,
            "format": "NetCDF-DataTree",
        },
        "group_count": len(groups),
        "variable_count": total_variables,
        "groups": groups,
    }
    manifest["content_sha256"] = canonical_sha256(groups)
    manifest["inventory_sha256"] = canonical_sha256(
        {
            "schema": MANIFEST_SCHEMA,
            "group_count": len(groups),
            "variable_count": total_variables,
            "groups": groups,
        }
    )
    if len(groups) != len(tuple(data_tree.groups)):
        raise ValueError("DataTree group enumeration changed during manifest construction")
    return manifest


def publish_manifest(data_tree: Any, sidecar_path: Path, manifest_path: Path) -> ToraxArtifact:
    """Publish the complete manifest and return immutable artifact custody."""
    import xarray as xr

    in_memory_manifest = build_manifest(data_tree, sidecar_path)
    restored = xr.open_datatree(sidecar_path)
    try:
        manifest = build_manifest(restored, sidecar_path)
    finally:
        restored.close()
    if canonical_sha256(in_memory_manifest["groups"]) != canonical_sha256(manifest["groups"]):
        raise ValueError("persisted NetCDF DataTree differs from the in-memory TORAX output")
    write_json_atomic(manifest_path, manifest)
    sidecar = cast(Mapping[str, object], manifest["sidecar"])
    return ToraxArtifact(
        sidecar_path=str(sidecar_path),
        sidecar_sha256=cast(str, sidecar["sha256"]),
        sidecar_bytes=cast(int, sidecar["bytes"]),
        manifest_path=str(manifest_path),
        manifest_sha256=file_sha256(manifest_path),
    )


def build_projection(data_tree: Any, request: ToraxRunRequest) -> ToraxProjection:
    """Build the typed Ti/Te/ne/psi, source, budget, and numerical projection."""
    profiles = data_tree["/profiles"].dataset
    numerics_dataset = data_tree["/numerics"].dataset
    time_s = _finite_array(data_tree.dataset.coords["time"].values, "time")
    time_ns = _seconds_to_ns(time_s)
    rho = _finite_array(profiles.coords["rho_norm"].values, "rho_norm")
    typed_profiles: dict[str, tuple[tuple[float, ...], ...]] = {}
    profile_units: dict[str, str] = {}
    for public_name, (torax_name, unit) in _PROFILE_NAMES.items():
        values = _finite_array(profiles[torax_name].values, torax_name)
        if values.shape != (len(time_ns), len(rho)):
            raise ValueError(f"TORAX {torax_name} shape does not match time/rho_norm")
        typed_profiles[public_name] = tuple(tuple(float(item) for item in row) for row in values)
        profile_units[public_name] = unit
    source_totals = _source_totals(profiles, request)
    state_budgets = _state_budgets(profiles, rho, request)
    numerics: dict[str, object] = {
        "sim_status": str(np.asarray(numerics_dataset["sim_status"].values).item()),
        "sim_error": int(np.asarray(numerics_dataset["sim_error"].values).item()),
        "sawtooth_crash": [bool(item) for item in numerics_dataset["sawtooth_crash"].values],
        "outer_solver_iterations": [
            int(item) for item in numerics_dataset["outer_solver_iterations"].values
        ],
        "inner_solver_iterations": [
            int(item) for item in numerics_dataset["inner_solver_iterations"].values
        ],
    }
    projection_payload: dict[str, object] = {
        "clock_domain": request.clock.domain,
        "clock_epoch": request.clock.epoch,
        "time_ns": list(time_ns),
        "rho_norm": rho.tolist(),
        "rho_unit": "1",
        "rho_frame": request.geometry.frame,
        "profiles": {name: [list(row) for row in rows] for name, rows in typed_profiles.items()},
        "profile_units": profile_units,
        "source_totals": {name: list(values) for name, values in source_totals.items()},
        "source_units": _SOURCE_UNITS,
        "state_budgets": [dict(row) for row in state_budgets],
        "budget_units": _BUDGET_UNITS,
        "numerics": numerics,
        "uncertainty": {"kind": "not_evaluated", "basis": "single_run"},
    }
    return ToraxProjection(
        time_ns=time_ns,
        rho_norm=tuple(float(item) for item in rho),
        profiles=MappingProxyType(typed_profiles),
        profile_units=MappingProxyType(profile_units),
        source_totals=MappingProxyType(source_totals),
        source_units=MappingProxyType(dict(_SOURCE_UNITS)),
        state_budgets=tuple(MappingProxyType(dict(row)) for row in state_budgets),
        budget_units=MappingProxyType(dict(_BUDGET_UNITS)),
        numerics=MappingProxyType(numerics),
        uncertainty=MappingProxyType({"kind": "not_evaluated", "basis": "single_run"}),
        scientific_sha256=canonical_sha256(projection_payload),
    )


def reached_time_ns(data_tree: Any) -> int:
    """Return the exact integer-nanosecond final sample of a TORAX DataTree."""
    times = _seconds_to_ns(_finite_array(data_tree.dataset.coords["time"].values, "time"))
    return times[-1]


def _source_totals(dataset: Any, request: ToraxRunRequest) -> dict[str, tuple[float, ...]]:
    radius = request.geometry.minor_radius_m
    major_radius = request.geometry.major_radius_m

    def integrate(name: str, *, area: bool = False) -> tuple[float, ...]:
        variable = dataset[name]
        radial_dimension = variable.dims[-1]
        rho = _finite_array(variable.coords[radial_dimension].values, f"{name}.rho")
        spacing = np.diff(rho)
        if spacing.size == 0 or not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"TORAX source grid for {name} must be uniform")
        measure = (
            2.0 * np.pi * radius**2 * rho
            if area
            else 4.0 * np.pi**2 * major_radius * radius**2 * rho
        )
        values = _finite_array(variable.values, name)
        return tuple(float(np.sum(row * measure) * spacing[0]) for row in values)

    return {
        "ion_heat": integrate("p_generic_heat_i"),
        "electron_heat": integrate("p_generic_heat_e"),
        "particles": integrate("s_generic_particle"),
        "driven_current": integrate("j_generic_current", area=True),
        "ion_electron_exchange": integrate("ei_exchange"),
    }


def _state_budgets(
    dataset: Any,
    rho: FloatArray,
    request: ToraxRunRequest,
) -> tuple[dict[str, float], ...]:
    volume_derivative = (
        4.0 * np.pi**2 * request.geometry.major_radius_m * request.geometry.minor_radius_m**2 * rho
    )
    ti = _finite_array(dataset["T_i"].values, "T_i")
    te = _finite_array(dataset["T_e"].values, "T_e")
    ne = _finite_array(dataset["n_e"].values, "n_e")
    psi = _finite_array(dataset["psi"].values, "psi")
    kev_j = 1.602176634e-16
    budgets: list[dict[str, float]] = []
    for index in range(ti.shape[0]):
        thermal_density = 1.5 * ne[index] * kev_j * (ti[index] + te[index])
        budgets.append(
            {
                "thermal_energy": _trapezoid(thermal_density * volume_derivative, rho),
                "particle_inventory": _trapezoid(ne[index] * volume_derivative, rho),
                "poloidal_flux_l2": float(np.linalg.norm(psi[index])),
            }
        )
    return tuple(budgets)


def _trapezoid(values: FloatArray, coordinate: FloatArray) -> float:
    implementation = getattr(np, "trapezoid", None)
    if callable(implementation):
        return float(implementation(values, coordinate))
    return float(np.trapz(values, coordinate))


def _seconds_to_ns(values: FloatArray) -> tuple[int, ...]:
    scaled = values * 1_000_000_000.0
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, rtol=0.0, atol=1e-3):
        raise ValueError("TORAX time samples are not representable as exact integer nanoseconds")
    result = tuple(int(item) for item in rounded)
    if not result or any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("TORAX time samples must be strictly increasing")
    return result


def _finite_array(value: object, label: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"TORAX {label} must be non-empty and finite")
    return array


def _array_sha256(values: npt.NDArray[np.generic]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"dtype": values.dtype.str, "shape": list(values.shape)}))
    if values.dtype.kind in {"O", "S", "U"}:
        digest.update(canonical_json_bytes(values.tolist()))
    else:
        normalized = np.ascontiguousarray(values.astype(values.dtype.newbyteorder("<"), copy=False))
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _json_attributes(attributes: Mapping[object, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in attributes.items():
        if isinstance(value, np.generic):
            result[str(key)] = value.item()
        elif isinstance(value, np.ndarray):
            result[str(key)] = value.tolist()
        elif isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result


_SOURCE_UNITS = {
    "ion_heat": "W",
    "electron_heat": "W",
    "particles": "s^-1",
    "driven_current": "A",
    "ion_electron_exchange": "W",
}
_BUDGET_UNITS = {
    "thermal_energy": "J",
    "particle_inventory": "1",
    "poloidal_flux_l2": "Wb/rad",
}


__all__ = [
    "MANIFEST_SCHEMA",
    "build_manifest",
    "build_projection",
    "publish_manifest",
    "reached_time_ns",
    "write_complete_sidecar",
]
