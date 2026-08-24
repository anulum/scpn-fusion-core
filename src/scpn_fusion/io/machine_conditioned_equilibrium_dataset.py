# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned Equilibrium Dataset Contract
"""Versioned storage and verification for machine-conditioned equilibria."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.io.safe_loaders import checked_json_load, checked_np_load

SCHEMA_VERSION = "scpn-fusion.machine-conditioned-equilibrium-dataset.v2"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ARRAY_BYTES = 64 * 1024 * 1024 * 1024
REQUIRED_ARRAYS = frozenset(
    {
        "inputs",
        "psi_total",
        "psi_vacuum",
        "diagnostics",
        "grid_r_m",
        "grid_z_m",
        "wall_rz_m",
        "limiter_rz_m",
        "plasma_support_weights",
        "coil_green_psi_per_current",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POST_SOLVE_INPUT_TOKENS = (
    "psi_axis",
    "psi_x",
    "r_axis",
    "z_axis",
    "r_x",
    "z_x",
    "residual",
    "iteration",
    "converged",
)
FloatArray: TypeAlias = NDArray[np.float64]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_contract(path: Path, array: FloatArray, *, role: str) -> dict[str, Any]:
    """Build a manifest contract for one already-written ``float64`` NPY file.

    Parameters
    ----------
    path : Path
        NPY file whose bytes are bound by the contract.
    array : FloatArray
        In-memory array written to ``path``.
    role : str
        Stable semantic role of the array.

    Returns
    -------
    dict[str, Any]
        JSON-compatible file, shape, dtype, size and digest contract.
    """
    return {
        "file": path.name,
        "role": role,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "logical_bytes": int(array.size * array.dtype.itemsize),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _failure_result(failures: list[str], *, dataset_id: str | None = None) -> dict[str, Any]:
    """Return the stable machine-readable failure envelope."""
    return {
        "status": "failed",
        "dataset_id": dataset_id,
        "failures": failures,
    }


def _as_dict(value: Any, *, name: str, failures: list[str]) -> dict[str, Any]:
    """Return a JSON object or append a named contract failure."""
    if not isinstance(value, dict):
        failures.append(f"{name} must be an object")
        return {}
    return cast(dict[str, Any], value)


def _array_path(root: Path, spec: dict[str, Any], *, key: str, failures: list[str]) -> Path:
    """Resolve one manifest-declared basename without permitting traversal."""
    raw_name = spec.get("file")
    if not isinstance(raw_name, str) or not raw_name:
        failures.append(f"arrays.{key}.file must be a non-empty basename")
        return root / f"__invalid_{key}"
    if Path(raw_name).name != raw_name or "/" in raw_name or "\\" in raw_name:
        failures.append(f"arrays.{key}.file must not contain a path")
        return root / f"__invalid_{key}"
    return root / raw_name


def _validate_declared_array(
    root: Path,
    key: str,
    spec: dict[str, Any],
    failures: list[str],
) -> FloatArray | None:
    """Authenticate and load one bounded NPY array from a manifest contract."""
    path = _array_path(root, spec, key=key, failures=failures)
    if not path.is_file() or path.is_symlink():
        failures.append(f"missing regular array file: {path.name}")
        return None

    shape = spec.get("shape")
    dtype_name = spec.get("dtype")
    size_bytes = spec.get("size_bytes")
    logical_bytes = spec.get("logical_bytes")
    digest = spec.get("sha256")
    if (
        not isinstance(shape, list)
        or not shape
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in shape)
    ):
        failures.append(f"arrays.{key}.shape is invalid")
        return None
    if dtype_name != "float64":
        failures.append(f"arrays.{key}.dtype must be float64")
        return None
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        failures.append(f"arrays.{key}.size_bytes is invalid")
        return None
    if isinstance(logical_bytes, bool) or not isinstance(logical_bytes, int) or logical_bytes < 0:
        failures.append(f"arrays.{key}.logical_bytes is invalid")
        return None
    expected_logical = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float64).itemsize
    if logical_bytes != expected_logical:
        failures.append(f"arrays.{key}.logical_bytes does not match shape")
        return None
    if size_bytes > MAX_ARRAY_BYTES or logical_bytes > MAX_ARRAY_BYTES:
        failures.append(f"arrays.{key} exceeds the {MAX_ARRAY_BYTES}-byte hard limit")
        return None
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        failures.append(f"arrays.{key}.sha256 is invalid")
        return None
    if path.stat().st_size != size_bytes:
        failures.append(f"size mismatch for {path.name}")
        return None
    if sha256_file(path) != digest:
        failures.append(f"SHA-256 mismatch for {path.name}")
        return None
    try:
        loaded = checked_np_load(
            path,
            max_bytes=size_bytes,
            max_member_bytes=logical_bytes,
            max_total_bytes=logical_bytes,
            mmap_mode="r",
        )
    except (OSError, ValueError) as exc:
        failures.append(f"cannot load {path.name}: {exc}")
        return None
    array = np.asarray(loaded)
    if list(array.shape) != shape:
        failures.append(f"shape mismatch for {path.name}")
        return None
    if array.dtype != np.dtype(np.float64):
        failures.append(f"dtype mismatch for {path.name}")
        return None
    return cast(FloatArray, array)


def _all_finite(array: FloatArray, *, chunk_rows: int = 64) -> bool:
    """Check finiteness in bounded leading-axis chunks."""
    if array.ndim == 0:
        return bool(np.isfinite(array))
    for start in range(0, len(array), chunk_rows):
        if not np.all(np.isfinite(array[start : start + chunk_rows])):
            return False
    return True


def _validate_geometry(arrays: dict[str, FloatArray], failures: list[str]) -> None:
    """Validate grid monotonicity, closed contours and static plasma support."""
    r = arrays["grid_r_m"]
    z = arrays["grid_z_m"]
    if r.ndim != 1 or len(r) < 4 or not np.all(np.diff(r) > 0.0):
        failures.append("grid_r_m must be a strictly increasing vector with at least four points")
    if z.ndim != 1 or len(z) < 4 or not np.all(np.diff(z) > 0.0):
        failures.append("grid_z_m must be a strictly increasing vector with at least four points")
    for key in ("wall_rz_m", "limiter_rz_m"):
        contour = arrays[key]
        if contour.ndim != 2 or contour.shape[1:] != (2,) or len(contour) < 4:
            failures.append(f"{key} must have shape (n_points, 2) with at least four points")
        elif not np.array_equal(contour[0], contour[-1]):
            failures.append(f"{key} must be explicitly closed")
    support = arrays["plasma_support_weights"]
    if support.shape != (len(z), len(r)):
        failures.append("plasma_support_weights must match the Z-R grid")
    elif np.any((support < 0.0) | (support > 1.0)):
        failures.append("plasma_support_weights must lie in [0, 1]")
    elif float(np.ptp(support)) <= 0.5:
        failures.append("plasma_support_weights must contain distinct core and exterior regions")


def _validate_feature_contract(
    manifest: dict[str, Any],
    inputs: FloatArray,
    failures: list[str],
) -> list[dict[str, Any]]:
    """Validate leakage-free per-sample feature declarations."""
    raw_features = manifest.get("features")
    if not isinstance(raw_features, list):
        failures.append("features must be an array")
        return []
    features: list[dict[str, Any]] = []
    names: list[str] = []
    for index, raw in enumerate(raw_features):
        feature = _as_dict(raw, name=f"features[{index}]", failures=failures)
        name = feature.get("name")
        if not isinstance(name, str) or not name:
            failures.append(f"features[{index}].name must be non-empty")
            continue
        if feature.get("index") != index:
            failures.append(f"features[{index}].index must equal {index}")
        if feature.get("role") != "pre_solve_control":
            failures.append(f"feature {name} is not declared as a pre-solve control")
        unit = feature.get("unit")
        if not isinstance(unit, str) or not unit:
            failures.append(f"feature {name} must declare a unit")
        minimum = feature.get("minimum")
        maximum = feature.get("maximum")
        if (
            not isinstance(minimum, (int, float))
            or isinstance(minimum, bool)
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or float(minimum) >= float(maximum)
        ):
            failures.append(f"feature {name} has an invalid sampling range")
        elif inputs.ndim == 2 and index < inputs.shape[1]:
            if np.any((inputs[:, index] < float(minimum)) | (inputs[:, index] > float(maximum))):
                failures.append(f"feature {name} contains a value outside its declared range")
        if any(token in name.lower() for token in _POST_SOLVE_INPUT_TOKENS):
            failures.append(f"post-solve feature is forbidden in inputs: {name}")
        names.append(name)
        features.append(feature)
    if len(set(names)) != len(names):
        failures.append("feature names must be unique")
    if inputs.ndim != 2 or inputs.shape[1] != len(raw_features):
        failures.append("inputs feature count does not match manifest")
    elif inputs.shape[0] > 1:
        for index, name in enumerate(names):
            if np.unique(inputs[:, index]).size < 2:
                failures.append(
                    f"per-sample feature is constant and belongs in machine metadata: {name}"
                )
    return features


def _validate_diagnostics(
    manifest: dict[str, Any],
    diagnostics: FloatArray,
    arrays: dict[str, FloatArray],
    features: list[dict[str, Any]],
    failures: list[str],
) -> None:
    """Validate accepted-sample convergence and interior topology diagnostics."""
    raw_names = manifest.get("diagnostic_names")
    if not isinstance(raw_names, list) or not all(isinstance(name, str) for name in raw_names):
        failures.append("diagnostic_names must be an array of strings")
        return
    names = cast(list[str], raw_names)
    if len(set(names)) != len(names):
        failures.append("diagnostic_names must be unique")
        return
    if diagnostics.ndim != 2 or diagnostics.shape[1] != len(names):
        failures.append("diagnostics column count does not match manifest")
        return
    required = {
        "candidate_index",
        "converged",
        "iterations",
        "relative_gs_residual_rms",
        "plasma_current_actual_a",
        "plasma_current_relative_error",
        "psi_axis",
        "psi_x",
        "psi_span_abs_wb",
        "plasma_delta_max_abs_wb",
    }
    if not required.issubset(names):
        failures.append(f"diagnostic_names missing {sorted(required - set(names))}")
        return
    index = {name: names.index(name) for name in names}
    candidate = diagnostics[:, index["candidate_index"]]
    if (
        not np.array_equal(candidate, np.floor(candidate))
        or np.any(candidate < 0.0)
        or np.any(np.diff(candidate) <= 0.0)
    ):
        failures.append("candidate_index must be a strictly increasing integer sequence")
    if not np.all(diagnostics[:, index["converged"]] == 1.0):
        failures.append("accepted diagnostics contain a non-converged sample")
    iterations = diagnostics[:, index["iterations"]]
    if np.any(iterations < 1.0) or not np.array_equal(iterations, np.floor(iterations)):
        failures.append("diagnostic iteration counts must be positive integers")
    solver = _as_dict(manifest.get("solver"), name="solver", failures=failures)
    iteration_cap = solver.get("n_iter")
    if isinstance(iteration_cap, bool) or not isinstance(iteration_cap, int) or iteration_cap < 1:
        failures.append("solver.n_iter must be a positive integer")
    elif np.any(iterations >= iteration_cap):
        failures.append("accepted diagnostics include a solve that reached the iteration cap")

    tolerance = _as_dict(manifest.get("tolerances"), name="tolerances", failures=failures)
    threshold_columns = {
        "relative_gs_residual_rms": ("relative_gs_residual_rms_max", np.less_equal),
        "plasma_current_relative_error": ("plasma_current_relative_error_max", np.less_equal),
        "psi_span_abs_wb": ("psi_span_abs_wb_min", np.greater_equal),
        "plasma_delta_max_abs_wb": ("plasma_delta_max_abs_wb_min", np.greater_equal),
    }
    for column, (threshold_name, comparison) in threshold_columns.items():
        threshold = tolerance.get(threshold_name)
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not np.isfinite(threshold)
            or threshold < 0.0
        ):
            failures.append(f"tolerances.{threshold_name} must be finite and non-negative")
            continue
        if not np.all(comparison(diagnostics[:, index[column]], float(threshold))):
            failures.append(f"accepted diagnostics violate {threshold_name}")
    feature_names = [str(feature.get("name")) for feature in features]
    if "plasma_current_target_a" not in feature_names:
        failures.append("inputs must include plasma_current_target_a")
    else:
        target = arrays["inputs"][:, feature_names.index("plasma_current_target_a")]
        actual = diagnostics[:, index["plasma_current_actual_a"]]
        observed = np.abs(actual - target) / np.maximum(np.abs(target), 1.0)
        declared = tolerance.get("plasma_current_relative_error_max")
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            if np.any(observed > float(declared)):
                failures.append("stored plasma current does not close against target input")
    replay_tolerance = tolerance.get("diagnostic_replay_max_abs")
    if (
        not isinstance(replay_tolerance, (int, float))
        or isinstance(replay_tolerance, bool)
        or not np.isfinite(replay_tolerance)
        or replay_tolerance <= 0.0
    ):
        failures.append("tolerances.diagnostic_replay_max_abs must be finite and positive")
    else:
        stored_delta = diagnostics[:, index["plasma_delta_max_abs_wb"]]
        for start in range(0, len(diagnostics), 64):
            stop = min(start + 64, len(diagnostics))
            replayed_delta = np.max(
                np.abs(arrays["psi_total"][start:stop] - arrays["psi_vacuum"][start:stop]),
                axis=(1, 2),
            )
            if np.any(np.abs(replayed_delta - stored_delta[start:stop]) > float(replay_tolerance)):
                failures.append("plasma_delta_max_abs_wb does not replay from stored fields")
                break


def _validate_vacuum_reconstruction(
    manifest: dict[str, Any],
    arrays: dict[str, FloatArray],
    features: list[dict[str, Any]],
    failures: list[str],
) -> float:
    """Check that stored coil currents and unit Green maps reconstruct vacuum psi."""
    machine = _as_dict(manifest.get("machine"), name="machine", failures=failures)
    raw_coils = machine.get("coils")
    if not isinstance(raw_coils, list) or not raw_coils:
        failures.append("machine.coils must be a non-empty array")
        return float("inf")
    coils = [
        _as_dict(item, name=f"machine.coils[{i}]", failures=failures)
        for i, item in enumerate(raw_coils)
    ]
    names = [coil.get("name") for coil in coils]
    if not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        failures.append("machine coil names must be unique non-empty strings")
        return float("inf")
    solver = _as_dict(manifest.get("solver"), name="solver", failures=failures)
    if solver.get("coil_field_decomposition") is not False:
        failures.append("v2 fixed-support datasets require solver.coil_field_decomposition=false")
    r = arrays["grid_r_m"]
    z = arrays["grid_z_m"]
    for coil in coils:
        r_m = coil.get("r_m")
        z_m = coil.get("z_m")
        if not isinstance(r_m, (int, float)) or not isinstance(z_m, (int, float)):
            failures.append(f"coil {coil.get('name')} coordinates must be numeric")
        elif float(r[0]) <= float(r_m) <= float(r[-1]) and float(z[0]) <= float(z_m) <= float(
            z[-1]
        ):
            failures.append(
                f"coil {coil.get('name')} lies inside the grid without field decomposition"
            )
    feature_names = [str(feature.get("name")) for feature in features]
    current_indices: list[int] = []
    for coil in coils:
        current_feature = coil.get("current_feature")
        turns = coil.get("turns")
        if current_feature not in feature_names:
            failures.append(f"coil {coil.get('name')} current_feature is absent from inputs")
            continue
        if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
            failures.append(f"coil {coil.get('name')} turns must be a positive integer")
        current_indices.append(feature_names.index(cast(str, current_feature)))
    green = arrays["coil_green_psi_per_current"]
    inputs = arrays["inputs"]
    vacuum = arrays["psi_vacuum"]
    expected_spatial = (len(arrays["grid_z_m"]), len(arrays["grid_r_m"]))
    if green.ndim != 3 or green.shape[0] != len(coils) or green.shape[1:] != expected_spatial:
        failures.append("coil Green array does not match machine coil count")
        return float("inf")
    if len(current_indices) != len(coils):
        return float("inf")
    tolerance = _as_dict(manifest.get("tolerances"), name="tolerances", failures=failures)
    raw_atol = tolerance.get("vacuum_reconstruction_max_abs")
    if not isinstance(raw_atol, (int, float)) or isinstance(raw_atol, bool) or raw_atol <= 0.0:
        failures.append("tolerances.vacuum_reconstruction_max_abs must be positive")
        return float("inf")
    atol = float(raw_atol)
    max_abs = 0.0
    for start in range(0, len(inputs), 32):
        currents = inputs[start : start + 32, current_indices]
        expected = np.einsum("sc,czr->szr", currents, green, optimize=True)
        difference = float(np.max(np.abs(expected - vacuum[start : start + 32])))
        max_abs = max(max_abs, difference)
        if difference > atol:
            failures.append(
                f"stored vacuum fields exceed Green reconstruction tolerance: {difference} > {atol}"
            )
            break
    return max_abs


def verify_machine_conditioned_dataset(
    dataset_dir: str | Path,
    *,
    full_field_scan: bool = True,
) -> dict[str, Any]:
    """Verify one machine-conditioned dataset directory fail closed.

    Parameters
    ----------
    dataset_dir : str | Path
        Directory containing ``manifest.json`` and every declared NPY array.
    full_field_scan : bool, optional
        Scan every field value for finiteness. Keep enabled for publication and
        custody verification; callers may disable it for a fast metadata check.

    Returns
    -------
    dict[str, Any]
        Machine-readable status, dataset identity, failures and Green-map error.
    """
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    failures: list[str] = []
    if not root.is_dir() or root.is_symlink() or not manifest_path.is_file():
        return _failure_result([f"dataset directory or manifest is missing: {root}"])
    try:
        manifest = _as_dict(
            checked_json_load(manifest_path, max_bytes=MAX_MANIFEST_BYTES),
            name="manifest",
            failures=failures,
        )
    except (OSError, ValueError) as exc:
        return _failure_result([f"cannot load manifest: {exc}"])
    dataset_id = manifest.get("dataset_id") if isinstance(manifest.get("dataset_id"), str) else None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"schema_version must be {SCHEMA_VERSION}")
    if dataset_id is None or not dataset_id:
        failures.append("dataset_id must be a non-empty string")
    claims = _as_dict(manifest.get("claims"), name="claims", failures=failures)
    for flag in ("facility_validated", "experimental_shot_data", "time_series_prediction"):
        if claims.get(flag) is not False:
            failures.append(f"claims.{flag} must be false for this synthetic dataset")
    limits = claims.get("limits")
    if (
        not isinstance(limits, list)
        or not limits
        or not all(isinstance(item, str) for item in limits)
    ):
        failures.append("claims.limits must be a non-empty string array")

    raw_specs = _as_dict(manifest.get("arrays"), name="arrays", failures=failures)
    if set(raw_specs) != REQUIRED_ARRAYS:
        failures.append(
            f"arrays must declare exactly {sorted(REQUIRED_ARRAYS)}; got {sorted(raw_specs)}"
        )
    arrays: dict[str, FloatArray] = {}
    declared_files = {"manifest.json"}
    for key in sorted(REQUIRED_ARRAYS & set(raw_specs)):
        spec = _as_dict(raw_specs[key], name=f"arrays.{key}", failures=failures)
        path = _array_path(root, spec, key=key, failures=failures)
        declared_files.add(path.name)
        array = _validate_declared_array(root, key, spec, failures)
        if array is not None:
            arrays[key] = array
    actual_files = {path.name for path in root.iterdir() if path.is_file() or path.is_symlink()}
    if actual_files != declared_files:
        failures.append(
            f"dataset file inventory mismatch; extra={sorted(actual_files - declared_files)}, "
            f"missing={sorted(declared_files - actual_files)}"
        )
    if set(arrays) != REQUIRED_ARRAYS:
        return _failure_result(failures, dataset_id=dataset_id)

    for key, array in arrays.items():
        if key in {"psi_total", "psi_vacuum"} and not full_field_scan:
            rows = array[:1] if len(array) else array
            if not np.all(np.isfinite(rows)):
                failures.append(f"{key} contains non-finite values in the sampled scan")
        elif not _all_finite(array):
            failures.append(f"{key} contains non-finite values")
    inputs = arrays["inputs"]
    total = arrays["psi_total"]
    vacuum = arrays["psi_vacuum"]
    diagnostics = arrays["diagnostics"]
    r = arrays["grid_r_m"]
    z = arrays["grid_z_m"]
    sample_count = len(inputs) if inputs.ndim > 0 else 0
    if sample_count < 1:
        failures.append("dataset must contain at least one sample")
    expected_field_shape = (sample_count, len(z), len(r))
    if total.shape != expected_field_shape or vacuum.shape != expected_field_shape:
        failures.append("psi arrays must have shape (sample, len(grid_z_m), len(grid_r_m))")
    if diagnostics.ndim != 2 or len(diagnostics) != sample_count:
        failures.append("diagnostics sample count must match inputs")
    generation = _as_dict(manifest.get("generation"), name="generation", failures=failures)
    if generation.get("accepted_samples") != sample_count:
        failures.append("generation.accepted_samples must match inputs")

    _validate_geometry(arrays, failures)
    if failures:
        return _failure_result(failures, dataset_id=dataset_id)
    features = _validate_feature_contract(manifest, inputs, failures)
    _validate_diagnostics(manifest, diagnostics, arrays, features, failures)
    green_error = _validate_vacuum_reconstruction(manifest, arrays, features, failures)
    return {
        "status": "passed" if not failures else "failed",
        "dataset_id": dataset_id,
        "samples_verified": sample_count,
        "full_field_scan": full_field_scan,
        "vacuum_reconstruction_max_abs": green_error,
        "failures": failures,
    }


__all__ = [
    "MAX_ARRAY_BYTES",
    "MAX_MANIFEST_BYTES",
    "REQUIRED_ARRAYS",
    "SCHEMA_VERSION",
    "array_contract",
    "sha256_file",
    "verify_machine_conditioned_dataset",
]
