#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — pinned FreeGSNKE inverse-equilibrium comparison
"""Run a pinned MAST-U-like FreeGSNKE inverse solve and audit the SCPN bridge.

This is a bounded comparison, not a facility-validation claim.  It executes the
upstream FreeGSNKE 3.0.1 diverted inverse problem, verifies its currents, total
flux, topology, and current limits against the upstream regression baselines,
then maps the same resolved circuit currents and every active-coil filament into
SCPN's public differentiable SI Green-function surface.  The SCPN/FreeGSNKE
vacuum-flux comparison is evaluated inside the limiter, away from the deliberate
filament self-field regularisation.  A JAX gradient of a fixed vacuum-flux
objective is checked independently by central finite differences.

The converged FreeGSNKE ``ConstrainPaxisIp`` profile is also sampled on SCPN's
public ``pprime``/``FFprime`` surface.  The comparison freezes normalised-flux
orientation, exact FreeGSNKE LCFS support, gauge invariance, and the shared
COCOS-3 flux-per-radian convention.  It admits the profile-source translation,
but not a separately executed self-consistent SCPN total-flux solve.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
# Only an exact pinned and digest-bound public upstream file is loaded.
import pickle  # nosec B403
import shutil
# Fixed git argv, no shell, bounded timeout.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, cast

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from freegs4e.critical import find_critical
from numpy.typing import NDArray

from freegsnke import GSstaticsolver, build_machine, equilibrium_update
from freegsnke.inverse import Inverse_optimizer
from freegsnke.jtor_update import ConstrainPaxisIp
from scpn_fusion.core.imas_equilibrium_io import DEFAULT_SOLVER_COCOS
from scpn_fusion.core.jax_free_boundary_gs import (
    MU0_SI,
    general_gs_source,
    normalised_flux_unclipped,
    vacuum_field_si,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/external/full_fidelity_public_sources/repos/freegsnke"
REPORT_PATH = ROOT / "validation/reports/freegsnke_inverse_comparison.json"
MARKDOWN_PATH = ROOT / "validation/reports/freegsnke_inverse_comparison.md"

SCHEMA_VERSION = "scpn-fusion.freegsnke-inverse-comparison.v2"
BENCHMARK_ID = "F-1c-freegsnke-mastu-diverted-inverse"
UPSTREAM_COMMIT = "00b79a08fbb6e642ac5ed2e440383cc5962e0358"
UPSTREAM_VERSION = "3.0.1"
FREEGS4E_VERSION = "0.13.1"
EXPECTED_RUNTIME_VERSIONS = {
    "jax": "0.7.1",
    "jaxlib": "0.7.1",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
}

CURRENT_ATOL_A = 5.0e-3
PSI_SPAN_ATOL_FRACTION = 3.0e-3
TOPOLOGY_ATOL_M = 5.0e-4
PASSIVE_CURRENT_ATOL_A = 1.0e-12
VACUUM_LIMITER_MAX_ABS_WB = 1.0e-10
GRADIENT_RELATIVE_ERROR_MAX = 1.0e-7
FINITE_DIFFERENCE_STEP_A = 1.0
PROFILE_SAMPLE_COUNT = 8193
PROFILE_SOURCE_RELATIVE_L2_MAX = 1.0e-7
PROFILE_TOTAL_CURRENT_RELATIVE_ERROR_MAX = 1.0e-7
PSIN_MAX_ABS_ERROR = 1.0e-14
GAUGE_SOURCE_RELATIVE_L2_MAX = 1.0e-12
GAUGE_OFFSET_WB_PER_RADIAN = 17.25

EXPECTED_AXIS_M = np.array([0.951053009, 0.0], dtype=np.float64)
EXPECTED_XPOINTS_M = np.array(
    [[0.59848009, -1.09716935], [0.59848008, 1.09716927]],
    dtype=np.float64,
)
CLAIM_BOUNDARY = {
    "control_admission": False,
    "facility_validation": False,
    "pcs_deployment": False,
    "safety_admission": False,
    "total_psi_cross_solver_parity": False,
}

FloatArray = NDArray[np.float64]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: object) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    descriptor = _canonical_json(
        {"dtype": "<f8", "shape": [int(size) for size in array.shape]}
    )
    return hashlib.sha256(descriptor + b"\0" + array.tobytes(order="C")).hexdigest()


def _git_head(source: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to authenticate the FreeGSNKE source checkout")
    completed = subprocess.run(
        [git, "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )  # nosec B603
    return completed.stdout.strip()


def validate_source(source: Path) -> dict[str, str]:
    """Fail closed unless the exact upstream source and runtime pins are present."""
    if not source.is_dir():
        raise FileNotFoundError(f"FreeGSNKE source checkout not found: {source}")
    head = _git_head(source)
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"FreeGSNKE source is {head}, expected {UPSTREAM_COMMIT}")
    versions = {
        "freegs4e": importlib.metadata.version("freegs4e"),
        "freegsnke": importlib.metadata.version("freegsnke"),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    if versions["freegsnke"] != UPSTREAM_VERSION:
        raise RuntimeError(
            f"FreeGSNKE runtime is {versions['freegsnke']}, expected {UPSTREAM_VERSION}"
        )
    if versions["freegs4e"] != FREEGS4E_VERSION:
        raise RuntimeError(
            f"FreeGS4E runtime is {versions['freegs4e']}, expected {FREEGS4E_VERSION}"
        )
    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        if versions[package] != expected:
            raise RuntimeError(
                f"{package} runtime is {versions[package]}, expected hash-locked {expected}"
            )
    return {"commit": head, **versions}


def _case_paths(source: Path) -> dict[str, Path]:
    machine = source / "machine_configs/MAST-U"
    baselines = source / "freegsnke/tests/baselines"
    return {
        "active_coils": machine / "MAST-U_like_active_coils.pickle",
        "passive_coils": machine / "MAST-U_like_passive_coils.pickle",
        "limiter": machine / "MAST-U_like_limiter.pickle",
        "wall": machine / "MAST-U_like_wall.pickle",
        "current_baseline": baselines / "test_inverse_control_currents.npy",
        "psi_baseline": baselines / "test_inverse_psi.npy",
        "upstream_test": source / "freegsnke/tests/test_inverse_static_solver.py",
    }


def build_inverse_case(source: Path) -> tuple[Any, Any, Any, list[list[float | None]]]:
    """Build the exact public case encoded by FreeGSNKE's pinned regression test."""
    paths = _case_paths(source)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned case artifacts: {missing}")

    tokamak = build_machine.tokamak(
        active_coils_path=str(paths["active_coils"]),
        passive_coils_path=str(paths["passive_coils"]),
        limiter_path=str(paths["limiter"]),
        wall_path=str(paths["wall"]),
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-2.2,
        Zmax=2.2,
        nx=65,
        ny=129,
    )
    profiles = ConstrainPaxisIp(
        eq=eq,
        paxis=8.0e3,
        Ip=6.0e5,
        fvac=0.5,
        alpha_m=1.8,
        alpha_n=1.2,
    )
    null_points = [[0.6, 0.6], [1.1, -1.1]]
    isoflux_set = np.array(
        [
            [
                [0.6, 0.6, 0.34, 1.4, 1.0, 1.0, 0.8, 0.8],
                [1.1, -1.1, 0.0, 0.0, 2.0, -2.0, 1.62, -1.62],
            ]
        ],
        dtype=np.float64,
    )
    coil_limits: list[list[float | None]] = [
        [5e3, 9e3, 9e3, 7e3, 7e3, 5e3, 4e3, 5e3, 0.0, 0.0, None],
        [-5e3, -9e3, -9e3, -7e3, -7e3, -5e3, -4e3, -5e3, -10e3, -10e3, None],
    ]
    constrain = Inverse_optimizer(
        null_points=null_points,
        isoflux_set=isoflux_set,
        coil_current_limits=coil_limits,
    )
    constrain.mu_coils = 1.0e5
    eq.tokamak.set_coil_current("Solenoid", 5000.0)
    eq.tokamak["Solenoid"].control = False
    return eq, profiles, constrain, coil_limits


def solve_inverse_case_with_profiles(
    source: Path,
) -> tuple[Any, Any, Any, list[list[float | None]]]:
    """Solve the frozen case and retain the converged FreeGSNKE profile object."""
    eq, profiles, constrain, coil_limits = build_inverse_case(source)
    solver = GSstaticsolver.NKGSsolver(eq=eq)
    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=constrain,
        target_relative_tolerance=1.0e-6,
        target_relative_psit_update=1.0e-3,
        verbose=False,
        l2_reg=np.array([1.0e-12] * 10 + [1.0e-6], dtype=np.float64),
    )
    return eq, profiles, solver, coil_limits


def solve_inverse_case(source: Path) -> tuple[Any, Any, list[list[float | None]]]:
    """Solve the frozen case while preserving the original F-1c return contract."""
    eq, _, solver, coil_limits = solve_inverse_case_with_profiles(source)
    return eq, solver, coil_limits


def _active_filaments(
    source: Path,
    circuit_currents: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, list[dict[str, Any]]]:
    """Expand public circuit currents into signed active-coil filament currents."""
    active_path = _case_paths(source)["active_coils"]
    with active_path.open("rb") as handle:
        # Exact pinned, digest-bound public upstream input; never accepts caller-supplied pickle.
        config = pickle.load(handle)  # noqa: S301  # nosec B301
    if not isinstance(config, dict) or len(config) != circuit_currents.size:
        raise ValueError("active-coil configuration does not match the 12 circuit currents")

    radii: list[float] = []
    heights: list[float] = []
    incidence_rows: list[list[float]] = []
    circuit_contract: list[dict[str, Any]] = []
    for circuit_index, (name, entry) in enumerate(config.items()):
        units = [entry] if "R" in entry else list(entry.values())
        start = len(radii)
        signed_turns = 0.0
        for unit in units:
            r_values = np.asarray(unit["R"], dtype=np.float64).reshape(-1)
            z_values = np.asarray(unit["Z"], dtype=np.float64).reshape(-1)
            if r_values.shape != z_values.shape or not np.all(r_values > 0.0):
                raise ValueError(f"invalid filament coordinates for circuit {name}")
            polarity = float(unit["polarity"]) * float(unit["multiplier"])
            for radius, height in zip(r_values, z_values):
                row = [0.0] * circuit_currents.size
                row[circuit_index] = polarity
                radii.append(float(radius))
                heights.append(float(height))
                incidence_rows.append(row)
            signed_turns += polarity * float(r_values.size)
        circuit_contract.append(
            {
                "circuit": str(name),
                "current_a": float(circuit_currents[circuit_index]),
                "filament_count": len(radii) - start,
                "signed_turns": signed_turns,
            }
        )

    incidence = np.asarray(incidence_rows, dtype=np.float64)
    filament_currents = incidence @ circuit_currents
    return (
        np.asarray(radii, dtype=np.float64),
        np.asarray(heights, dtype=np.float64),
        filament_currents,
        incidence,
        circuit_contract,
    )


def _gradient_audit(
    coil_r: FloatArray,
    coil_z: FloatArray,
    incidence: FloatArray,
    circuit_currents: FloatArray,
) -> dict[str, Any]:
    """Compare JAX circuit-current gradients with central finite differences."""
    r_grid = jnp.linspace(0.35, 1.55, 9, dtype=jnp.float64)
    z_grid = jnp.linspace(-1.6, 1.6, 9, dtype=jnp.float64)
    r_fil = jnp.asarray(coil_r)
    z_fil = jnp.asarray(coil_z)
    mapping = jnp.asarray(incidence)

    @jax.jit
    def objective(currents: jax.Array) -> jax.Array:
        psi = vacuum_field_si(r_grid, z_grid, r_fil, z_fil, mapping @ currents)
        return jnp.mean(jnp.square(psi))

    current_jax = jnp.asarray(circuit_currents)
    autodiff = np.asarray(jax.grad(objective)(current_jax), dtype=np.float64)
    finite_difference = np.empty_like(autodiff)
    for index in range(circuit_currents.size):
        delta = np.zeros_like(circuit_currents)
        delta[index] = FINITE_DIFFERENCE_STEP_A
        plus = float(objective(jnp.asarray(circuit_currents + delta)))
        minus = float(objective(jnp.asarray(circuit_currents - delta)))
        finite_difference[index] = (plus - minus) / (2.0 * FINITE_DIFFERENCE_STEP_A)

    difference = autodiff - finite_difference
    denominator = max(float(np.linalg.norm(finite_difference)), np.finfo(float).tiny)
    relative_error = float(np.linalg.norm(difference) / denominator)
    all_finite = bool(np.all(np.isfinite(autodiff)) and np.all(np.isfinite(finite_difference)))
    return {
        "all_finite": all_finite,
        "autodiff_sha256": _array_sha256(autodiff),
        "finite_difference_sha256": _array_sha256(finite_difference),
        "finite_difference_step_a": FINITE_DIFFERENCE_STEP_A,
        "max_abs_error": float(np.max(np.abs(difference))),
        "objective": "mean(square(vacuum_psi_wb)) on fixed 9x9 interior diagnostic grid",
        "relative_l2_error": relative_error,
        "threshold_relative_l2_error_max": GRADIENT_RELATIVE_ERROR_MAX,
        "passed": all_finite and relative_error <= GRADIENT_RELATIVE_ERROR_MAX,
    }


def _topology_metrics(eq: Any) -> dict[str, Any]:
    opt, xpt = find_critical(
        eq.R,
        eq.Z,
        eq.psi(),
        eq.mask_inside_limiter.astype(bool),
        None,
    )
    axis_count = len(opt)
    xpoint_count = len(xpt)
    axis = np.asarray(opt[0][:2], dtype=np.float64) if axis_count == 1 else None
    axis_error = (
        float(np.linalg.norm(axis - EXPECTED_AXIS_M)) if axis is not None else float("inf")
    )
    xpoint_errors = []
    if xpoint_count:
        candidates = np.asarray(xpt[:, :2], dtype=np.float64)
        for expected in EXPECTED_XPOINTS_M:
            xpoint_errors.append(float(np.min(np.linalg.norm(candidates - expected, axis=1))))
    else:
        xpoint_errors = [float("inf")] * len(EXPECTED_XPOINTS_M)
    passed = (
        axis_count == 1
        and xpoint_count >= 2
        and axis_error <= TOPOLOGY_ATOL_M
        and max(xpoint_errors) <= TOPOLOGY_ATOL_M
    )
    return {
        "axis_count": axis_count,
        "axis_error_m": axis_error,
        "axis_m": axis.tolist() if axis is not None else None,
        "expected_axis_m": EXPECTED_AXIS_M.tolist(),
        "expected_xpoints_m": EXPECTED_XPOINTS_M.tolist(),
        "primary_xpoint_errors_m": xpoint_errors,
        "threshold_position_error_m_max": TOPOLOGY_ATOL_M,
        "xpoint_candidate_count": xpoint_count,
        "passed": bool(passed),
    }


def _current_limits(
    circuit_names: list[str],
    currents: FloatArray,
    coil_limits: list[list[float | None]],
) -> dict[str, Any]:
    upper = [None, *coil_limits[0]]
    lower = [None, *coil_limits[1]]
    rows: list[dict[str, Any]] = []
    passed = True
    for name, current, lower_limit, upper_limit in zip(circuit_names, currents, lower, upper):
        row_passed = bool(
            (lower_limit is None or current >= lower_limit)
            and (upper_limit is None or current <= upper_limit)
        )
        rows.append(
            {
                "circuit": name,
                "current_a": float(current),
                "lower_a": lower_limit,
                "upper_a": upper_limit,
                "passed": row_passed,
            }
        )
        passed = passed and row_passed
    return {"passed": passed, "rows": rows}


def _relative_l2(candidate: FloatArray, reference: FloatArray) -> float:
    """Return a finite relative L2 error against a non-zero reference."""
    denominator = float(np.linalg.norm(reference))
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("relative L2 reference must be non-zero")
    return float(np.linalg.norm(candidate - reference) / denominator)


def _profile_source_bridge(eq: Any, profiles: Any) -> dict[str, Any]:
    """Freeze the converged FreeGSNKE profile on SCPN's sampled source surface."""
    if not hasattr(profiles, "inputs") or len(profiles.inputs) != 3:
        raise ValueError("converged FreeGSNKE profile does not expose axis, boundary, and mask")

    psi_rz = np.asarray(eq.psi(), dtype=np.float64)
    r_grid = np.asarray(eq.R[:, 0], dtype=np.float64)
    axis = float(profiles.inputs[0])
    boundary = float(profiles.inputs[1])
    support_rz = np.asarray(profiles.inputs[2], dtype=bool)
    reference_jtor = np.asarray(profiles.jtor, dtype=np.float64)
    if psi_rz.shape != support_rz.shape or psi_rz.shape != reference_jtor.shape:
        raise ValueError("FreeGSNKE profile bridge arrays have inconsistent shapes")
    if not np.any(support_rz) or not np.all(r_grid > 0.0):
        raise ValueError("FreeGSNKE profile bridge requires non-empty support and positive R")

    knots = np.linspace(0.0, 1.0, PROFILE_SAMPLE_COUNT, dtype=np.float64)
    pprime = np.asarray(profiles.pprime(knots), dtype=np.float64)
    ffprime = np.asarray(profiles.ffprime(knots), dtype=np.float64)
    psi_zr = psi_rz.T

    def translated_jtor(
        psi: FloatArray,
        psi_axis: float,
        psi_boundary: float,
    ) -> FloatArray:
        rhs_zr = np.asarray(
            general_gs_source(
                jnp.asarray(psi.T),
                jnp.asarray(r_grid),
                jnp.asarray(psi_axis),
                jnp.asarray(psi_boundary),
                jnp.asarray(knots),
                jnp.asarray(pprime),
                jnp.asarray(ffprime),
                axis_connected=False,
            ),
            dtype=np.float64,
        )
        raw_jtor_rz = -rhs_zr.T / (float(MU0_SI) * np.asarray(eq.R, dtype=np.float64))
        return np.where(support_rz, raw_jtor_rz, 0.0)

    scpn_jtor = translated_jtor(psi_rz, axis, boundary)
    gauge_jtor = translated_jtor(
        psi_rz + GAUGE_OFFSET_WB_PER_RADIAN,
        axis + GAUGE_OFFSET_WB_PER_RADIAN,
        boundary + GAUGE_OFFSET_WB_PER_RADIAN,
    )
    direct_psin = (psi_rz - axis) / (boundary - axis)
    scpn_psin = np.asarray(
        normalised_flux_unclipped(
            jnp.asarray(psi_zr),
            jnp.asarray(axis),
            jnp.asarray(boundary),
        ),
        dtype=np.float64,
    ).T

    source_relative_l2 = _relative_l2(scpn_jtor, reference_jtor)
    gauge_relative_l2 = _relative_l2(gauge_jtor, scpn_jtor)
    psin_max_abs_error = float(np.max(np.abs(scpn_psin - direct_psin)))
    d_area = float((eq.R_1D[1] - eq.R_1D[0]) * (eq.Z_1D[1] - eq.Z_1D[0]))
    reference_current = float(np.sum(reference_jtor) * d_area)
    scpn_current = float(np.sum(scpn_jtor) * d_area)
    current_relative_error = abs(scpn_current - reference_current) / abs(reference_current)

    scale_candidates = {
        "identity": 1.0,
        "scaled_by_2pi": 2.0 * np.pi,
        "scaled_by_inv_2pi": 1.0 / (2.0 * np.pi),
        "scaled_by_minus_1": -1.0,
        "scaled_by_minus_2pi": -2.0 * np.pi,
    }
    scale_errors = {
        name: _relative_l2(scale * scpn_jtor, reference_jtor)
        for name, scale in scale_candidates.items()
    }
    best_scale = min(scale_errors, key=scale_errors.__getitem__)

    checks = {
        "anchor_values_match_equilibrium": (
            axis == float(eq.psi_axis) and boundary == float(eq.psi_bndry)
        ),
        "gauge_invariant_source": gauge_relative_l2 <= GAUGE_SOURCE_RELATIVE_L2_MAX,
        "identity_cocos_scale_selected": (
            DEFAULT_SOLVER_COCOS == 3 and best_scale == "identity"
        ),
        "normalised_flux_orientation": psin_max_abs_error <= PSIN_MAX_ABS_ERROR,
        "sampled_profile_source_parity": (
            source_relative_l2 <= PROFILE_SOURCE_RELATIVE_L2_MAX
        ),
        "total_current_preserved": (
            current_relative_error <= PROFILE_TOTAL_CURRENT_RELATIVE_ERROR_MAX
        ),
    }
    return {
        "checks": checks,
        "convention": {
            "freegsnke_grid_orientation": "R,Z",
            "profile_derivative_variable": "psi = poloidal flux per radian, Phi_p/(2*pi)",
            "psi_axis_wb_per_radian": axis,
            "psi_boundary_wb_per_radian": boundary,
            "psi_span_wb_per_radian": boundary - axis,
            "scpn_grid_orientation": "Z,R",
            "solver_cocos": DEFAULT_SOLVER_COCOS,
            "source_equation": "J_phi = R*pprime + FFprime/(mu0*R)",
            "support": "exact converged FreeGSNKE limiter_core_mask",
        },
        "digests": {
            "ffprime_samples_sha256": _array_sha256(ffprime),
            "freegsnke_jtor_sha256": _array_sha256(reference_jtor),
            "pprime_samples_sha256": _array_sha256(pprime),
            "psin_knots_sha256": _array_sha256(knots),
            "scpn_translated_jtor_sha256": _array_sha256(scpn_jtor),
            "support_mask_sha256": hashlib.sha256(
                np.asarray(support_rz, dtype=np.uint8, order="C").tobytes(order="C")
            ).hexdigest(),
        },
        "gauge_audit": {
            "offset_wb_per_radian": GAUGE_OFFSET_WB_PER_RADIAN,
            "relative_l2_error": gauge_relative_l2,
            "threshold_relative_l2_error_max": GAUGE_SOURCE_RELATIVE_L2_MAX,
        },
        "normalised_flux_audit": {
            "definition": "(psi - psi_axis)/(psi_boundary - psi_axis)",
            "max_abs_error": psin_max_abs_error,
            "threshold_max_abs_error": PSIN_MAX_ABS_ERROR,
        },
        "profile_sampling": {
            "ffprime_max": float(np.max(ffprime)),
            "ffprime_min": float(np.min(ffprime)),
            "interpolation": "linear on monotonic psi_N knots",
            "knot_count": PROFILE_SAMPLE_COUNT,
            "pprime_max": float(np.max(pprime)),
            "pprime_min": float(np.min(pprime)),
        },
        "scale_audit": {
            "adapter": best_scale,
            "candidate_relative_l2_errors": scale_errors,
        },
        "source_parity": {
            "freegsnke_total_current_a": reference_current,
            "max_abs_current_density_error_a_per_m2": float(
                np.max(np.abs(scpn_jtor - reference_jtor))
            ),
            "relative_l2_error": source_relative_l2,
            "scpn_total_current_a": scpn_current,
            "support_point_count": int(np.count_nonzero(support_rz)),
            "threshold_relative_l2_error_max": PROFILE_SOURCE_RELATIVE_L2_MAX,
            "total_current_relative_error": current_relative_error,
            "total_current_threshold_relative_error_max": (
                PROFILE_TOTAL_CURRENT_RELATIVE_ERROR_MAX
            ),
        },
        "passed": all(checks.values()),
    }


def build_report(source: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    """Execute the frozen inverse case and return its fail-closed comparison report."""
    source = source.resolve()
    versions = validate_source(source)
    paths = _case_paths(source)
    eq, profiles, solver, coil_limits = solve_inverse_case_with_profiles(source)

    currents = np.asarray(eq.tokamak.getCurrentsVec(), dtype=np.float64)
    active_currents = currents[:12]
    passive_currents = currents[12:]
    reference_currents = np.load(paths["current_baseline"]).astype(np.float64)
    reference_psi = np.load(paths["psi_baseline"]).astype(np.float64)
    solved_psi = np.asarray(eq.psi(), dtype=np.float64)

    current_max_abs_error = float(np.max(np.abs(active_currents - reference_currents)))
    psi_difference = solved_psi - reference_psi
    psi_span = float(np.ptp(reference_psi))
    psi_atol = PSI_SPAN_ATOL_FRACTION * psi_span
    psi_max_abs_error = float(np.max(np.abs(psi_difference)))
    psi_rmse = float(np.sqrt(np.mean(np.square(psi_difference))))
    passive_max_abs = float(np.max(np.abs(passive_currents))) if passive_currents.size else 0.0

    coil_r, coil_z, filament_i, incidence, circuits = _active_filaments(
        source, active_currents
    )
    scpn_vacuum = np.asarray(
        vacuum_field_si(
            jnp.asarray(eq.R[:, 0]),
            jnp.asarray(eq.Z[0, :]),
            jnp.asarray(coil_r),
            jnp.asarray(coil_z),
            jnp.asarray(filament_i),
        ),
        dtype=np.float64,
    ).T
    freegsnke_vacuum = np.asarray(eq.tokamak_psi, dtype=np.float64)
    limiter_mask = np.asarray(eq.mask_inside_limiter, dtype=bool)
    if limiter_mask.shape != scpn_vacuum.shape or not np.any(limiter_mask):
        raise ValueError("invalid FreeGSNKE limiter mask")
    vacuum_difference = scpn_vacuum[limiter_mask] - freegsnke_vacuum[limiter_mask]
    vacuum_max_abs = float(np.max(np.abs(vacuum_difference)))
    vacuum_rmse = float(np.sqrt(np.mean(np.square(vacuum_difference))))

    topology = _topology_metrics(eq)
    circuit_names = [str(name) for name in list(eq.tokamak.coils_dict)[:12]]
    limits = _current_limits(circuit_names, active_currents, coil_limits)
    gradient = _gradient_audit(coil_r, coil_z, incidence, active_currents)
    profile_bridge = _profile_source_bridge(eq, profiles)

    checks = {
        "active_current_regression": current_max_abs_error <= CURRENT_ATOL_A,
        "current_limits": bool(limits["passed"]),
        "passive_currents_zero": passive_max_abs <= PASSIVE_CURRENT_ATOL_A,
        "scpn_coil_gradient": bool(gradient["passed"]),
        "scpn_freegsnke_vacuum_parity_inside_limiter": (
            vacuum_max_abs <= VACUUM_LIMITER_MAX_ABS_WB
        ),
        "scpn_freegsnke_profile_source_bridge": bool(profile_bridge["passed"]),
        "topology_regression": bool(topology["passed"]),
        "total_psi_regression": psi_max_abs_error <= psi_atol,
    }
    passed = all(checks.values())

    artifacts = {
        name: {
            "path": str(path.relative_to(source)),
            "sha256": _file_sha256(path),
        }
        for name, path in sorted(paths.items())
    }
    report: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "blockers": [
            (
                "The sampled profile-source convention is frozen, but a self-consistent "
                "SCPN solve from the translated pprime/FFprime and exact topology support "
                "has not yet demonstrated full total-psi cross-solver parity"
            )
        ],
        "case_contract": {
            "grid_shape_r_z": [65, 129],
            "ip_target_a": 600000.0,
            "profile": {
                "alpha_m": 1.8,
                "alpha_n": 1.2,
                "fvac": 0.5,
                "paxis_pa": 8000.0,
            },
            "solver": {
                "l2_reg": [1.0e-12] * 10 + [1.0e-6],
                "target_relative_psit_update": 1.0e-3,
                "target_relative_tolerance": 1.0e-6,
            },
        },
        "checks": checks,
        "claim_boundary": CLAIM_BOUNDARY,
        "comparison_scope": {
            "admitted": [
                "pinned FreeGSNKE inverse solve against upstream current/psi baselines",
                "pinned diverted topology and active-current limit regression",
                "same-current same-filament SCPN/FreeGSNKE vacuum psi parity inside limiter",
                "SCPN circuit-current autodiff gradient against central finite differences",
                (
                    "FreeGSNKE ConstrainPaxisIp to SCPN sampled pprime/FFprime source parity "
                    "with exact LCFS support, gauge invariance, and identity COCOS-3 scale"
                ),
            ],
            "not_admitted": [
                "self-consistent total-psi SCPN/FreeGSNKE parity",
                "facility, PCS, safety, or control validation",
                "latency or real-time readiness",
            ],
        },
        "digests": {
            "active_circuit_currents_sha256": _array_sha256(active_currents),
            "active_filament_currents_sha256": _array_sha256(filament_i),
            "active_filament_incidence_sha256": _array_sha256(incidence),
            "active_filament_r_sha256": _array_sha256(coil_r),
            "active_filament_z_sha256": _array_sha256(coil_z),
            "freegsnke_total_psi_sha256": _array_sha256(solved_psi),
            "freegsnke_vacuum_psi_sha256": _array_sha256(freegsnke_vacuum),
            "scpn_vacuum_psi_sha256": _array_sha256(scpn_vacuum),
        },
        "gradient_audit": gradient,
        "inverse_regression": {
            "active_current_max_abs_error_a": current_max_abs_error,
            "active_current_threshold_a": CURRENT_ATOL_A,
            "passive_current_max_abs_a": passive_max_abs,
            "passive_current_threshold_a": PASSIVE_CURRENT_ATOL_A,
            "solver_relative_change": float(solver.relative_change),
            "total_psi_atol_wb": psi_atol,
            "total_psi_max_abs_error_wb": psi_max_abs_error,
            "total_psi_rmse_wb": psi_rmse,
            "total_psi_span_wb": psi_span,
        },
        "machine_contract": {
            "active_circuit_count": int(active_currents.size),
            "active_filament_count": int(coil_r.size),
            "circuits": circuits,
            "current_limits": limits,
            "limiter_comparison_point_count": int(np.count_nonzero(limiter_mask)),
            "passive_circuit_count": int(passive_currents.size),
        },
        "milestones": ["F-1c", "F-1d"],
        "payload_sha256": "",
        "profile_source_bridge": profile_bridge,
        "provenance": {
            "artifacts": artifacts,
            "dependency_note": (
                "FreeGSNKE 3.0.1 declares freegs4e~=0.13, which admits incompatible "
                "0.14.0; this benchmark pins the upstream-compatible 0.13.1 runtime"
            ),
            "local_artifacts": {
                "benchmark_generator": {
                    "path": "validation/benchmark_freegsnke_inverse.py",
                    "sha256": _file_sha256(Path(__file__)),
                },
                "dependency_input": {
                    "path": "requirements/ci-freegsnke.in",
                    "sha256": _file_sha256(ROOT / "requirements/ci-freegsnke.in"),
                },
                "dependency_lock": {
                    "path": "requirements/ci-freegsnke.txt",
                    "sha256": _file_sha256(ROOT / "requirements/ci-freegsnke.txt"),
                },
                "remote_workflow": {
                    "path": ".github/workflows/freegsnke-inverse.yml",
                    "sha256": _file_sha256(ROOT / ".github/workflows/freegsnke-inverse.yml"),
                },
            },
            "source_commit": UPSTREAM_COMMIT,
            "source_url": "https://github.com/FusionComputingLab/freegsnke",
            "versions": versions,
        },
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "thresholds_predeclared": {
            "active_current_max_abs_error_a": CURRENT_ATOL_A,
            "gradient_relative_l2_error_max": GRADIENT_RELATIVE_ERROR_MAX,
            "profile_gauge_relative_l2_error_max": GAUGE_SOURCE_RELATIVE_L2_MAX,
            "profile_psin_max_abs_error": PSIN_MAX_ABS_ERROR,
            "profile_source_relative_l2_error_max": PROFILE_SOURCE_RELATIVE_L2_MAX,
            "profile_total_current_relative_error_max": (
                PROFILE_TOTAL_CURRENT_RELATIVE_ERROR_MAX
            ),
            "passive_current_max_abs_a": PASSIVE_CURRENT_ATOL_A,
            "topology_position_error_m_max": TOPOLOGY_ATOL_M,
            "total_psi_span_atol_fraction": PSI_SPAN_ATOL_FRACTION,
            "vacuum_limiter_max_abs_error_wb": VACUUM_LIMITER_MAX_ABS_WB,
        },
        "topology": topology,
        "vacuum_bridge": {
            "domain": "FreeGSNKE mask_inside_limiter",
            "freegsnke_orientation": "R,Z",
            "max_abs_error_wb": vacuum_max_abs,
            "rmse_wb": vacuum_rmse,
            "scpn_orientation_before_transpose": "Z,R",
            "threshold_max_abs_error_wb": VACUUM_LIMITER_MAX_ABS_WB,
        },
    }
    report["payload_sha256"] = _payload_sha256(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    inverse = cast(dict[str, Any], report["inverse_regression"])
    vacuum = cast(dict[str, Any], report["vacuum_bridge"])
    gradient = cast(dict[str, Any], report["gradient_audit"])
    profile = cast(dict[str, Any], report["profile_source_bridge"])
    profile_source = cast(dict[str, Any], profile["source_parity"])
    profile_gauge = cast(dict[str, Any], profile["gauge_audit"])
    profile_scale = cast(dict[str, Any], profile["scale_audit"])
    topology = cast(dict[str, Any], report["topology"])
    checks = cast(dict[str, bool], report["checks"])
    lines = [
        "# FreeGSNKE inverse-equilibrium comparison",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        "This is bounded research evidence. It is not facility, PCS, safety, or control admission.",
        "",
        "## Results",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| FreeGSNKE active-current max error | {inverse['active_current_max_abs_error_a']:.6g} A |",
        f"| FreeGSNKE total-psi max error | {inverse['total_psi_max_abs_error_wb']:.6g} Wb |",
        f"| Magnetic-axis error | {topology['axis_error_m']:.6g} m |",
        f"| Worst primary X-point error | {max(topology['primary_xpoint_errors_m']):.6g} m |",
        f"| SCPN vs FreeGSNKE vacuum-psi max error inside limiter | {vacuum['max_abs_error_wb']:.6g} Wb |",
        f"| SCPN current-gradient relative L2 error | {gradient['relative_l2_error']:.6g} |",
        f"| Sampled pprime/FFprime source relative L2 error | {profile_source['relative_l2_error']:.6g} |",
        f"| Gauge-shifted source relative L2 error | {profile_gauge['relative_l2_error']:.6g} |",
        f"| Selected COCOS-3 source adapter | {profile_scale['adapter']} |",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in checks.items())
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "The FreeGSNKE profile-source translation into SCPN sampled `pprime`/`FFprime` "
                "is frozen for this case, including exact LCFS support, gauge invariance, and "
                "identity COCOS-3 flux-per-radian scaling. A self-consistent SCPN solve has not "
                "yet demonstrated full total-psi cross-solver parity, so that broader claim "
                "remains explicitly unadmitted."
            ),
            "",
            f"Payload SHA-256: `{report['payload_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"{BENCHMARK_ID}: {report['status']}")
    print(f"JSON: {args.output}")
    print(f"Markdown: {args.markdown}")
    if args.strict and report["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
