# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Shared physical case public regression tests
"""Exercise the shared corpus through the public loader and NumPy solver."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.jax_gs_solver import gs_solve_np, jax_gs_solve
from scpn_fusion.core.physical_case import GradShafranovCase, case_from_toml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_CORPUS = Path(__file__).resolve().parents[1] / "validation/polyglot/case_contract"
with (_CORPUS / "manifest.toml").open("rb") as _stream:
    _MANIFEST = tomllib.load(_stream)
_CASES: list[dict[str, Any]] = _MANIFEST["case"]


@pytest.mark.parametrize("fixture", _CASES, ids=[row["name"] for row in _CASES])
def test_shared_physical_case_corpus(fixture: dict[str, Any]) -> None:
    """Preserve exact declared physical mappings and reject every invalid deck."""
    path = _CORPUS / fixture["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"]
    if not fixture["valid"]:
        with pytest.raises((TypeError, ValueError)):
            case_from_toml(path)
        return
    case = case_from_toml(path)
    assert case.as_mapping() == fixture["expected"]
    assert GradShafranovCase(**case.as_mapping()) == case
    if fixture["solve"]:
        flux = gs_solve_np(**case.as_mapping())
        assert flux.shape == (case.NZ, case.NR)
        assert flux.dtype == np.float64
        assert np.all(np.isfinite(flux))
        assert np.all(flux[[0, -1], :] == 0.0)
        assert np.all(flux[:, [0, -1]] == 0.0)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("NR", True, TypeError),
        ("NZ", 3.0, TypeError),
        ("n_picard", 1 << 63, ValueError),
        ("n_jacobi", -1, ValueError),
        ("Ip_target", 9007199254740993, ValueError),
        ("Ip_target", (1 << 63) - 1, ValueError),
        ("mu0", "1e-6", TypeError),
        ("R_min", 0.0, ValueError),
        ("alpha", float("nan"), ValueError),
        ("NR", 1026, ValueError),
        ("n_jacobi", 10001, ValueError),
    ],
)
def test_direct_solver_admission(field: str, value: object, error: type[Exception]) -> None:
    """Validate delivered direct solver arguments before full mesh allocation."""
    arguments: dict[str, Any] = dict(case_from_toml(_CORPUS / "cases/reference.toml").as_mapping())
    arguments[field] = value
    with pytest.raises(error):
        gs_solve_np(**arguments)
    with pytest.raises(error):
        jax_gs_solve(**arguments, use_jax=False)


def test_admitted_geometry_arithmetic_failure() -> None:
    """Reject overflow in the actual public solve instead of returning zero/NaN."""
    case = case_from_toml(_CORPUS / "cases/reference.toml").as_mapping()
    case.update({"R_min": 1e200, "R_max": 2e200, "NR": 3, "NZ": 3, "n_picard": 1, "n_jacobi": 1})
    GradShafranovCase(**case)
    with pytest.raises(RuntimeError, match="arithmetic"):
        gs_solve_np(**case)


class _IntegerSubclass(int):
    """An integer with hostile conversion hooks that admission must not call."""

    def __int__(self) -> int:
        """Refuse a custom conversion path."""
        raise AssertionError("custom integer conversion executed")


class _FloatSubclass(float):
    """A float with hostile conversion hooks that admission must not call."""

    def __float__(self) -> float:
        """Refuse a custom conversion path."""
        raise AssertionError("custom float conversion executed")


def test_scalar_subclass_storage_semantics() -> None:
    """Read primitive stored values without executing custom scalar conversion."""
    case = case_from_toml(_CORPUS / "cases/reference.toml").as_mapping()
    case.update({"NR": _IntegerSubclass(17), "Ip_target": _FloatSubclass(1e6)})
    loaded = GradShafranovCase(**case)
    assert type(loaded.NR) is int
    assert type(loaded.Ip_target) is float
    assert loaded.as_mapping() == case_from_toml(_CORPUS / "cases/reference.toml").as_mapping()


@pytest.mark.parametrize("witness", ["seed_square", "seed_exponent", "scaled_current"])
def test_hidden_intermediate_overflow(witness: str) -> None:
    """Refuse admitted arithmetic before exponential or physical rescaling hides overflow."""
    case = case_from_toml(_CORPUS / "cases/reference.toml").as_mapping()
    case.update({"n_picard": 1, "n_jacobi": 1})
    if witness == "scaled_current":
        case.update(
            {"R_min": 1.0, "R_max": 1.01, "Z_min": -0.01, "Z_max": 0.01, "Ip_target": 1e308}
        )
    else:
        case.update({"R_min": 1e154, "R_max": 1e155 if witness == "seed_square" else 3e154})
    GradShafranovCase(**case)
    with pytest.raises(RuntimeError):
        gs_solve_np(**case)
    with pytest.raises(RuntimeError):
        jax_gs_solve(**case, use_jax=False)
