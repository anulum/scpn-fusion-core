# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Control Exception Specificity Tests
"""Regression tests for narrowed control exception handlers."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.control.neuro_cybernetic_controller import run_neuro_cybernetic_control

FloatArray = NDArray[np.float64]


class _CoilConfig(TypedDict):
    current: float


class _PhysicsConfig(TypedDict, total=False):
    plasma_current_target: float
    beta_N: float


class _KernelConfig(TypedDict):
    physics: _PhysicsConfig
    coils: list[_CoilConfig]


class _FaultingProfile:
    """Profile object whose array conversion fails outside NumPy coercion errors."""

    def __array__(self, dtype: object | None = None) -> FloatArray:
        """Raise a runtime failure that must not be hidden by coercion fallback."""
        raise RuntimeError("profile source failed")


class _ProfileKernel:
    """Minimal FusionKernel stand-in for neuro-controller profile coercion tests."""

    def __init__(self, _config_file: str) -> None:
        self.cfg: _KernelConfig = {
            "physics": {"plasma_current_target": 5.0, "beta_N": 0.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R: FloatArray = np.linspace(6.0, 6.4, 9, dtype=np.float64)
        self.Z: FloatArray = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
        rr, zz = np.meshgrid(self.R, self.Z)
        self.Psi: FloatArray = 1.0 - ((rr - 6.2) ** 2 + zz**2)
        self.Te: object = np.asarray([1.0, 1.1], dtype=np.float64)
        self.ne: object = np.asarray([1.0, 1.2], dtype=np.float64)

    def solve_equilibrium(self) -> None:
        """Keep the analytic flux map fixed for deterministic controller tests."""
        return None


class _BadNumericProfileKernel(_ProfileKernel):
    """Kernel whose profile values raise NumPy coercion errors."""

    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.Te = ["not-a-number"]
        self.ne = ["also-not-a-number"]


class _RuntimeFaultProfileKernel(_ProfileKernel):
    """Kernel whose profile object raises a non-coercion runtime error."""

    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.Te = _FaultingProfile()


def test_neuro_profile_coercion_errors_degrade_to_zero() -> None:
    """Expected NumPy profile conversion errors keep the run fail-closed."""
    summary = run_neuro_cybernetic_control(
        config_file="dummy.json",
        shot_duration=3,
        save_plot=False,
        verbose=False,
        kernel_factory=_BadNumericProfileKernel,
    )

    assert summary["steps"] == 3
    assert summary["safety_contract_violations"] >= 0


def test_neuro_profile_runtime_errors_are_not_swallowed() -> None:
    """Unexpected profile runtime failures propagate instead of becoming zero."""
    with pytest.raises(RuntimeError, match="profile source failed"):
        run_neuro_cybernetic_control(
            config_file="dummy.json",
            shot_duration=3,
            save_plot=False,
            verbose=False,
            kernel_factory=_RuntimeFaultProfileKernel,
        )


def _load_traceable_runtime_with_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    failure: BaseException,
) -> ModuleType:
    """Load the traceable runtime while forcing one optional backend import to fail."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scpn_fusion"
        / "control"
        / "jax_traceable_runtime.py"
    )
    real_import = builtins.__import__

    def _raising_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == module_name or name.startswith(f"{module_name}."):
            raise failure
        return real_import(name, globals, locals, fromlist, level)

    unique_name = f"_traceable_runtime_import_failure_{module_name}"
    monkeypatch.setattr(builtins, "__import__", _raising_import)
    spec = importlib.util.spec_from_file_location(unique_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to build traceable runtime import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(unique_name, None)
    return module


def test_traceable_runtime_jax_attribute_error_import_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional JAX initialization AttributeError keeps the NumPy backend usable."""
    module = _load_traceable_runtime_with_import_failure(
        monkeypatch,
        module_name="jax",
        failure=AttributeError("jax version unavailable"),
    )

    assert module._HAS_JAX is False
    assert "numpy" in module.available_traceable_backends()


def test_traceable_runtime_torch_runtime_error_import_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional Torch initialization RuntimeError keeps the NumPy backend usable."""
    module = _load_traceable_runtime_with_import_failure(
        monkeypatch,
        module_name="torch",
        failure=RuntimeError("torch extension load failed"),
    )

    assert module._HAS_TORCH is False
    assert "numpy" in module.available_traceable_backends()
