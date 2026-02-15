# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Rust Compat Solver Method Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import pytest


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "validation" / "iter_validated_config.json"


def _require_rust_extension():
    try:
        import scpn_fusion_rs as rs  # type: ignore
    except ImportError:
        pytest.skip("scpn_fusion_rs extension is not available in this environment")
    return rs


def test_solver_method_aliases_roundtrip() -> None:
    rs = _require_rust_extension()
    kernel = rs.PyFusionKernel(str(_config_path()))

    assert kernel.solver_method() == "sor"

    for alias in ("multigrid", "picard_multigrid", "mg"):
        kernel.set_solver_method(alias)
        assert kernel.solver_method() == "multigrid"

    for alias in ("sor", "picard_sor"):
        kernel.set_solver_method(alias)
        assert kernel.solver_method() == "sor"


def test_solver_method_rejects_unknown_value() -> None:
    rs = _require_rust_extension()
    kernel = rs.PyFusionKernel(str(_config_path()))

    with pytest.raises(ValueError, match="Unknown solver method"):
        kernel.set_solver_method("not-a-solver")
