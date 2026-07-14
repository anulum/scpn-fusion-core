# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
"""Pytest tests for the equilibrium-coupled burn physics (FusionBurnPhysics)."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_fusion.core.fusion_burn_physics as fusion_burn_physics
from scpn_fusion.core.fusion_burn_physics import FusionBurnPhysics, run_ignition_experiment


def _bare_burn_lab() -> FusionBurnPhysics:
    """Build a FusionBurnPhysics with grid geometry but no full kernel init."""
    lab = FusionBurnPhysics.__new__(FusionBurnPhysics)
    lab.R = np.linspace(3.0, 9.0, 33)
    lab.Z = np.linspace(-4.0, 4.0, 33)
    lab.dR = lab.R[1] - lab.R[0]
    lab.dZ = lab.Z[1] - lab.Z[0]
    lab.RR, lab.ZZ = np.meshgrid(lab.R, lab.Z)
    lab.Psi = np.exp(-((lab.RR - 6.2) ** 2 + lab.ZZ**2) / 4.0)
    lab.cfg = {
        "physics": {"plasma_current_target": 15.0e6},
        "dimensions": {
            "R_min": 4.0,
            "R_max": 8.4,
            "Z_min": -4.0,
            "Z_max": 4.0,
            "B0": 5.3,
            "R0": 6.2,
            "kappa": 1.7,
        },
    }
    return lab


def test_calculate_thermodynamics_reports_power_balance() -> None:
    """The equilibrium thermodynamics map returns a finite fusion power balance."""
    out = _bare_burn_lab().calculate_thermodynamics(P_aux_MW=50.0)
    for key in ("P_fusion_MW", "P_alpha_MW", "P_loss_MW", "Net_MW", "Q", "W_MJ"):
        assert key in out
        assert np.isfinite(out[key])
    assert out["P_fusion_MW"] > 0.0
    assert out["Q"] > 0.0


def test_calculate_thermodynamics_zero_aux_gives_zero_q() -> None:
    """With no auxiliary heating the gain is defined as zero."""
    out = _bare_burn_lab().calculate_thermodynamics(P_aux_MW=0.0)
    assert out["Q"] == 0.0


def test_calculate_thermodynamics_rejects_negative_aux() -> None:
    """A negative auxiliary-power input is rejected."""
    with pytest.raises(ValueError, match="P_aux_MW"):
        _bare_burn_lab().calculate_thermodynamics(P_aux_MW=-10.0)


def test_calculate_thermodynamics_limiter_boundary_fallback() -> None:
    """A near-flat flux map falls back to the minimum as the plasma boundary."""
    lab = _bare_burn_lab()
    lab.Psi = np.full((33, 33), 0.5)
    out = lab.calculate_thermodynamics(P_aux_MW=50.0)
    assert np.isfinite(out["P_fusion_MW"])


def test_run_ignition_experiment_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone ignition power-ramp demo runs and renders its plot."""
    import matplotlib.pyplot as plt

    lab = _bare_burn_lab()
    lab.solve_equilibrium = lambda: None  # type: ignore[assignment, misc]
    # run_ignition_experiment resolves ``FusionBurnPhysics`` in its own module
    # namespace, so the patch target is fusion_burn_physics, not the facade.
    monkeypatch.setattr(fusion_burn_physics, "FusionBurnPhysics", lambda _path: lab)

    saved: list[str] = []
    monkeypatch.setattr(plt, "savefig", lambda path, *a, **k: saved.append(str(path)))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    run_ignition_experiment()

    assert saved


def test_fusion_burn_physics_constructs_from_config() -> None:
    from scpn_fusion._data_paths import data_root

    cfg = data_root() / "validation" / "iter_validated_config.json"
    lab = FusionBurnPhysics(str(cfg))
    assert isinstance(lab, FusionBurnPhysics)
