# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Configuration Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""Tests for TokamakConfig dataclass and named presets."""

from __future__ import annotations

import pytest

from scpn_fusion.core.tokamak_config import TokamakConfig


_PRESET_R0 = {
    "iter": 6.2,
    "sparc": 1.85,
    "diiid": 1.67,
    "jet": 2.96,
}


@pytest.mark.parametrize("preset,expected_R0", list(_PRESET_R0.items()))
def test_preset_R0(preset, expected_R0):
    cfg = getattr(TokamakConfig, preset)()
    assert pytest.approx(expected_R0) == cfg.R0


def test_custom_construction():
    cfg = TokamakConfig(
        name="TEST",
        R0=3.0,
        a=1.0,
        B0=5.0,
        Ip=10.0,
        kappa=1.5,
        delta=0.25,
        n_e=8.0,
        T_e=12.0,
        P_aux=20.0,
    )
    assert cfg.name == "TEST"
    assert cfg.R0 == 3.0


def test_aspect_ratio():
    cfg = TokamakConfig(
        name="AR",
        R0=6.0,
        a=2.0,
        B0=5.0,
        Ip=10.0,
        kappa=1.5,
        delta=0.3,
        n_e=5.0,
        T_e=10.0,
        P_aux=30.0,
    )
    assert cfg.aspect_ratio == pytest.approx(3.0)
    assert cfg.epsilon == pytest.approx(1.0 / 3.0)


def test_aspect_ratio_presets():
    for preset in ("iter", "sparc", "diiid", "jet"):
        cfg = getattr(TokamakConfig, preset)()
        assert cfg.aspect_ratio == pytest.approx(cfg.R0 / cfg.a)


def test_frozen():
    cfg = TokamakConfig.iter()
    with pytest.raises(AttributeError):
        cfg.R0 = 99.0


def test_scaling_law_accepts_config_fields():
    """TokamakConfig fields feed directly into ipb98y2_tau_e."""
    from scpn_fusion.core.scaling_laws import ipb98y2_tau_e

    for preset in ("iter", "sparc", "diiid", "jet"):
        cfg = getattr(TokamakConfig, preset)()
        tau = ipb98y2_tau_e(
            Ip=cfg.Ip,
            BT=cfg.B0,
            ne19=cfg.n_e,
            Ploss=cfg.P_aux,
            R=cfg.R0,
            kappa=cfg.kappa,
            epsilon=cfg.epsilon,
        )
        assert tau > 0.0
