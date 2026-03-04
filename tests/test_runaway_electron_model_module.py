"""Direct module-link tests for extracted runaway electron model."""

from __future__ import annotations

from scpn_fusion.control.runaway_electron_model import (
    RunawayElectronModel,
    RunawayElectronResult,
)


def test_runaway_model_simulate_smoke() -> None:
    model = RunawayElectronModel(n_e=1e20, T_e_keV=20.0, z_eff=2.0)
    out = model.simulate(duration_s=0.005, dt_s=5e-5)
    assert isinstance(out, RunawayElectronResult)
    assert out.peak_re_current_ma >= 0.0
    assert out.final_re_current_ma >= 0.0
