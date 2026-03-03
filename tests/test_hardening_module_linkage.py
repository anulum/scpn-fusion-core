from __future__ import annotations

from pathlib import Path

import numpy as np


def test_hardening_split_modules_importable() -> None:
    import scpn_fusion.control.disruption_risk_runtime as disruption_risk_runtime
    import scpn_fusion.core.integrated_transport_solver_coupling as transport_coupling
    import scpn_fusion.core.integrated_transport_solver_model as transport_model
    import scpn_fusion.io.tokamak_disruption_archive as disruption_archive
    import scpn_fusion.io.tokamak_live_payload as live_payload
    import scpn_fusion.io.tokamak_synthetic_archive as synthetic_archive
    import scpn_fusion.scpn.controller_backend_mixin as controller_backend_mixin
    import scpn_fusion.scpn.controller_features_mixin as controller_features_mixin

    modules = [
        disruption_risk_runtime,
        transport_coupling,
        transport_model,
        disruption_archive,
        live_payload,
        synthetic_archive,
        controller_backend_mixin,
        controller_features_mixin,
    ]
    assert all(mod.__name__.startswith("scpn_fusion.") for mod in modules)


def test_hardening_split_modules_smoke(tmp_path: Path) -> None:
    import scpn_fusion.control.disruption_risk_runtime as disruption_risk_runtime
    import scpn_fusion.io.tokamak_disruption_archive as disruption_archive
    import scpn_fusion.io.tokamak_live_payload as live_payload
    import scpn_fusion.io.tokamak_synthetic_archive as synthetic_archive

    signal, label, ttd = disruption_risk_runtime.simulate_tearing_mode(
        steps=16, rng=np.random.default_rng(0)
    )
    assert signal.shape == (16,)
    assert int(label) in {0, 1}
    assert isinstance(ttd, (int, np.integer))

    listed = disruption_archive.list_disruption_shots(disruption_dir=tmp_path)
    assert listed == []

    shot_payload = synthetic_archive.generate_synthetic_shot_database(
        n_shots=1,
        output_dir=tmp_path,
        seed=7,
    )
    assert len(shot_payload) >= 1
    assert any(tmp_path.glob("*.npz"))

    scalar = live_payload.to_scalar([np.nan, 1.0, 2.0])
    assert scalar == 2.0
