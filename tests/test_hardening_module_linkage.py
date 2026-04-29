# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

from pathlib import Path

import numpy as np


def test_hardening_split_modules_importable() -> None:
    import scpn_fusion.control.disruption_risk_runtime as disruption_risk_runtime
    import scpn_fusion.core._gk_nonlinear_operators as gk_nonlinear_operators
    import scpn_fusion.core._gk_nonlinear_setup as gk_nonlinear_setup
    import scpn_fusion.core._gk_nonlinear_time as gk_nonlinear_time
    import scpn_fusion.core._gk_nonlinear_types as gk_nonlinear_types
    import scpn_fusion.core._neural_transport_analytic as neural_transport_analytic
    import scpn_fusion.core._neural_transport_runtime as neural_transport_runtime
    import scpn_fusion.core._neural_transport_types as neural_transport_types
    import scpn_fusion.core._integrated_transport_solver_model_backend as transport_backend
    import scpn_fusion.core._integrated_transport_solver_model_common as transport_common
    import scpn_fusion.core._integrated_transport_solver_model_pedestal as transport_pedestal
    import scpn_fusion.core.integrated_transport_solver_coupling as transport_coupling
    import scpn_fusion.core.integrated_transport_solver_model as transport_model
    import scpn_fusion.core.gk_nonlinear as gk_nonlinear
    import scpn_fusion.core.neural_transport as neural_transport
    import scpn_fusion.io.tokamak_disruption_archive as disruption_archive
    import scpn_fusion.io.tokamak_live_payload as live_payload
    import scpn_fusion.io.tokamak_synthetic_archive as synthetic_archive
    import scpn_fusion.scpn.controller_backend_mixin as controller_backend_mixin
    import scpn_fusion.scpn.controller_features_mixin as controller_features_mixin

    modules = [
        disruption_risk_runtime,
        gk_nonlinear_operators,
        gk_nonlinear_setup,
        gk_nonlinear_time,
        gk_nonlinear_types,
        neural_transport_analytic,
        neural_transport_runtime,
        neural_transport_types,
        transport_backend,
        transport_common,
        transport_pedestal,
        transport_coupling,
        transport_model,
        gk_nonlinear,
        neural_transport,
        disruption_archive,
        live_payload,
        synthetic_archive,
        controller_backend_mixin,
        controller_features_mixin,
    ]
    assert all(mod.__name__.startswith("scpn_fusion.") for mod in modules)
    assert issubclass(
        transport_model.TransportSolverModelMixin,
        transport_backend.TransportSolverBackendMixin,
    )
    assert issubclass(
        transport_model.TransportSolverModelMixin,
        transport_pedestal.TransportSolverPedestalMixin,
    )
    assert neural_transport.NeuralTransportModel is neural_transport_runtime.NeuralTransportModel
    assert neural_transport.TransportInputs is neural_transport_types.TransportInputs
    assert (
        neural_transport.critical_gradient_model
        is neural_transport_analytic.critical_gradient_model
    )
    assert gk_nonlinear.NonlinearGKConfig is gk_nonlinear_types.NonlinearGKConfig
    assert issubclass(
        gk_nonlinear.NonlinearGKSolver,
        gk_nonlinear_setup.NonlinearGKSetupMixin,
    )
    assert issubclass(
        gk_nonlinear.NonlinearGKSolver,
        gk_nonlinear_operators.NonlinearGKOperatorsMixin,
    )
    assert issubclass(
        gk_nonlinear.NonlinearGKSolver,
        gk_nonlinear_time.NonlinearGKTimeMixin,
    )


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
