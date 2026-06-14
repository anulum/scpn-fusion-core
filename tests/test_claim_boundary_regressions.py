# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Claim Boundary Regression Tests

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scpn_fusion.control.rmf_phase_lock import RMFPhaseLockController
from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.pulsed_compression import slough_fig5_acceptance_status
from tools.parallel_gen_iter import (
    DEFAULT_SHARED_WORKERS,
    default_worker_count,
    is_boundary_xpoint,
    validate_worker_policy,
)
from tools import train_frc_quantized_surrogate
from tools.train_frc_snn_surrogate import export_to_verilog


def test_rotating_frc_public_solver_fails_closed() -> None:
    inputs = RigidRotorFRCInputs(
        n0=1.0e20,
        T_i_eV=10_000.0,
        T_e_eV=5_000.0,
        theta_dot=1.0,
        R_s=0.2,
        B_ext=5.0,
        delta=0.02,
    )
    rho = np.linspace(0.0, 0.4, 32)

    with pytest.raises(NotImplementedError, match="rotating rigid-rotor BVP"):
        solve_frc_equilibrium(inputs, rho)


def test_slough_sidecar_presence_is_not_acceptance_pass() -> None:
    status = slough_fig5_acceptance_status()

    assert status["status"] != "passed"
    assert status["status"] == "blocked_reconstructed_reference_not_public_digitised"


def test_placeholder_hardware_exports_fail_closed(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="FPGA export is not implemented"):
        RMFPhaseLockController().export_to_fpga(str(tmp_path))

    with pytest.raises(NotImplementedError, match="Verilog export is not implemented"):
        export_to_verilog({"W": np.zeros((1, 1)), "b": np.zeros(1)}, tmp_path / "frc_snn_core.v")


def test_quantized_frc_surrogate_rejects_nonfinite_solver_output(monkeypatch) -> None:
    def fake_solver(*_args, **_kwargs):
        return SimpleNamespace(B_z=np.array([1.0, np.nan, 3.0], dtype=np.float64))

    monkeypatch.setattr(train_frc_quantized_surrogate, "solve_frc_equilibrium", fake_solver)

    with pytest.raises(ValueError, match="nominal B_z must be finite"):
        train_frc_quantized_surrogate.compute_quantized_jacobian(
            np.array([3.0, 10.0, 5.0, 0.0, 0.2, 5.0, 0.02], dtype=np.float64),
            grid_size=3,
        )


def test_iter_generator_rejects_boundary_xpoints_and_unjustified_full_host() -> None:
    assert default_worker_count(DEFAULT_SHARED_WORKERS + 10) == DEFAULT_SHARED_WORKERS
    assert is_boundary_xpoint(2.0, -4.0, 2.0, 8.0, -4.0, 4.0)
    assert not is_boundary_xpoint(5.0, 0.0, 2.0, 8.0, -4.0, 4.0)

    with pytest.raises(ValueError, match="workers above 12"):
        validate_worker_policy(
            DEFAULT_SHARED_WORKERS + 1,
            allow_full_host=False,
            run_justification="",
        )

    validate_worker_policy(
        DEFAULT_SHARED_WORKERS + 1,
        allow_full_host=True,
        run_justification="dedicated ML350 data-generation window",
    )
