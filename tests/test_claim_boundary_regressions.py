# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Claim Boundary Regression Tests

from __future__ import annotations

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
    assert status["status"] == "reference_available_validation_pending"


def test_placeholder_hardware_exports_fail_closed(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="FPGA export is not implemented"):
        RMFPhaseLockController().export_to_fpga(str(tmp_path))

    with pytest.raises(NotImplementedError, match="Verilog export is not implemented"):
        export_to_verilog({"W": np.zeros((1, 1)), "b": np.zeros(1)}, tmp_path / "frc_snn_core.v")


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
