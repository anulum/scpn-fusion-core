# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Original Rust checkpoint API contracts
"""Exercise real PyO3 owner snapshots, restoration and recipient boundaries."""

from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.io.safe_loaders import checked_json_load


class _NativeKernel(Protocol):
    """Actual native interface exercised directly through the loaded extension."""

    def snapshot_state(self) -> object:
        """Return the original opaque token issued by this native owner."""
        ...

    def restore_state(self, checkpoint: object) -> None:
        """Restore only original state issued by the same native owner."""
        ...

    def get_psi(self) -> NDArray[np.float64]:
        """Return actual copied native flux values."""
        ...

    def get_j_phi(self) -> NDArray[np.float64]:
        """Return actual copied native current values."""
        ...

    def solve_equilibrium(self) -> object:
        """Execute the real retained native equilibrium solver."""
        ...

    def set_solver_method(self, method: str) -> None:
        """Select one maintained native relaxation method."""
        ...

    def solver_method(self) -> str:
        """Return the actual selected native method."""
        ...


def _kernel(tmp_path: Path) -> _NativeKernel:
    """Construct a real bounded native owner from the original production schema."""
    root = Path(__file__).resolve().parents[1]
    config = checked_json_load(root / "validation/iter_config.json")
    config["grid_resolution"] = [17, 17]
    solver = config["solver"]
    assert isinstance(solver, dict)
    solver["max_iterations"] = 2
    solver["convergence_threshold"] = 1e-12
    path = tmp_path / "native_checkpoint.json"
    path.write_text(json.dumps(config))
    module = importlib.import_module("scpn_fusion_rs")
    kernel: _NativeKernel = module.PyFusionKernel(str(path))
    return kernel


def _values(kernel: _NativeKernel) -> tuple[bytes, bytes, str]:
    """Retain actual binary64 values and the public native method selection."""
    return kernel.get_psi().tobytes(), kernel.get_j_phi().tobytes(), kernel.solver_method()


def test_actual_cold_native_token_restores_state_and_method(tmp_path: Path) -> None:
    """Restore the original cold native state after actual nonlinear execution.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Exact original-schema bounded solver configuration namespace.
    """
    kernel = _kernel(tmp_path)
    before = _values(kernel)
    checkpoint = kernel.snapshot_state()
    kernel.set_solver_method("multigrid")
    kernel.solve_equilibrium()
    assert _values(kernel) != before
    kernel.restore_state(checkpoint)
    assert _values(kernel) == before
    kernel.restore_state(checkpoint)
    assert _values(kernel) == before


def test_actual_warm_native_token_is_independent_and_replays(tmp_path: Path) -> None:
    """Keep original warm bytes independent of later native work and exported arrays.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Actual native configuration and output namespace.
    """
    kernel = _kernel(tmp_path)
    kernel.solve_equilibrium()
    before = _values(kernel)
    checkpoint = kernel.snapshot_state()
    exported = kernel.get_psi()
    exported[:] = np.nan
    assert _values(kernel) == before
    kernel.set_solver_method("multigrid")
    kernel.solve_equilibrium()
    assert _values(kernel) != before
    kernel.restore_state(checkpoint)
    assert _values(kernel) == before
    kernel.solve_equilibrium()
    assert _values(kernel) == before


def test_actual_native_token_cannot_transfer_to_identical_owner(tmp_path: Path) -> None:
    """Refuse another actual owner before changing any recipient state.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Identical explicit configuration used by two independent kernels.
    """
    origin, recipient = _kernel(tmp_path), _kernel(tmp_path)
    checkpoint = origin.snapshot_state()
    recipient.set_solver_method("multigrid")
    recipient.solve_equilibrium()
    before = _values(recipient)
    with pytest.raises(ValueError, match="belongs to another kernel"):
        recipient.restore_state(checkpoint)
    assert _values(recipient) == before


def test_actual_native_token_has_no_caller_state_constructor(tmp_path: Path) -> None:
    """Keep caller bytes, construction and serialized state outside restoration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Original bounded native owner namespace.
    """
    kernel = _kernel(tmp_path)
    checkpoint = kernel.snapshot_state()
    before = _values(kernel)
    with pytest.raises(TypeError):
        kernel.restore_state(b"caller-state")
    module = importlib.import_module("scpn_fusion_rs")
    with pytest.raises(TypeError):
        module.PyKernelCheckpoint()
    with pytest.raises(TypeError):
        pickle.dumps(checkpoint)
    assert _values(kernel) == before
