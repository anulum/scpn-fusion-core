# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Solver State
"""Complete Python and native history for an outer equilibrium transaction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_fusion.core.fusion_kernel_numerics import FloatArray
from scpn_fusion.hpc._hpc_solver_state import HPCSolverState


@dataclass
class FreeBoundaryState:
    """Owned fields, currents and inner result from one equilibrium execution.

    Parameters
    ----------
    fields : dict[str, FloatArray]
        Flux, source and magnetic fields on the same grid.
    currents : FloatArray
        Physical coil currents used by that execution.
    inner_result : dict[str, Any] or None
        Copied actual inner result; absent for an unsolved pre-call state.
    native : HPCSolverState or None
        Complete history of an available native solver.
    absent_fields : tuple[str, ...]
        Fields not yet allocated in an unsolved pre-call kernel.
    """

    fields: dict[str, FloatArray]
    currents: FloatArray
    inner_result: dict[str, Any] | None
    native: HPCSolverState | None
    absent_fields: tuple[str, ...]

    @classmethod
    def capture(
        cls, kernel: Any, coils: Any, inner_result: dict[str, Any] | None = None
    ) -> FreeBoundaryState:
        """Copy bound state and native history without mutating the kernel.

        Parameters
        ----------
        kernel : Any
            Public equilibrium kernel.
        coils : Any
            Physical coil request used by the current solve.
        inner_result : dict[str, Any] or None
            Actual result of the latest execution, if present.

        Returns
        -------
        FreeBoundaryState
            Owned complete transaction state.
        """
        native = kernel.hpc.snapshot_state() if kernel.hpc.is_available() else None
        fields = {
            name: np.asarray(getattr(kernel, name), dtype=np.float64).copy()
            for name in ("Psi", "J_phi", "B_R", "B_Z")
            if hasattr(kernel, name)
        }
        return cls(
            fields,
            np.asarray(coils.currents, dtype=np.float64).copy(),
            deepcopy(inner_result),
            native,
            tuple(name for name in ("Psi", "J_phi", "B_R", "B_Z") if name not in fields),
        )

    def restore(self, kernel: Any, coils: Any) -> None:
        """Restore native and Python state before another candidate is run.

        Parameters
        ----------
        kernel : Any
            Originating equilibrium kernel.
        coils : Any
            Coil request whose currents must match the restored fields.
        """
        if self.native is not None:
            kernel.hpc.restore_state(self.native)
        for name, field in self.fields.items():
            if hasattr(kernel, name):
                getattr(kernel, name)[...] = field
            else:
                setattr(kernel, name, field.copy())
        for name in self.absent_fields:
            if hasattr(kernel, name):
                delattr(kernel, name)
        coils.currents = self.currents.copy()
