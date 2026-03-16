# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Kuramoto Edge Path Tests
"""Coverage for GlobalPsiDriver unknown mode (line 84),
upde.py psi_mode guards (86, 92), and eqdsk psi_to_norm (103)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.phase.knm import KnmSpec
from scpn_fusion.phase.kuramoto import GlobalPsiDriver
from scpn_fusion.phase.upde import UPDESystem


class TestGlobalPsiDriver:
    def test_unknown_mode_raises(self):
        """Unknown mode raises ValueError (line 84)."""
        driver = GlobalPsiDriver(mode="bogus")
        theta = np.zeros(10)
        with pytest.raises(ValueError, match="Unknown mode"):
            driver.resolve(theta, psi_external=0.0)


class TestUPDEPsiMode:
    def _make_system(self, psi_mode: str) -> UPDESystem:
        L = 3
        K = np.ones((L, L)) * 0.1
        spec = KnmSpec(K=K)
        return UPDESystem(spec=spec, psi_mode=psi_mode)

    def test_external_mode_no_driver_raises(self):
        """psi_mode='external' without psi_driver raises (line 86)."""
        sys = self._make_system("external")
        theta = [np.zeros(5) for _ in range(3)]
        omega = [np.ones(5) for _ in range(3)]
        with pytest.raises(ValueError, match="psi_driver"):
            sys.step(theta, omega)

    def test_unknown_psi_mode_raises(self):
        """Unknown psi_mode raises ValueError (line 92)."""
        sys = self._make_system("bogus")
        theta = [np.zeros(5) for _ in range(3)]
        omega = [np.ones(5) for _ in range(3)]
        with pytest.raises(ValueError, match="Unknown psi_mode"):
            sys.step(theta, omega)
