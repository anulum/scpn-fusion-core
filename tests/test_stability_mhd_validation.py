# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests: MHD stability input validation branches

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.stability_mhd import (
    QProfile,
    compute_q_profile,
    mercier_stability,
)

_NR = 50
_RHO = np.linspace(0.0, 1.0, _NR)
_NE = 10.0 * np.ones(_NR)
_TI = np.ones(_NR)
_TE = np.ones(_NR)
_R0 = 6.2
_A = 2.0
_B0 = 5.3
_IP_MA = 15.0


def _valid_qp() -> QProfile:
    """A physically valid safety-factor profile from the shape-aware builder."""
    return compute_q_profile(_RHO, _NE, _TI, _TE, _R0, _A, _B0, _IP_MA)


class TestComputeQProfileInputValidation:
    """compute_q_profile validates the raw profile arrays before building q."""

    def test_rho_must_be_one_dimensional(self) -> None:
        rho2d = np.ones((3, 3))
        with pytest.raises(ValueError, match="rho must be one-dimensional"):
            compute_q_profile(rho2d, np.ones(9), np.ones(9), np.ones(9), _R0, _A, _B0, _IP_MA)

    def test_rho_needs_at_least_three_points(self) -> None:
        rho = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="at least 3 points"):
            compute_q_profile(rho, np.ones(2), np.ones(2), np.ones(2), _R0, _A, _B0, _IP_MA)

    def test_profiles_must_be_finite(self) -> None:
        rho = np.linspace(0.0, 1.0, 3)
        ne = np.array([10.0, np.nan, 10.0])
        with pytest.raises(ValueError, match="finite values"):
            compute_q_profile(rho, ne, np.ones(3), np.ones(3), _R0, _A, _B0, _IP_MA)

    def test_profiles_must_be_non_negative(self) -> None:
        rho = np.linspace(0.0, 1.0, 3)
        ne = np.array([10.0, -1.0, 10.0])
        with pytest.raises(ValueError, match="non-negative"):
            compute_q_profile(rho, ne, np.ones(3), np.ones(3), _R0, _A, _B0, _IP_MA)


class TestValidateQProfileGuards:
    """_validate_q_profile (via mercier_stability) rejects each malformed field."""

    def test_non_qprofile_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a QProfile"):
            mercier_stability("not-a-qprofile")  # type: ignore[arg-type]

    def test_rho_needs_three_points(self) -> None:
        qp = _valid_qp()
        qp.rho = np.array([0.0, 0.5])
        with pytest.raises(ValueError, match="at least 3 points"):
            mercier_stability(qp)

    def test_arrays_must_share_shape(self) -> None:
        qp = _valid_qp()
        qp.q = qp.q[:-1]
        with pytest.raises(ValueError, match="same one-dimensional shape"):
            mercier_stability(qp)

    def test_arrays_must_be_finite(self) -> None:
        qp = _valid_qp()
        q = qp.q.copy()
        q[1] = np.nan
        qp.q = q
        with pytest.raises(ValueError, match="finite values"):
            mercier_stability(qp)

    def test_rho_must_strictly_increase(self) -> None:
        qp = _valid_qp()
        qp.rho = np.linspace(1.0, 0.0, _NR)
        with pytest.raises(ValueError, match="strictly increasing"):
            mercier_stability(qp)

    def test_rho_must_lie_within_unit_interval(self) -> None:
        qp = _valid_qp()
        qp.rho = np.linspace(0.0, 2.0, _NR)
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            mercier_stability(qp)

    def test_q_must_be_strictly_positive(self) -> None:
        qp = _valid_qp()
        q = qp.q.copy()
        q[1] = 0.0
        qp.q = q
        with pytest.raises(ValueError, match="q must be strictly positive"):
            mercier_stability(qp)

    def test_q_edge_must_be_positive(self) -> None:
        qp = _valid_qp()
        qp.q_edge = -1.0
        with pytest.raises(ValueError, match="q_edge must be finite"):
            mercier_stability(qp)

    def test_q_min_must_be_positive(self) -> None:
        qp = _valid_qp()
        qp.q_min = -1.0
        with pytest.raises(ValueError, match="q_min must be finite"):
            mercier_stability(qp)

    def test_q_min_rho_must_be_finite(self) -> None:
        qp = _valid_qp()
        qp.q_min_rho = float("inf")
        with pytest.raises(ValueError, match="q_min_rho must be finite"):
            mercier_stability(qp)
