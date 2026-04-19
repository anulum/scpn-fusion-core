# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for extracted TGLF validation runtime module."""

from __future__ import annotations

import numpy as np

import scpn_fusion.core.tglf_interface as tglf_mod
from scpn_fusion.core.tglf_validation_runtime import validate_against_tglf


class _DummyTransport:
    def __init__(self) -> None:
        self.rho = np.linspace(0.0, 1.0, 9, dtype=np.float64)
        self.Te = np.linspace(8.0, 12.0, 9, dtype=np.float64)
        self.Ti = np.linspace(7.5, 11.5, 9, dtype=np.float64)
        self.ne = np.linspace(7.0, 9.0, 9, dtype=np.float64)
        self.chi_i = np.linspace(0.8, 1.6, 9, dtype=np.float64)
        self.chi_e = np.linspace(1.2, 2.0, 9, dtype=np.float64)
        self._chi_i_profile = np.ones(9, dtype=np.float64) * 9.0
        self._chi_e_profile = np.ones(9, dtype=np.float64) * 7.0
        self.R0 = 6.2
        self.a = 2.0


def test_validate_against_tglf_with_stubbed_binary(monkeypatch) -> None:
    def _stub_run(deck, _path):
        return tglf_mod.TGLFOutput(
            rho=deck.rho,
            chi_i=1.0,
            chi_e=0.5,
            gamma_max=0.1,
        )

    monkeypatch.setattr(tglf_mod, "run_tglf_binary", _stub_run)
    result = validate_against_tglf(
        _DummyTransport(),
        tglf_binary_path="C:/fake/tglf",
        rho_indices=[2, 3, 4],
    )
    assert result.case_name == "Live TGLF validation"
    assert len(result.rho_points) == 3


def test_validate_against_tglf_prefers_public_transport_profiles(monkeypatch) -> None:
    def _stub_run(deck, _path):
        return tglf_mod.TGLFOutput(
            rho=deck.rho,
            chi_i=1.0,
            chi_e=0.5,
            gamma_max=0.1,
        )

    monkeypatch.setattr(tglf_mod, "run_tglf_binary", _stub_run)
    result = validate_against_tglf(
        _DummyTransport(),
        tglf_binary_path="C:/fake/tglf",
        rho_indices=[2, 4, 6],
    )
    expected_i = np.interp(result.rho_points, np.linspace(0.0, 1.0, 9), np.linspace(0.8, 1.6, 9))
    expected_e = np.interp(result.rho_points, np.linspace(0.0, 1.0, 9), np.linspace(1.2, 2.0, 9))
    np.testing.assert_allclose(result.our_chi_i, expected_i)
    np.testing.assert_allclose(result.our_chi_e, expected_e)
