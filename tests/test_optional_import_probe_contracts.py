# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Optional Import Probe Contract Tests
"""Regression tests for optional backend import probes."""

from __future__ import annotations

import sys
import types

import pytest

from scpn_fusion.core import _multi_compat
from scpn_fusion.core.gk_interface import GKLocalParams
from scpn_fusion.core.gk_qualikiz import _try_qualikiz_python


class _NonMappingQuaLiKiz(types.ModuleType):
    """QuaLiKiz API double that returns a malformed payload."""

    def run(self, **_kwargs: object) -> list[object]:
        """Return a non-mapping result that the adapter must reject."""
        return ["chi_i", 1.0]


def test_qualikiz_python_rejects_non_mapping_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QuaLiKiz Python API results must expose mapping-style transport fields."""
    monkeypatch.setitem(sys.modules, "qualikiz_tools", _NonMappingQuaLiKiz("qualikiz_tools"))

    params = GKLocalParams(R_L_Ti=6.0, R_L_Te=6.0, R_L_ne=2.0, q=1.4, s_hat=0.8)

    assert _try_qualikiz_python(params) is None


def test_julia_probe_rejects_module_without_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Julia probing fails closed when juliacall lacks the expected Main facade."""
    monkeypatch.delenv("SCPN_DISABLE_JULIA", raising=False)
    monkeypatch.setitem(sys.modules, "juliacall", types.ModuleType("juliacall"))

    assert _multi_compat._probe_julia() is False
