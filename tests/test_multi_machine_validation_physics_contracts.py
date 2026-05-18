# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Physics-contract tests for multi-machine validation diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "multi_machine_validation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("multi_machine_validation", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_soft_xray_uses_chord_resolved_forward_brightness(monkeypatch) -> None:
    module = _load_module()
    suite = module.SyntheticDiagnosticSuite()
    rho = np.linspace(0.0, 1.0, 96)
    te = 0.6 + 7.0 * (1.0 - rho**2) ** 1.5
    ne = 1.0 + 8.0 * (1.0 - rho**2) ** 0.5

    monkeypatch.setattr(module.np.random, "randn", lambda n: np.zeros(n))

    brightness = suite.soft_xray(te, ne, rho, n_chords=9)

    assert brightness.shape == (9,)
    assert np.all(np.isfinite(brightness))
    assert brightness[0] > brightness[-1] * 2.0
    assert not np.allclose(brightness, brightness[0])


def test_soft_xray_preserves_density_squared_scaling(monkeypatch) -> None:
    module = _load_module()
    suite = module.SyntheticDiagnosticSuite()
    rho = np.linspace(0.0, 1.0, 96)
    te = 1.0 + 5.0 * (1.0 - rho**2)
    ne = 1.0 + 6.0 * (1.0 - rho**2)

    monkeypatch.setattr(module.np.random, "randn", lambda n: np.zeros(n))

    base = suite.soft_xray(te, ne, rho, n_chords=7)
    doubled_density = suite.soft_xray(te, 2.0 * ne, rho, n_chords=7)

    np.testing.assert_allclose(doubled_density / base, 4.0, rtol=0.03, atol=0.0)
