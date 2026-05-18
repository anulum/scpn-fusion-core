# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Physics-contract tests for the hybrid transport accuracy benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_hybrid_accuracy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_hybrid_accuracy", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transport_estimator_interpolates_gk_reference_calibration(monkeypatch) -> None:
    module = _load_module()

    def fake_gk_chi(r_l_ti: float) -> tuple[float, float, float]:
        return 2.0 * r_l_ti, 3.0 * r_l_ti, 0.5 * r_l_ti

    monkeypatch.setattr(module, "_gk_chi", fake_gk_chi)

    assert module._gk_calibrated_transport_estimator(6.0) == (12.0, 18.0, 3.0)
    assert module._gk_calibrated_transport_estimator(6.5) == (13.0, 19.5, 3.25)


def test_transport_estimator_rejects_nonfinite_gk_calibration(monkeypatch) -> None:
    module = _load_module()

    monkeypatch.setattr(module, "_gk_chi", lambda _r_l_ti: (np.nan, 1.0, 1.0))

    try:
        module._gk_calibrated_transport_estimator(5.0)
    except ValueError as exc:
        assert "GK calibration" in str(exc)
    else:
        raise AssertionError("Expected invalid GK calibration to fail.")
