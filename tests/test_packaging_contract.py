# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Regression tests for tools/check_packaging_contract.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_packaging_contract.py"
SPEC = importlib.util.spec_from_file_location("check_packaging_contract", MODULE_PATH)
assert SPEC and SPEC.loader
check_packaging_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_packaging_contract)


def test_packaging_contract_passes_current_pyproject() -> None:
    payload = check_packaging_contract._load_pyproject(ROOT / "pyproject.toml")
    summary = check_packaging_contract.evaluate_contract(payload)
    assert summary["overall_pass"] is True
    assert summary["blocked_in_base"] == []
    assert summary["missing_required_extras"] == []
    assert summary["missing_from_full_extra"] == []


def test_packaging_contract_detects_blocked_base_dependency() -> None:
    payload = {
        "project": {
            "dependencies": ["numpy", "streamlit"],
            "optional-dependencies": {
                "ui": ["streamlit"],
                "ml": ["jax<0.5.0"],
                "rl": ["gymnasium>=1.0.0"],
                "snn": ["nengo>=4.0"],
                "full-physics": ["freegs>=0.6"],
                "rust": ["maturin>=1.7,<2.0"],
                "full": [
                    "streamlit",
                    "jax<0.5.0",
                    "gymnasium>=1.0.0",
                    "nengo>=4.0",
                    "freegs>=0.6",
                    "maturin>=1.7,<2.0",
                ],
            },
        }
    }
    summary = check_packaging_contract.evaluate_contract(payload)
    assert summary["overall_pass"] is False
    assert "streamlit" in summary["blocked_in_base"]
