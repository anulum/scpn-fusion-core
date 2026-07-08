# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Regression tests for tools/check_packaging_contract.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "check_packaging_contract.py"
SPEC = importlib.util.spec_from_file_location("tools.check_packaging_contract", MODULE_PATH)
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


def test_packaging_contract_fallback_parser_handles_requirement_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check_packaging_contract, "tomllib", None)
    payload = check_packaging_contract._load_pyproject(ROOT / "pyproject.toml")
    optional = payload["project"]["optional-dependencies"]

    assert "jax[cuda12]>=0.4.20" in optional["gpu"]
    assert "cupy-cuda12x>=13.6,<14.0" in optional["gpu"]
    assert "nvidia-cuda-nvrtc-cu12>=12.0,<13.0" in optional["gpu"]


def test_inline_array_parser_handles_escapes_and_rejects_unterminated() -> None:
    items = check_packaging_contract._parse_inline_array(
        ['dependencies = ["pkg\\\\name", "quoted \\"value\\"", "tail"]'],
        "dependencies",
    )
    assert len(items) == 3
    assert items[-1] == "tail"

    with pytest.raises(ValueError, match="unterminated"):
        check_packaging_contract._parse_inline_array(['dependencies = ["numpy"'], "dependencies")


def test_load_pyproject_rejects_non_table_tomllib_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\n", encoding="utf-8")
    monkeypatch.setattr(check_packaging_contract, "tomllib", SimpleNamespace(loads=lambda _text: []))

    with pytest.raises(ValueError, match="not a table"):
        check_packaging_contract._load_pyproject(pyproject)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("[project]\ndependencies =\n", r"\[project\]\.dependencies"),
        ("[project.optional-dependencies]\nui =\n", r"\[project\.optional-dependencies\]\.ui"),
    ],
)
def test_fallback_parser_rejects_non_inline_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    match: str,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(body, encoding="utf-8")
    monkeypatch.setattr(check_packaging_contract, "tomllib", None)

    with pytest.raises(ValueError, match=match):
        check_packaging_contract._load_pyproject(pyproject)


def test_fallback_parser_handles_multiline_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = [
  "numpy",
]

[project.optional-dependencies]
ui = [
  "streamlit",
]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_packaging_contract, "tomllib", None)

    payload = check_packaging_contract._load_pyproject(pyproject)

    assert payload["project"]["dependencies"] == ["numpy"]
    assert payload["project"]["optional-dependencies"]["ui"] == ["streamlit"]


def test_fallback_parser_handles_single_line_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = ["numpy"]

[project.optional-dependencies]
ui = ["streamlit"]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_packaging_contract, "tomllib", None)

    payload = check_packaging_contract._load_pyproject(pyproject)

    assert payload["project"]["dependencies"] == ["numpy"]
    assert payload["project"]["optional-dependencies"]["ui"] == ["streamlit"]


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


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"project": []}, r"\[project\] table"),
        ({"project": {"dependencies": "numpy"}}, r"\[project\]\.dependencies"),
        ({"project": {"dependencies": [], "optional-dependencies": []}}, r"optional-dependencies"),
        (
            {"project": {"dependencies": [], "optional-dependencies": {"full": "numpy"}}},
            r"\.full",
        ),
        (
            {
                "project": {
                    "dependencies": [],
                    "optional-dependencies": {"full": [], "ui": "streamlit"},
                }
            },
            r"\.ui",
        ),
    ],
)
def test_packaging_contract_rejects_malformed_metadata(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        check_packaging_contract.evaluate_contract(payload)


def test_main_writes_summary_and_returns_success(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"

    assert (
        check_packaging_contract.main(
            [
                "--pyproject",
                str(ROOT / "pyproject.toml"),
                "--summary-json",
                str(summary_path),
            ]
        )
        == 0
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True


def test_main_resolves_relative_input_and_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = ["numpy"]

[project.optional-dependencies]
ui = ["streamlit"]
ml = ["jax"]
rl = ["gymnasium"]
snn = ["nengo"]
full-physics = ["freegs"]
rust = ["maturin"]
full = ["streamlit", "jax", "gymnasium", "nengo", "freegs", "maturin"]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_packaging_contract, "REPO_ROOT", tmp_path)

    assert check_packaging_contract.main(["--pyproject", "pyproject.toml", "--summary-json", "out.json"]) == 0
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["overall_pass"] is True


def test_main_returns_failure_for_blocked_base_dependency(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    summary_path = tmp_path / "summary.json"
    pyproject.write_text(
        """
[project]
dependencies = ["streamlit"]

[project.optional-dependencies]
ui = ["streamlit"]
ml = ["jax"]
rl = ["gymnasium"]
snn = ["nengo"]
full-physics = ["freegs"]
rust = ["maturin"]
full = ["streamlit", "jax", "gymnasium", "nengo", "freegs", "maturin"]
""",
        encoding="utf-8",
    )

    assert (
        check_packaging_contract.main(
            [
                "--pyproject",
                str(pyproject),
                "--summary-json",
                str(summary_path),
            ]
        )
        == 1
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["blocked_in_base"] == ["streamlit"]
