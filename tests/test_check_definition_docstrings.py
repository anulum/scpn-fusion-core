# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — all-definition documentation gate tests
"""Check documentation enforcement using maintained source and real missing-doc edits."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.check_definition_docstrings import main, missing_docstrings

ROOT = Path(__file__).resolve().parents[1]


def test_documented_maintained_scope_passes() -> None:
    """Private helpers, tests and the gate itself all satisfy the native-doc contract."""
    paths = [
        ROOT / "tools/check_definition_docstrings.py",
        Path(__file__),
        ROOT / "tools/build_rustbca_reference.py",
        ROOT / "tools/check_test_module_linkage.py",
        ROOT / "tests/test_check_test_module_linkage.py",
        *sorted((ROOT / "validation").glob("rustbca_*.py")),
        *sorted((ROOT / "integration_tests/rustbca").glob("*.py")),
    ]
    assert missing_docstrings(paths) == []
    assert main([str(path) for path in paths]) == 0


@pytest.mark.parametrize(
    ("source", "symbol"),
    [
        ("tools/build_rustbca_reference.py", "_sha"),
        ("tools/check_test_module_linkage.py", "TestLinkageIndex"),
        (
            "integration_tests/rustbca/test_builder.py",
            "test_actual_compiler_timeout_reaps_process_group",
        ),
        ("src/scpn_fusion/control/disruption_predictor.py", "__init__"),
        ("tools/check_definition_docstrings.py", "<module>"),
    ],
)
def test_missing_real_definition_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str, symbol: str
) -> None:
    """Removing a real module, class, private, nested or test docstring fails the gate."""
    path = tmp_path / Path(source).name
    text = (ROOT / source).read_text()
    path.write_text(text)
    previous = missing_docstrings([path])
    tree = ast.parse(text)
    if symbol == "<module>":
        owner: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef = tree
    else:
        owner = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == symbol
            and ast.get_docstring(node)
        )
    assert ast.get_docstring(owner)
    doc = owner.body[0]
    assert doc.end_lineno is not None
    lines = text.splitlines(keepends=True)
    lines[doc.lineno - 1 : doc.end_lineno] = [
        " " * doc.col_offset + "''\n",
        *["\n"] * (doc.end_lineno - doc.lineno),
    ]
    path.write_text("".join(lines))
    findings = missing_docstrings([path])
    assert len(findings) == len(previous) + 1
    assert main([str(path)]) == 1
    assert f"missing docstring: {symbol}" in capsys.readouterr().out


def test_empty_scope_cannot_report_success() -> None:
    """The public checker rejects a scope that would silently inspect nothing."""
    with pytest.raises(ValueError, match="at least one file"):
        missing_docstrings([])
