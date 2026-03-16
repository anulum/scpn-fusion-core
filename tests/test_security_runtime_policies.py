# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Runtime security policy guards for deserialization and subprocess usage."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "validation", "tools")


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_DIRS:
        base = ROOT / rel
        if not base.exists():
            continue
        files.extend(sorted(base.rglob("*.py")))
    return files


def test_numpy_load_always_disables_pickle() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "load":
                continue
            receiver = node.func.value
            is_numpy_call = (
                isinstance(receiver, ast.Name)
                and receiver.id == "np"
                or isinstance(receiver, ast.Name)
                and receiver.id == "numpy"
            )
            if not is_numpy_call:
                continue
            allow_kw = next((kw for kw in node.keywords if kw.arg == "allow_pickle"), None)
            if allow_kw is None:
                violations.append(f"{path}:{node.lineno} missing allow_pickle=False")
                continue
            if not isinstance(allow_kw.value, ast.Constant) or allow_kw.value.value is not False:
                violations.append(f"{path}:{node.lineno} allow_pickle must be False")
    assert not violations, "\n".join(violations)


def test_subprocess_calls_do_not_enable_shell_mode() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"run", "Popen"}:
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "subprocess":
                continue
            shell_kw = next((kw for kw in node.keywords if kw.arg == "shell"), None)
            if shell_kw is None:
                continue
            if isinstance(shell_kw.value, ast.Constant) and shell_kw.value.value is True:
                violations.append(f"{path}:{node.lineno} subprocess.{node.func.attr}(shell=True)")
    assert not violations, "\n".join(violations)
