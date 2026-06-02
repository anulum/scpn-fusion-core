# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MyPy Expansion Guard Tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "mypy_expansion_guard.py"
SPEC = importlib.util.spec_from_file_location("mypy_expansion_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_baseline(path: Path, *, required: list[str]) -> None:
    payload = {
        "allowed_missing_import_modules": [["numpy", "numpy.*"]],
        "min_typed_file_count": len(required),
        "required_typed_files": required,
    }
    _write(path, json.dumps(payload, indent=2))


def test_guard_passes_when_scope_expands_without_new_ignores(tmp_path: Path) -> None:
    typed = tmp_path / "typed.py"
    added = tmp_path / "added.py"
    typed.write_text("", encoding="utf-8")
    added.write_text("", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, required=[str(typed)])
    _write(
        pyproject,
        f"""
[tool.mypy]
strict = true
ignore_missing_imports = false
files = ["{typed}", "{added}"]

[[tool.mypy.overrides]]
module = ["numpy", "numpy.*"]
ignore_missing_imports = true
""",
    )

    summary = guard.evaluate(pyproject_path=pyproject, baseline_path=baseline)

    assert summary["overall_pass"] is True
    assert summary["typed_file_count"] == 2


def test_guard_fails_when_required_typed_file_is_removed(tmp_path: Path) -> None:
    typed = tmp_path / "typed.py"
    typed.write_text("", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, required=[str(typed)])
    _write(
        pyproject,
        """
[tool.mypy]
strict = true
ignore_missing_imports = false
files = []
""",
    )

    summary = guard.evaluate(pyproject_path=pyproject, baseline_path=baseline)

    assert summary["overall_pass"] is False
    assert "required typed files removed" in " ".join(summary["failures"])


def test_guard_fails_on_new_ignore_missing_imports_override(tmp_path: Path) -> None:
    typed = tmp_path / "typed.py"
    typed.write_text("", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, required=[str(typed)])
    _write(
        pyproject,
        f"""
[tool.mypy]
strict = true
ignore_missing_imports = false
files = ["{typed}"]

[[tool.mypy.overrides]]
module = ["new_untyped_lib", "new_untyped_lib.*"]
ignore_missing_imports = true
""",
    )

    summary = guard.evaluate(pyproject_path=pyproject, baseline_path=baseline)

    assert summary["overall_pass"] is False
    assert summary["new_missing_import_overrides"] == [["new_untyped_lib", "new_untyped_lib.*"]]


def test_main_writes_summary_and_returns_nonzero(tmp_path: Path) -> None:
    typed = tmp_path / "typed.py"
    typed.write_text("", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    baseline = tmp_path / "baseline.json"
    summary_path = tmp_path / "summary.json"
    _write_baseline(baseline, required=[str(typed)])
    _write(
        pyproject,
        f"""
[tool.mypy]
strict = false
ignore_missing_imports = false
files = ["{typed}"]
""",
    )

    rc = guard.main(
        [
            "--pyproject",
            str(pyproject),
            "--baseline",
            str(baseline),
            "--summary-json",
            str(summary_path),
        ]
    )

    assert rc == 1
    assert json.loads(summary_path.read_text(encoding="utf-8"))["overall_pass"] is False
