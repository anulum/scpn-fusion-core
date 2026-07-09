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

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "mypy_expansion_guard.py"
SPEC = importlib.util.spec_from_file_location("tools.mypy_expansion_guard", MODULE_PATH)
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


def test_helpers_validate_bad_payload_shapes(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    _write(bad_json, "[]")
    with pytest.raises(ValueError, match="expected JSON object"):
        guard._load_json_object(bad_json)

    no_tool = tmp_path / "no_tool.toml"
    _write(no_tool, "[project]\nname = 'x'\n")
    with pytest.raises(ValueError, match=r"\[tool\]"):
        guard._load_mypy_config(no_tool)

    no_mypy = tmp_path / "no_mypy.toml"
    _write(no_mypy, "[tool.ruff]\nline-length = 100\n")
    with pytest.raises(ValueError, match=r"\[tool.mypy\]"):
        guard._load_mypy_config(no_mypy)

    with pytest.raises(ValueError, match="must be a list"):
        guard._string_list("x", label="bad")
    with pytest.raises(ValueError, match="non-empty string"):
        guard._string_list([""], label="bad")


def test_override_and_baseline_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allowed_missing_import_modules"):
        guard._allowed_missing_import_modules({})

    with pytest.raises(ValueError, match="Duplicate"):
        guard._typed_files({"files": ["a.py", "a.py"]})

    with pytest.raises(ValueError, match="overrides"):
        guard._ignore_missing_import_modules({"overrides": {}})

    with pytest.raises(ValueError, match="must be an object"):
        guard._ignore_missing_import_modules({"overrides": ["bad"]})

    assert guard._module_tuple("numpy", label="module") == ("numpy",)
    assert guard._ignore_missing_import_modules(
        {"overrides": [{"ignore_missing_imports": True, "module": "numpy"}]}
    ) == {("numpy",)}
    assert guard._display_path(tmp_path / "outside.py").endswith("outside.py")


def test_guard_reports_policy_and_missing_file_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, required=[])
    _write(
        pyproject,
        """
[tool.mypy]
strict = true
ignore_missing_imports = true
exclude = "legacy"
files = ["missing.py"]
""",
    )
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    summary = guard.evaluate(pyproject_path=pyproject, baseline_path=baseline)

    assert summary["overall_pass"] is False
    joined = " ".join(summary["failures"])
    assert "ignore_missing_imports" in joined
    assert "exclude" in joined
    assert "configured typed files do not exist" in joined


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


def test_main_resolves_relative_paths_and_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    typed = tmp_path / "typed.py"
    typed.write_text("", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, required=["typed.py"])
    _write(
        pyproject,
        """
[tool.mypy]
strict = true
ignore_missing_imports = false
files = ["typed.py"]
""",
    )
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    rc = guard.main(
        [
            "--pyproject",
            "pyproject.toml",
            "--baseline",
            "baseline.json",
            "--summary-json",
            "summary.json",
        ]
    )

    assert rc == 0
    assert (
        json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["overall_pass"] is True
    )
