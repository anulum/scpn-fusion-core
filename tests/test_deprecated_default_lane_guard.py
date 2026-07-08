# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``tools/deprecated_default_lane_guard.py``."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "deprecated_default_lane_guard.py"
SPEC = importlib.util.spec_from_file_location("tools.deprecated_default_lane_guard", MODULE_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def test_evaluate_passes_when_fno_is_non_default_and_non_public() -> None:
    """Evaluator passes when FNO lanes stay surrogate and non-default."""
    summary = guard.evaluate(
        mode_specs={
            "kernel": {"module": "scpn_fusion.core.fusion_kernel", "maturity": "public"},
            "fno-training": {
                "module": "scpn_fusion.core.fno_jax_training",
                "maturity": "surrogate",
            },
        },
        default_modes=["kernel"],
        release_commands=["scpn-fusion flight --mode=neuro-control"],
    )
    assert summary["overall_pass"] is True
    assert summary["default_contains_deprecated_fno"] is False
    assert summary["fno_public_modes"] == []


def test_evaluate_fails_when_default_includes_deprecated_fno_module() -> None:
    """Default modes fail when they resolve to deprecated FNO modules."""
    summary = guard.evaluate(
        mode_specs={
            "kernel": {
                "module": "scpn_fusion.core.fno_turbulence_suppressor",
                "maturity": "public",
            },
        },
        default_modes=["kernel"],
        release_commands=[],
    )
    assert summary["default_contains_deprecated_fno"] is True
    assert summary["overall_pass"] is False


def test_evaluate_fails_when_fno_mode_is_public() -> None:
    """FNO modes fail when they are marked public."""
    summary = guard.evaluate(
        mode_specs={
            "fno-training": {"module": "scpn_fusion.core.fno_jax_training", "maturity": "public"},
        },
        default_modes=[],
        release_commands=[],
    )
    assert summary["fno_public_modes"] == ["fno-training"]
    assert summary["overall_pass"] is False


def test_evaluate_fails_when_release_docs_expose_fno_without_surrogate_unlock() -> None:
    """Release commands fail when FNO training lacks surrogate unlock flags."""
    summary = guard.evaluate(
        mode_specs={
            "fno-training": {
                "module": "scpn_fusion.core.fno_jax_training",
                "maturity": "surrogate",
            },
        },
        default_modes=[],
        release_commands=["scpn-fusion fno-training"],
    )
    assert summary["docs_violations"] == ["scpn-fusion fno-training"]
    assert summary["overall_pass"] is False


def test_evaluate_allows_release_docs_with_surrogate_unlock() -> None:
    """Release commands may mention FNO training when explicitly surrogate-gated."""
    for command in (
        "scpn-fusion fno-training --surrogate",
        "SCPN_SURROGATE=1 scpn-fusion fno-training",
    ):
        summary = guard.evaluate(
            mode_specs={
                "fno-training": {
                    "module": "scpn_fusion.core.fno_jax_training",
                    "maturity": "surrogate",
                },
            },
            default_modes=[],
            release_commands=[command],
        )
        assert summary["docs_violations"] == []
        assert summary["overall_pass"] is True


def test_main_resolves_absolute_and_repo_relative_summary_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main writes summaries to absolute and repo-relative paths."""
    monkeypatch.setattr(
        guard,
        "_load_runtime_state",
        lambda: ({"kernel": {"module": "scpn_fusion.core.fusion_kernel"}}, ["kernel"]),
    )
    monkeypatch.setattr(guard, "_load_release_commands", lambda: [])
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    absolute = tmp_path / "absolute.json"
    relative = tmp_path / "artifacts" / "relative.json"

    assert guard.main(["--summary-json", str(absolute)]) == 0
    assert guard.main(["--summary-json", "artifacts/relative.json"]) == 0
    assert absolute.exists()
    assert relative.exists()


def test_main_scans_release_surfaces_and_skips_missing_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main scans release docs, ignores missing docs, and reports FNO command violations."""
    present = tmp_path / "README.md"
    missing = tmp_path / "missing.md"
    present.write_text(
        """
        plain prose
          scpn-fusion flight --mode=kernel
        python -m other.tool
        scpn-fusion fno-training
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        guard,
        "_load_runtime_state",
        lambda: (
            {
                "fno-training": {
                    "module": "scpn_fusion.core.fno_jax_training",
                    "maturity": "surrogate",
                }
            },
            [],
        ),
    )
    monkeypatch.setattr(guard, "RELEASE_SURFACES", (present, missing))
    summary_path = tmp_path / "summary.json"

    rc = guard.main(["--summary-json", str(summary_path)])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert rc == 1
    assert payload["release_command_count"] == 2
    assert payload["docs_violations"] == ["scpn-fusion fno-training"]


def test_main_uses_real_runtime_state_and_inserts_absolute_src_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main loads the real CLI mode contract and restores the absolute source root."""
    src_root = str(guard.REPO_ROOT / "src")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != src_root])
    monkeypatch.setattr(guard, "RELEASE_SURFACES", ())
    summary_path = tmp_path / "summary.json"

    rc = guard.main(["--summary-json", str(summary_path)])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert src_root in sys.path
    assert "kernel" in payload["default_modes"]


def test_main_writes_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Main writes a passing summary and returns success."""
    monkeypatch.setattr(
        guard,
        "_load_runtime_state",
        lambda: (
            {
                "kernel": {"module": "scpn_fusion.core.fusion_kernel", "maturity": "public"},
                "fno-training": {
                    "module": "scpn_fusion.core.fno_jax_training",
                    "maturity": "surrogate",
                },
            },
            ["kernel"],
        ),
    )
    monkeypatch.setattr(guard, "_load_release_commands", lambda: ["scpn-fusion flight"])

    summary_path = tmp_path / "summary.json"
    rc = guard.main(["--summary-json", str(summary_path)])
    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True


def test_main_returns_failure_for_policy_violation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Main returns failure while still writing the violation summary."""
    monkeypatch.setattr(
        guard,
        "_load_runtime_state",
        lambda: (
            {"kernel": {"module": "scpn_fusion.core.fno_turbulence_suppressor"}},
            ["kernel"],
        ),
    )
    monkeypatch.setattr(guard, "_load_release_commands", lambda: [])

    summary_path = tmp_path / "summary.json"
    rc = guard.main(["--summary-json", str(summary_path)])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert rc == 1
    assert payload["default_contains_deprecated_fno"] is True


def test_script_entrypoint_exits_with_main_return_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The script entrypoint delegates through ``main`` and exits with its code."""
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [str(MODULE_PATH), "--summary-json", str(summary_path)])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code in (0, 1)
    assert summary_path.exists()
