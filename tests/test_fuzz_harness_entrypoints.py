# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fuzz Harness Entrypoint Tests
"""Tests for executable fuzz harness entrypoints over malformed input seeds."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_harness(name: str) -> ModuleType:
    path = ROOT / "fuzz" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geqdsk_fuzz_harness_rejects_absurd_grid_seed() -> None:
    harness = _load_harness("fuzz_geqdsk")
    harness.TestOneInput(b"absurd 0 1000000 1000000\n")


def test_config_fuzz_harness_rejects_deeply_invalid_json_seed() -> None:
    harness = _load_harness("fuzz_fusion_config")
    harness.TestOneInput(b'{"grid_resolution": [1, 1], "dimensions": {}}')


def test_npz_fuzz_harness_rejects_truncated_zip_seed() -> None:
    harness = _load_harness("fuzz_disruption_npz")
    harness.TestOneInput(b"PK\x03\x04truncated")


def test_imas_fuzz_harness_rejects_deeply_invalid_json_seed() -> None:
    harness = _load_harness("fuzz_imas_ids")
    harness.TestOneInput(b'{"ids_properties": {"homogeneous_time": 1}, "time_slice": "bad"}')


def test_snn_artifact_fuzz_harness_rejects_invalid_artifact_seed() -> None:
    harness = _load_harness("fuzz_snn_artifact")
    harness.TestOneInput(b'{"meta": {"artifact_version": 1}, "topology": {"places": []}}')


def test_snn_artifact_fuzz_harness_rejects_non_utf8_seed() -> None:
    harness = _load_harness("fuzz_snn_artifact")
    harness.TestOneInput(b"\xff\xfe\x00\x01not-a-controller")


def test_checked_npz_loader_rejects_oversized_disruption_archive(tmp_path: Path) -> None:
    from scpn_fusion.io.tokamak_disruption_archive import load_disruption_shot
    from scpn_fusion.io.safe_loaders import MAX_NPZ_BYTES

    path = tmp_path / "shot.npz"
    with path.open("wb") as handle:
        handle.truncate(MAX_NPZ_BYTES + 1)

    try:
        load_disruption_shot(path, disruption_dir=tmp_path)
    except ValueError as exc:
        assert "NumPy archive file too large" in str(exc)
    else:
        raise AssertionError("oversized NPZ archive was accepted")


def test_checked_npz_loader_rejects_compressed_expansion_bomb(tmp_path: Path) -> None:
    import numpy as np

    from scpn_fusion.io.safe_loaders import MAX_NPZ_MEMBER_BYTES
    from scpn_fusion.io.tokamak_disruption_archive import load_disruption_shot

    path = tmp_path / "shot.npz"
    oversized_points = MAX_NPZ_MEMBER_BYTES // np.dtype(np.float64).itemsize + 1
    np.savez_compressed(
        path,
        time_s=np.zeros(oversized_points, dtype=np.float64),
        Ip_MA=np.zeros(1),
        BT_T=np.zeros(1),
        beta_N=np.zeros(1),
        q95=np.zeros(1),
        ne_1e19=np.zeros(1),
        n1_amp=np.zeros(1),
        n2_amp=np.zeros(1),
        locked_mode_amp=np.zeros(1),
        dBdt_gauss_per_s=np.zeros(1),
        vertical_position_m=np.zeros(1),
        is_disruption=np.array(False),
        disruption_time_idx=np.array(-1),
        disruption_type=np.array("none"),
    )
    assert path.stat().st_size < MAX_NPZ_MEMBER_BYTES

    with pytest.raises(ValueError, match="member too large"):
        load_disruption_shot(path, disruption_dir=tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "fuzz_geqdsk",
        "fuzz_imas_ids",
        "fuzz_fusion_config",
        "fuzz_disruption_npz",
        "fuzz_snn_artifact",
    ],
)
def test_fuzz_harness_main_instruments_before_starting(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every executable harness instruments loaded Python before fuzzing."""
    events: list[tuple[str, Any]] = []
    fake_atheris = SimpleNamespace(
        instrument_func=lambda target: events.append(("instrument_func", target)),
        Setup=lambda argv, target: events.append(("setup", (argv, target))),
        Fuzz=lambda: events.append(("fuzz", None)),
    )
    monkeypatch.setitem(sys.modules, "atheris", fake_atheris)
    harness = _load_harness(name)

    harness.main()

    assert [event[0] for event in events] == [
        "instrument_func",
        "instrument_func",
        "setup",
        "fuzz",
    ]
    assert events[1][1] is harness.TestOneInput
    assert events[2][1][1] is harness.TestOneInput


def _load_workflow(name: str) -> dict[str, Any]:
    """Load a workflow without YAML 1.1 boolean coercion of the `on` key."""
    payload = yaml.load(
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(payload, dict)
    return payload


def test_python_fuzz_workflow_is_nightly_complete_and_bounded() -> None:
    """The nightly matrix runs exactly the five shipped Atheris targets."""
    workflow = _load_workflow("python-fuzz.yml")
    assert "schedule" in workflow["on"]
    job = workflow["jobs"]["fuzz"]
    rows = job["strategy"]["matrix"]["include"]
    assert {row["target"] for row in rows} == {
        "geqdsk",
        "imas_ids",
        "fusion_config",
        "disruption_npz",
        "snn_artifact",
    }
    assert all(1 <= int(row["max_len"]) <= 1024 * 1024 for row in rows)
    text = (ROOT / ".github" / "workflows" / "python-fuzz.yml").read_text(encoding="utf-8")
    for token in (
        "-atheris_runs=512",
        "-timeout=10",
        "-rss_limit_mb=2048",
        "asan_with_fuzzer.so",
        "ASAN_OPTIONS=detect_leaks=0",
        'LD_PRELOAD="${sanitizer}"',
        "requirements/fuzz.txt",
        'SCPN_DISABLE_JULIA: "1"',
        "GIT_CONFIG_KEY_0: init.defaultBranch",
        "GIT_CONFIG_VALUE_0: main",
        "if: failure()",
    ):
        assert token in text


def test_codeql_workflow_covers_python_rust_and_go() -> None:
    """CodeQL uses supported extraction modes for all repository languages."""
    workflow = _load_workflow("codeql.yml")
    rows = workflow["jobs"]["analyze"]["strategy"]["matrix"]["include"]
    assert {(row["language"], row["build-mode"]) for row in rows} == {
        ("python", "none"),
        ("rust", "none"),
        ("go", "manual"),
    }
    steps = workflow["jobs"]["analyze"]["steps"]
    go_build = next(
        step for step in steps if step.get("name") == "Build Go module for CodeQL extraction"
    )
    assert go_build["if"] == "matrix.language == 'go'"
    assert go_build["working-directory"] == "scpn-fusion-go"
    assert go_build["run"] == "go build ./..."


def test_fuzz_requirement_is_hash_pinned() -> None:
    """Nightly Atheris installation is version and digest pinned."""
    text = (ROOT / "requirements" / "fuzz.txt").read_text(encoding="utf-8")
    assert "atheris==3.1.0" in text
    assert "sha256:ec5e11f21a4c197fe91f7aea2b2de88e623c73a21fc07b105ac6329a1588457b" in text
