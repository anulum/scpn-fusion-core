# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime Client Tests
"""Real process and artifact-boundary tests for the TORAX runtime client."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from scpn_fusion.integrations.torax.__main__ import main as torax_cli_main
from scpn_fusion.integrations.torax.client import ToraxRuntimeClient
from scpn_fusion.integrations.torax.contracts import ToraxFailureCode
from scpn_fusion.integrations.torax.serialization import (
    file_sha256,
    load_json_object,
    write_json_atomic,
)
from validation.benchmark_torax_runtime_contract import build_request

ROOT = Path(__file__).resolve().parents[1]


def test_missing_and_non_torax_python_fail_through_public_process_boundary(tmp_path: Path) -> None:
    """Missing and ordinary project interpreters yield typed backend-unavailable outcomes."""
    request = build_request(dt_s=0.01, request_id="unavailable", event_id="event-u")
    missing = ToraxRuntimeClient(tmp_path / "missing-python", working_directory=ROOT).run(
        request,
        request_path=tmp_path / "missing.request.json",
        result_path=tmp_path / "missing.result.json",
        sidecar_path=tmp_path / "missing.nc",
    )
    assert missing.failure_code is ToraxFailureCode.BACKEND_UNAVAILABLE

    ordinary = ToraxRuntimeClient(ROOT / ".venv/bin/python", working_directory=ROOT).run(
        request,
        request_path=tmp_path / "ordinary.request.json",
        result_path=tmp_path / "ordinary.result.json",
        sidecar_path=tmp_path / "ordinary.nc",
    )
    assert ordinary.failure_code is ToraxFailureCode.BACKEND_UNAVAILABLE
    assert (
        load_json_object(tmp_path / "ordinary.result.json")["failure_code"] == "BACKEND_UNAVAILABLE"
    )


def test_cli_invalid_request_writes_typed_failure_record(tmp_path: Path) -> None:
    """The public one-request CLI atomically reports malformed input instead of crashing."""
    request_path = tmp_path / "invalid.json"
    result_path = tmp_path / "result.json"
    request_path.write_text('{"request_id":"bad","event_id":"event-bad"}\n', encoding="utf-8")
    process = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "scpn_fusion.integrations.torax",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
            "--output-sidecar",
            str(tmp_path / "invalid.nc"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    outcome = load_json_object(result_path)
    assert outcome["failure_code"] == "INVALID_REQUEST"
    assert outcome["request_id"] == "bad"
    assert callable(torax_cli_main)


def test_tracked_fixture_verification_detects_post_run_sidecar_corruption(tmp_path: Path) -> None:
    """Portable consumer loading verifies the complete sidecar and manifest before use."""
    source_result = load_json_object(
        ROOT / "validation/reference_data/torax/torax_runtime_result_v1.json"
    )
    source_sidecar = ROOT / "validation/reference_data/torax/torax_runtime_primary_v1.nc"
    source_manifest = (
        ROOT / "validation/reference_data/torax/torax_runtime_primary_v1.nc.manifest.json"
    )
    sidecar = tmp_path / "copy.nc"
    manifest_path = tmp_path / "copy.nc.manifest.json"
    sidecar.write_bytes(source_sidecar.read_bytes())
    manifest = load_json_object(source_manifest)
    write_json_atomic(manifest_path, manifest)
    result = copy.deepcopy(source_result)
    artifact = result["artifact"]
    assert isinstance(artifact, dict)
    artifact["sidecar_path"] = str(sidecar)
    artifact["manifest_path"] = str(manifest_path)
    artifact["manifest_sha256"] = file_sha256(manifest_path)
    result_path = tmp_path / "result.json"
    write_json_atomic(result_path, result)
    client = ToraxRuntimeClient(ROOT / ".venv/bin/python", working_directory=ROOT)
    assert client.load_verified_outcome(
        result_path=result_path,
        expected_sidecar_path=sidecar,
    ).success
    payload = bytearray(sidecar.read_bytes())
    payload[len(payload) // 2] ^= 1
    sidecar.write_bytes(payload)
    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        client.load_verified_outcome(
            result_path=result_path,
            expected_sidecar_path=sidecar,
        )


def test_import_surface_does_not_import_torax_or_jax() -> None:
    """The ordinary consumer process can import contracts with both backends blocked."""
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torax' or name.startswith('torax.') or name == 'jax' or name.startswith('jax.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from scpn_fusion.integrations.torax import ToraxRunRequest, ToraxRuntimeClient
print(ToraxRunRequest.__name__, ToraxRuntimeClient.__name__)
"""
    process = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert process.stdout.strip() == "ToraxRunRequest ToraxRuntimeClient"
