# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real TORAX Runtime Backend Tests
"""Actual TORAX 1.4.3 success, truncation, invalid-config, and timeout tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scpn_fusion.integrations.torax import (
    ToraxFailureCode,
    ToraxRunRequest,
    ToraxRuntimeClient,
)
from scpn_fusion.integrations.torax.serialization import canonical_sha256, load_json_object
from scpn_fusion.integrations.torax.projection import MANIFEST_SCHEMA
from scpn_fusion.integrations.torax.worker import execute_request
from validation.benchmark_torax_runtime_contract import build_request

ROOT = Path(__file__).resolve().parents[1]
TORAX_PYTHON = ROOT / ".venv-torax/bin/python"
pytestmark = pytest.mark.external_reference


def _client() -> ToraxRuntimeClient:
    if not TORAX_PYTHON.is_file():
        pytest.skip("the pinned local TORAX 1.4.3 environment is unavailable")
    return ToraxRuntimeClient(TORAX_PYTHON, working_directory=ROOT)


def test_real_backend_success_retains_every_group_and_critical_projection(tmp_path: Path) -> None:
    """Real TORAX succeeds through the public CLI and its NetCDF is independently readable."""
    request = build_request(dt_s=0.01, request_id="real-success", event_id="event-real")
    sidecar = tmp_path / "real.nc"
    outcome = _client().run(
        request,
        request_path=tmp_path / "real.request.json",
        result_path=tmp_path / "real.result.json",
        sidecar_path=sidecar,
    )
    assert outcome.success
    assert outcome.projection is not None
    assert outcome.projection.time_ns == (0, 10_000_000, 20_000_000)
    assert outcome.projection.scientific_sha256
    manifest = load_json_object(tmp_path / "real.nc.manifest.json")
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert callable(execute_request)
    assert manifest["group_count"] == 4
    assert manifest["variable_count"] == 204
    groups = manifest["groups"]
    assert isinstance(groups, list)
    by_path = {group["path"]: group for group in groups}
    assert set(by_path) == {"/", "/numerics", "/profiles", "/scalars"}
    assert {"T_i", "T_e", "n_e", "psi"} <= set(by_path["/profiles"]["data_variables"])
    assert {"q95", "li3", "beta_N", "W_thermal_total"} <= set(by_path["/scalars"]["data_variables"])


def test_real_backend_truncation_is_failure_with_diagnostic_artifact(tmp_path: Path) -> None:
    """A real max-step truncation retains diagnostics but cannot become successful plant state."""
    request = build_request(
        dt_s=0.01,
        request_id="real-truncated",
        event_id="event-truncated",
        max_steps=1,
    )
    outcome = _client().run(
        request,
        request_path=tmp_path / "truncated.request.json",
        result_path=tmp_path / "truncated.result.json",
        sidecar_path=tmp_path / "truncated.nc",
    )
    assert not outcome.success
    assert outcome.failure_code is ToraxFailureCode.DID_NOT_REACH_T_FINAL
    assert not outcome.complete
    assert outcome.artifact is not None
    assert outcome.projection is not None
    with pytest.raises(Exception, match="DID_NOT_REACH_T_FINAL"):
        outcome.require_success()


def test_real_backend_rejects_invalid_config_and_enforces_timeout(tmp_path: Path) -> None:
    """Actual configuration parsing and process timeout paths return distinct typed failures."""
    payload = build_request(
        dt_s=0.01,
        request_id="invalid-config",
        event_id="event-invalid-config",
    ).to_dict()
    config = copy.deepcopy(payload["torax_config"])
    assert isinstance(config, dict)
    solver = config["solver"]
    assert isinstance(solver, dict)
    solver["solver_type"] = "not-a-real-solver"
    payload["torax_config"] = config
    custody = payload["custody"]
    assert isinstance(custody, dict)
    custody["config_sha256"] = canonical_sha256(config)
    invalid = ToraxRunRequest.from_dict(payload)
    invalid_outcome = _client().run(
        invalid,
        request_path=tmp_path / "invalid.request.json",
        result_path=tmp_path / "invalid.result.json",
        sidecar_path=tmp_path / "invalid.nc",
    )
    assert invalid_outcome.failure_code is ToraxFailureCode.CONFIGURATION_REJECTED

    timeout = build_request(
        dt_s=0.01,
        request_id="real-timeout",
        event_id="event-timeout",
        timeout_s=0.01,
    )
    timeout_outcome = _client().run(
        timeout,
        request_path=tmp_path / "timeout.request.json",
        result_path=tmp_path / "timeout.result.json",
        sidecar_path=tmp_path / "timeout.nc",
    )
    assert timeout_outcome.failure_code is ToraxFailureCode.TIMEOUT
