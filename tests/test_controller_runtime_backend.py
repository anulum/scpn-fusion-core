# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core import _multi_compat as multi
from scpn_fusion.control.rmf_phase_lock import RMFPhaseLockConfig, RMFPhaseLockController
from scpn_fusion.scpn.controller_runtime_backend import probe_rust_runtime_bindings
from tools.remote_mast_digestor import build_digest_report


def test_probe_rust_runtime_bindings_handles_missing_module() -> None:
    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()
    assert isinstance(has_runtime, bool)
    if not has_runtime:
        assert dense_fn is None
        assert update_fn is None
        assert sample_fn is None


def test_probe_rust_runtime_bindings_uses_available_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("scpn_fusion_rs")

    def dense(*_args: object, **_kwargs: object) -> None:
        return None

    def update(*_args: object, **_kwargs: object) -> None:
        return None

    def sample(*_args: object, **_kwargs: object) -> None:
        return None

    fake.__dict__["scpn_dense_activations"] = dense
    fake.__dict__["scpn_marking_update"] = update
    fake.__dict__["scpn_sample_firing"] = sample
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake)

    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()
    assert has_runtime is True
    assert dense_fn is dense
    assert update_fn is update
    assert sample_fn is sample


def test_probe_rust_runtime_bindings_uses_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_dispatch(symbol_name: str) -> Any:
        calls.append(symbol_name)
        return lambda *_args, **_kwargs: symbol_name

    monkeypatch.setattr(multi, "dispatch_rust_symbol", fake_dispatch)

    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()

    assert has_runtime is True
    assert calls == [
        "scpn_dense_activations",
        "scpn_marking_update",
        "scpn_sample_firing",
    ]
    assert dense_fn is not None
    assert update_fn is not None
    assert sample_fn is not None
    assert dense_fn(np.array([1.0]), np.array([0.5])) == "scpn_dense_activations"


def test_probe_rust_runtime_bindings_returns_disabled_when_dispatch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_dispatch(symbol_name: str) -> Any:
        if symbol_name == "scpn_marking_update":
            raise AttributeError(symbol_name)
        return lambda *_args, **_kwargs: symbol_name

    monkeypatch.setattr(multi, "dispatch_rust_symbol", fake_dispatch)

    has_runtime, dense_fn, update_fn, sample_fn = probe_rust_runtime_bindings()

    assert has_runtime is False
    assert dense_fn is not None
    assert update_fn is None
    assert sample_fn is None


def test_rmf_horizon_bounds_phase_error_for_frequency_offset() -> None:
    cfg = RMFPhaseLockConfig(
        f_rmf_nom_hz=1.0e6,
        f_sampling_hz=20.0e6,
        k_p=5.0e8,
        k_d=0.0,
        n_neurons=0,
    )
    ctrl = RMFPhaseLockController(cfg)
    plasma_omega = 2.0 * np.pi * 1.01e6
    steps = np.arange(2_000, dtype=np.float64)
    plasma_phis = np.mod(plasma_omega * ctrl.dt * steps, 2.0 * np.pi)

    antenna_phis = np.asarray(ctrl.step_horizon(plasma_phis))
    phase_error = np.abs(np.sin(antenna_phis - plasma_phis))
    early_mean = float(np.mean(phase_error[100:300]))
    late_mean = float(np.mean(phase_error[1_800:]))

    assert early_mean < 0.40
    assert late_mean < 0.40
    assert abs(late_mean - early_mean) < 0.02
    assert abs(float(ctrl.omega_rmf) - plasma_omega) / plasma_omega < 0.02


def test_remote_mast_digestor_blocks_partial_shot_results() -> None:
    report = build_digest_report(
        target_shots=[30419, 30420],
        results=[
            {
                "shot_id": 30419,
                "mean_residual": 1.0e-6,
                "samples": 10,
                "max_ip_ma": 0.8,
            },
            None,
        ],
        elapsed_s=0.5,
    )

    assert report["status"] == "blocked_incomplete_mast_digest"
    assert report["accepted_full_fidelity_ready"] is False
    assert report["processed_shots"] == [30419]
    assert report["missing_shots"] == [30420]
    assert report["claim_boundary"] == (
        "MAST digest reports are local ingestion diagnostics only; same-case "
        "magnetic-geometry validation is required before physics-validation claims."
    )
