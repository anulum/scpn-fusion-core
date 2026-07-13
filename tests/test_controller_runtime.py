# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neuro-Symbolic Logic Compiler
"""Runtime-selection, marking-setter, and traceable-step guards of the controller.

These exercise the branches the main controller suite never reaches: the
constructor's ``runtime_profile`` / ``runtime_backend`` validation, the
rust→numpy backend fallback telemetry (both the explicit-``rust`` request and
the ``auto`` path when the Rust runtime is absent), the ``marking`` setter
success path, and ``step_traceable`` — the not-ready guard, the oracle-enabled
branch, and the JSONL logging branch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest

from scpn_fusion.scpn.artifact import Artifact, load_artifact, save_artifact
from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
from scpn_fusion.scpn.structure import StochasticPetriNet
from scpn_fusion.scpn import controller as controller_mod
from scpn_fusion.scpn.controller import NeuroSymbolicController


def _build_controller_net() -> StochasticPetriNet:
    """8-place, 4-transition pass-through controller net."""
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("x_Z_pos", initial_tokens=0.0)
    net.add_place("x_Z_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_place("a_Z_pos", initial_tokens=0.0)
    net.add_place("a_Z_neg", initial_tokens=0.0)
    net.add_transition("T_Rp", threshold=0.1, delay_ticks=0)
    net.add_transition("T_Rn", threshold=0.1, delay_ticks=0)
    net.add_transition("T_Zp", threshold=0.1, delay_ticks=0)
    net.add_transition("T_Zn", threshold=0.1, delay_ticks=0)
    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("x_Z_pos", "T_Zp", weight=1.0)
    net.add_arc("x_Z_neg", "T_Zn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.add_arc("T_Zp", "a_Z_pos", weight=1.0)
    net.add_arc("T_Zn", "a_Z_neg", weight=1.0)
    net.compile()
    return net


def _build_artifact(tmp_path: Path) -> Artifact:
    """Compile the pass-through net, export, and round-trip through disk."""
    net = _build_controller_net()
    compiler = FusionCompiler(bitstream_length=1024, seed=42)
    compiled = compiler.compile(net, firing_mode="binary", firing_margin=0.05)
    readout_config = {
        "actions": [
            {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
            {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
        ],
        "gains": [1000.0, 1000.0],
        "abs_max": [5000.0, 5000.0],
        "slew_per_s": [1e6, 1e6],
    }
    injection_config = [
        {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    ]
    artifact = compiled.export_artifact(
        name="test_controller_runtime",
        dt_control_s=0.001,
        readout_config=readout_config,
        injection_config=injection_config,
    )
    path = tmp_path / "artifact.scpnctl.json"
    save_artifact(artifact, str(path))
    return load_artifact(str(path))


def _make_controller(
    artifact: Artifact,
    *,
    runtime_profile: str = "adaptive",
    runtime_backend: str = "auto",
    enable_oracle_diagnostics: bool = True,
) -> NeuroSymbolicController:
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=123456789,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        runtime_profile=runtime_profile,
        runtime_backend=runtime_backend,
        enable_oracle_diagnostics=enable_oracle_diagnostics,
    )


class TestConstructorRuntimeGuards:
    """Reject malformed runtime-selection strings."""

    def test_rejects_unknown_runtime_profile(self, tmp_path: Path) -> None:
        artifact = _build_artifact(tmp_path)
        with pytest.raises(ValueError, match="runtime_profile must be"):
            _make_controller(artifact, runtime_profile="turbo")

    def test_rejects_unknown_runtime_backend(self, tmp_path: Path) -> None:
        artifact = _build_artifact(tmp_path)
        with pytest.raises(ValueError, match="runtime_backend must be"):
            _make_controller(artifact, runtime_backend="cuda")


class TestBackendFallbackTelemetry:
    """Rust→numpy fallback records a telemetry event and selects numpy."""

    def _patch_no_rust(self, monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []

        def _capture(component: str, reason: str, **_: object) -> None:
            events.append((component, reason))

        monkeypatch.setattr(controller_mod, "_HAS_RUST_SCPN_RUNTIME", False)
        monkeypatch.setattr(controller_mod, "_record_fallback_event", _capture)
        return events

    def test_explicit_rust_request_without_runtime_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = self._patch_no_rust(monkeypatch)
        artifact = _build_artifact(tmp_path)
        controller = _make_controller(artifact, runtime_backend="rust")
        assert controller.runtime_backend_name == "numpy"
        assert ("scpn_controller", "rust_backend_unavailable") in events

    def test_auto_backend_falls_back_when_rust_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = self._patch_no_rust(monkeypatch)
        artifact = _build_artifact(tmp_path)
        controller = _make_controller(artifact, runtime_backend="auto")
        assert controller.runtime_backend_name == "numpy"
        assert ("scpn_controller", "auto_backend_numpy_due_to_missing_rust") in events


class TestMarkingSetter:
    """The marking setter clamps and stores a correctly shaped vector."""

    def test_setter_accepts_valid_length(self, tmp_path: Path) -> None:
        artifact = _build_artifact(tmp_path)
        controller = _make_controller(artifact)
        n = len(controller.marking)
        controller.marking = [1.5] + [0.0] * (n - 1)
        stored = controller.marking
        assert len(stored) == n
        # 1.5 is clamped into [0, 1].
        assert stored[0] == 1.0

    def test_setter_rejects_wrong_length(self, tmp_path: Path) -> None:
        artifact = _build_artifact(tmp_path)
        controller = _make_controller(artifact)
        with pytest.raises(ValueError, match="marking must have length"):
            controller.marking = [0.0, 0.0]


class TestStepTraceable:
    """Traceable-step guard, oracle branch, and JSONL logging."""

    def test_not_ready_with_passthrough_source(self, tmp_path: Path) -> None:
        artifact = _build_artifact(tmp_path)
        # A non-axis injection source makes the traceable fast path unavailable.
        artifact.initial_state.place_injections[0].source = "external_probe"
        controller = _make_controller(artifact)
        with pytest.raises(RuntimeError, match="passthrough sources"):
            controller.step_traceable([0.5, 0.1], 0)

    def test_traceable_step_runs_oracle_path(self, tmp_path: Path) -> None:
        artifact = _build_artifact(tmp_path)
        controller = _make_controller(artifact, enable_oracle_diagnostics=True)
        actions = controller.step_traceable([0.5, 0.1], 0)
        assert actions.shape == (2,)
        assert controller.last_oracle_marking  # oracle diagnostics populated

    def test_traceable_step_writes_jsonl_log(self, tmp_path: Path) -> None:
        artifact = _build_artifact(tmp_path)
        controller = _make_controller(artifact)
        log_path = tmp_path / "trace.jsonl"
        controller.step_traceable([0.5, 0.1], 0, log_path=str(log_path))
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["k"] == 0
        assert set(record["obs"]) == {"R_axis_m", "Z_axis_m"}
        assert "actions" in record
