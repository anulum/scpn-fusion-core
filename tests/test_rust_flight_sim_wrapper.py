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

import pytest

from scpn_fusion.core import _multi_compat as multi
from scpn_fusion.control import rust_flight_sim_wrapper


def test_run_exits_when_rust_extension_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_missing(_symbol_name: str) -> Any:
        raise ImportError("forced-missing")

    monkeypatch.setattr(multi, "dispatch_rust_symbol", _raise_missing)
    monkeypatch.setattr(sys, "argv", ["rust_flight_sim_wrapper.py"])
    monkeypatch.delitem(sys.modules, "scpn_fusion_rs", raising=False)
    with pytest.raises(SystemExit) as exc:
        rust_flight_sim_wrapper.run()
    assert exc.value.code == 1


def test_run_executes_when_rust_extension_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Report:
        steps = 1000
        wall_time_ms = 1.5
        mean_abs_r_error = 0.01
        disrupted = False

    class _Sim:
        def __init__(self, _r_target: float, _z_target: float, _hz: float) -> None:
            pass

        def run_shot(self, _duration: float, deterministic: bool = False) -> _Report:
            assert deterministic is True
            return _Report()

    fake_rs = types.SimpleNamespace(PyRustFlightSim=_Sim)
    monkeypatch.setattr(
        multi,
        "dispatch_rust_symbol",
        lambda symbol_name: getattr(fake_rs, symbol_name),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rust_flight_sim_wrapper.py",
            "--hz",
            "5000",
            "--duration",
            "0.25",
            "--deterministic",
        ],
    )

    rust_flight_sim_wrapper.run()


def test_run_resolves_flight_sim_through_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Report:
        steps = 10
        wall_time_ms = 0.25
        mean_abs_r_error = 0.02
        disrupted = False

    class _Sim:
        def __init__(self, _r_target: float, _z_target: float, _hz: float) -> None:
            pass

        def run_shot(self, _duration: float, deterministic: bool = False) -> _Report:
            return _Report()

    def fake_dispatch(symbol_name: str) -> Any:
        calls.append(symbol_name)
        return _Sim

    monkeypatch.setattr(multi, "dispatch_rust_symbol", fake_dispatch)
    monkeypatch.setattr(sys, "argv", ["rust_flight_sim_wrapper.py"])

    rust_flight_sim_wrapper.run()

    assert calls == ["PyRustFlightSim"]
