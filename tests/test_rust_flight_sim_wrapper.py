from __future__ import annotations

import builtins
import types

import pytest

from scpn_fusion.control import rust_flight_sim_wrapper


def test_run_exits_when_rust_extension_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "scpn_fusion_rs":
            raise ImportError("forced-missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(rust_flight_sim_wrapper.sys, "argv", ["rust_flight_sim_wrapper.py"])
    monkeypatch.delitem(rust_flight_sim_wrapper.sys.modules, "scpn_fusion_rs", raising=False)
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
    monkeypatch.setitem(rust_flight_sim_wrapper.sys.modules, "scpn_fusion_rs", fake_rs)
    monkeypatch.setattr(
        rust_flight_sim_wrapper.sys,
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
