# ----------------------------------------------------------------------
# SCPN Fusion Core -- Training Tool NPZ Hardening Tests
# ----------------------------------------------------------------------
"""Tests for secure .npz loading in training tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_tool_module(name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_train_neural_loader_uses_allow_pickle_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_tool_module(
        "train_neural_transport_qlknn",
        "tools/train_neural_transport_qlknn.py",
    )
    npz_path = tmp_path / "train.npz"
    np.savez(npz_path, X=np.ones((2, 3)), Y=np.zeros((2, 3)))

    real_np_load = np.load
    calls: list[object] = []

    def _fake_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs.get("allow_pickle"))
        return real_np_load(*args, **kwargs)

    monkeypatch.setattr(module.np, "load", _fake_load)
    loaded = module._load_npz_required(npz_path, required_keys=("X", "Y"))
    assert loaded["X"].shape == (2, 3)
    assert loaded["Y"].shape == (2, 3)
    assert calls == [False]


def test_train_neural_loader_rejects_missing_required_key(tmp_path: Path) -> None:
    module = _load_tool_module(
        "train_neural_transport_qlknn_missing",
        "tools/train_neural_transport_qlknn.py",
    )
    npz_path = tmp_path / "train_missing.npz"
    np.savez(npz_path, X=np.ones((2, 3)))

    with pytest.raises(KeyError, match="missing required NPZ keys"):
        module._load_npz_required(npz_path, required_keys=("X", "Y"))


def test_train_fno_loader_uses_allow_pickle_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_tool_module(
        "train_fno_qlknn_spatial",
        "tools/train_fno_qlknn_spatial.py",
    )
    npz_path = tmp_path / "train.npz"
    np.savez(npz_path, X=np.ones((2, 8, 8)), Y=np.zeros((2, 8, 8)))

    real_np_load = np.load
    calls: list[object] = []

    def _fake_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs.get("allow_pickle"))
        return real_np_load(*args, **kwargs)

    monkeypatch.setattr(module.np, "load", _fake_load)
    x, y = module._load_spatial_split(npz_path)
    assert x.shape == (2, 8, 8)
    assert y.shape == (2, 8, 8)
    assert calls == [False]


def test_train_fno_loader_rejects_missing_required_keys(tmp_path: Path) -> None:
    module = _load_tool_module(
        "train_fno_qlknn_spatial_missing",
        "tools/train_fno_qlknn_spatial.py",
    )
    npz_path = tmp_path / "val_missing.npz"
    np.savez(npz_path, X=np.ones((2, 8, 8)))

    with pytest.raises(KeyError, match="missing required NPZ keys"):
        module._load_spatial_split(npz_path)
