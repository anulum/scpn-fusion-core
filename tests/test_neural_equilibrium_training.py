# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests for Neural Equilibrium Training Runtime

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.neural_equilibrium_training import run_training_cli, train_on_sparc


class _DummyTrainingResult:
    def __init__(self) -> None:
        self.n_samples = 1
        self.n_components = 1
        self.explained_variance = 1.0
        self.final_loss = 0.0
        self.train_time_s = 0.0
        self.weights_path = ""
        self.val_loss = 0.0
        self.test_mse = 0.0
        self.test_max_error = 0.0


class _DummyAccel:
    def __init__(self) -> None:
        self.saved_path: Path | None = None
        self.files_seen: list[Path] = []

    def train_from_geqdsk(
        self, files: list[Path], n_perturbations: int, seed: int
    ) -> _DummyTrainingResult:
        self.files_seen = list(files)
        assert n_perturbations > 0
        assert seed >= 0
        return _DummyTrainingResult()

    def save_weights(self, path: str | Path) -> None:
        self.saved_path = Path(path)


def test_train_on_sparc_with_stub_accelerator(monkeypatch, tmp_path: Path) -> None:
    import scpn_fusion.core.neural_equilibrium as ne_mod

    sparc_dir = tmp_path / "sparc"
    sparc_dir.mkdir(parents=True, exist_ok=True)
    (sparc_dir / "shot_001.geqdsk").write_text("stub", encoding="utf-8")
    save_path = tmp_path / "weights_out.npz"

    monkeypatch.setattr(ne_mod, "NeuralEquilibriumAccelerator", _DummyAccel)
    result = train_on_sparc(sparc_dir=sparc_dir, save_path=save_path, n_perturbations=3, seed=7)

    assert result.weights_path == str(save_path)


def test_run_training_cli_returns_1_when_data_missing(monkeypatch, tmp_path: Path) -> None:
    import scpn_fusion.core.neural_equilibrium as ne_mod

    monkeypatch.setattr(ne_mod, "REPO_ROOT", tmp_path)
    assert run_training_cli() == 1


class _CliAccel:
    """Stand-in accelerator covering the CLI's load/predict/benchmark calls."""

    def load_weights(self, path: str) -> None:
        self.loaded = path

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full((4, 4), 0.9)

    def benchmark(self, features: np.ndarray) -> dict[str, float]:
        return {"mean_ms": 0.5, "median_ms": 0.4}


class _FakeEquilibrium:
    """GEQDSK stand-in exposing every attribute the CLI validation reads."""

    rbbbs = np.array([1.0, 1.2, 1.4, 1.6, 1.8])
    zbbbs = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    qpsi = np.array([1.0, 2.0, 3.0, 4.0])
    current = 1.0e6
    bcentr = 5.3
    rmaxis = 1.85
    zmaxis = 0.0
    simag = 0.1
    sibry = 0.2
    psirz = np.full((8, 8), 1.0)


def test_train_on_sparc_defaults_raise_without_files(monkeypatch, tmp_path: Path) -> None:
    import scpn_fusion.core.neural_equilibrium as ne_mod

    # save_path=None -> DEFAULT_WEIGHTS_PATH; sparc_dir=None -> REPO_ROOT default,
    # which has no GEQDSK/EQDSK files, so the loader raises.
    monkeypatch.setattr(ne_mod, "DEFAULT_WEIGHTS_PATH", tmp_path / "weights.npz")
    monkeypatch.setattr(ne_mod, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="No GEQDSK/EQDSK files"):
        train_on_sparc()


def test_run_training_cli_full_path(monkeypatch, tmp_path: Path, capsys) -> None:
    import scpn_fusion.core.eqdsk as eqdsk_mod
    import scpn_fusion.core.neural_equilibrium as ne_mod
    import scpn_fusion.core.neural_equilibrium_training as net_mod

    sparc_dir = tmp_path / "validation" / "reference_data" / "sparc"
    sparc_dir.mkdir(parents=True)
    (sparc_dir / "shot.geqdsk").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(ne_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ne_mod, "NeuralEquilibriumAccelerator", _CliAccel)
    monkeypatch.setattr(net_mod, "train_on_sparc", lambda _dir: _DummyTrainingResult())
    monkeypatch.setattr(eqdsk_mod, "read_geqdsk", lambda _path: _FakeEquilibrium())

    assert run_training_cli() == 0
    out = capsys.readouterr().out
    assert "Training Neural Equilibrium" in out
    assert "Validation relative L2" in out
    assert "Inference:" in out
