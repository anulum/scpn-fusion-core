# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tests for Neural Equilibrium Training Runtime
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path

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
