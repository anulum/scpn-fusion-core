# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — entry-point tests of four training tools
"""Each training tool runs end to end from its CLI and saves weights without pickles.

The NumPy 2 migration changed the ``np.savez(..., allow_pickle=False, ...)`` call in
each of these tools, and no test ran it. Every test here drives the real ``main()``
with small settings, writes into a temporary folder, and checks the saved archive
loads with ``allow_pickle=False`` and holds exactly the model arrays. Nothing is
stubbed: the FRC tools use the Rust equilibrium solver, and the neural-equilibrium
tool reads two of the repository's GEQDSK files.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import train_fno_qlknn_spatial as fno
from tools import train_frc_snn_surrogate as frc_snn
from tools import train_frc_surrogate as frc_mlp
from tools import train_neural_eq_v3 as neural_eq

REPO_ROOT = Path(__file__).resolve().parents[1]


def _archive_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            assert archive[name].dtype != object
        return sorted(archive.files)


def _write_spatial_splits(data_dir: Path, *, offset: float) -> None:
    """Write 8x8 train/val splits whose target equals the input: ``offset`` plus a wave."""
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    data_dir.mkdir(parents=True)
    for split, count in (("train", 32), ("val", 8)):
        amplitude = rng.uniform(0.5, 1.5, (count, 1, 1))
        phase = rng.uniform(0.0, 2.0 * np.pi, (count, 1, 1))
        wave = np.sin(grid[None, :, None] + phase) * np.cos(grid[None, None, :])
        field = offset + 0.3 * amplitude * wave
        np.savez(data_dir / f"{split}.npz", X=field, Y=field.copy())


def _fno_argv(data_dir: Path, output: Path) -> list[str]:
    return [
        "train_fno_qlknn_spatial.py",
        "--data-dir",
        str(data_dir),
        "--output",
        str(output),
        "--modes",
        "2",
        "--width",
        "4",
        "--n-layers",
        "1",
        "--epochs",
        "20",
        "--lr",
        "0.01",
        "--batch-size",
        "8",
    ]


def test_fno_cli_saves_weights_and_metrics_once_the_gate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    output = tmp_path / "out" / "fno.npz"
    # Positive fields: the FNO projects through softplus, as for transport fluxes.
    _write_spatial_splits(data_dir, offset=1.0)
    monkeypatch.setattr(sys, "argv", _fno_argv(data_dir, output))

    fno.main()

    assert _archive_keys(output) == sorted(
        ["lift_w", "lift_b", "w0_real", "w0_imag", "b0", "skip0", "fc1", "fc1_b", "fc2", "fc2_b"]
    )
    metrics = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["val_relative_l2"] < 0.40
    assert metrics["grid_size"] == 8


def test_fno_cli_refuses_to_save_a_model_that_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    output = tmp_path / "out" / "fno.npz"
    # Sign-changing targets cannot be reached through the softplus head.
    _write_spatial_splits(data_dir, offset=0.0)
    monkeypatch.setattr(sys, "argv", _fno_argv(data_dir, output))

    with pytest.raises(SystemExit) as raised:
        fno.main()

    assert raised.value.code == 1
    assert not output.parent.exists()


def test_frc_snn_cli_saves_the_spiking_surrogate_and_reports_its_rate_in_khz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "out" / "snn.npz"
    argv = ["train_frc_snn_surrogate.py", "--samples", "16", "--grid", "16", "--out", str(output)]
    monkeypatch.setattr(sys, "argv", argv)

    with caplog.at_level(logging.INFO, logger=frc_snn.__name__):
        frc_snn.main()

    assert _archive_keys(output) == ["W", "W_out", "b", "b_out", "pca_components", "pca_mean"]
    [record] = [r for r in caplog.records if "inference latency" in r.getMessage()]
    assert isinstance(record.args, tuple)
    latency_us, rate_khz = record.args
    assert isinstance(latency_us, float) and isinstance(rate_khz, float)
    # One inference per latency_us microseconds is 1000 / latency_us kilohertz.
    assert rate_khz == pytest.approx(1000.0 / latency_us)


def test_frc_mlp_cli_saves_every_layer_and_the_normalisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out" / "mlp.npz"
    argv = [
        "train_frc_surrogate.py",
        "--samples",
        "16",
        "--grid",
        "16",
        "--epochs",
        "2",
        "--out",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    frc_mlp.main()

    layers = len(frc_mlp.HIDDEN_SIZES) + 1
    expected = [f"w{i}" for i in range(layers)] + [f"b{i}" for i in range(layers)]
    expected += ["grid_size", "input_mean", "input_std", "n_components"]
    assert _archive_keys(output) == sorted([*expected, "pca_components", "pca_mean"])
    with np.load(output, allow_pickle=False) as archive:
        assert int(archive["grid_size"][0]) == 16


def test_neural_eq_cli_trains_from_a_chosen_reference_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference"
    for machine, name in (("jet", "jet_lmode_2MA.geqdsk"), ("diiid", "diiid_snowflake.geqdsk")):
        (reference / machine).mkdir(parents=True)
        shutil.copy(
            REPO_ROOT / "validation" / "reference_data" / machine / name, reference / machine
        )
    output = tmp_path / "out" / "neural_eq.npz"
    argv = [
        "train_neural_eq_v3.py",
        "--reference-dir",
        str(reference),
        "--output",
        str(output),
        "--epochs",
        "2",
        "--perturbations",
        "30",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    status = neural_eq.main()

    keys = _archive_keys(output)
    assert {"pca_mean", "pca_components", "input_mean", "input_std", "w0", "b4"} <= set(keys)
    metrics = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["epochs_run"] == 2
    assert metrics["n_samples"] == 2 * 31
    assert set(metrics["per_machine"]) == {"jet", "diiid"}
    assert metrics["acceptance"] == ("PASS" if status == 0 else "FAIL")
    assert status in (0, 2)


@pytest.mark.parametrize(("flag", "value"), [("--epochs", "0"), ("--perturbations", "-1")])
def test_neural_eq_cli_rejects_settings_it_cannot_train_with(
    monkeypatch: pytest.MonkeyPatch, flag: str, value: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["train_neural_eq_v3.py", flag, value])

    with pytest.raises(SystemExit) as raised:
        neural_eq.main()

    assert raised.value.code == 2
