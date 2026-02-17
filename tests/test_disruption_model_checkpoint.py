# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Predictor Checkpoint Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Checkpoint load/train tests for disruption predictor."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

import scpn_fusion.control.disruption_predictor as dp


pytestmark = pytest.mark.skipif(dp.torch is None, reason="torch is required for checkpoint tests")


def test_load_or_train_creates_checkpoint_when_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "disruption_model.pth"
    model, meta = dp.load_or_train_predictor(
        model_path=model_path,
        seq_len=32,
        train_kwargs={
            "seq_len": 32,
            "n_shots": 8,
            "epochs": 1,
            "seed": 7,
            "save_plot": False,
        },
    )
    assert model is not None
    assert meta["trained"] is True
    assert meta["seq_len"] == 32
    assert model_path.exists()


def test_load_or_train_reuses_existing_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "disruption_model.pth"

    dp.train_predictor(
        seq_len=24,
        n_shots=8,
        epochs=1,
        model_path=model_path,
        seed=11,
        save_plot=False,
    )
    model, meta = dp.load_or_train_predictor(
        model_path=model_path,
        seq_len=64,
        force_retrain=False,
        train_kwargs={
            "seq_len": 64,
            "n_shots": 8,
            "epochs": 1,
            "seed": 12,
            "save_plot": False,
        },
    )
    assert model is not None
    assert meta["trained"] is False
    assert meta["seq_len"] == 24
    assert Path(meta["model_path"]) == model_path


def test_load_or_train_returns_fallback_on_corrupted_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "corrupted_model.pth"
    model_path.write_bytes(b"not a valid torch checkpoint")

    model, meta = dp.load_or_train_predictor(
        model_path=model_path,
        seq_len=32,
        train_if_missing=False,
        allow_fallback=True,
    )

    assert model is None
    assert meta["trained"] is False
    assert meta["fallback"] is True
    assert meta["reason"].startswith("checkpoint_load_failed:")
    assert meta["seq_len"] == 32


def test_predict_disruption_risk_safe_falls_back_on_corrupted_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "corrupted_model.pth"
    model_path.write_bytes(b"still not a valid torch checkpoint")

    signal = np.linspace(0.25, 0.95, 80)
    toroidal = {
        "toroidal_n1_amp": 0.16,
        "toroidal_n2_amp": 0.07,
        "toroidal_n3_amp": 0.03,
        "toroidal_asymmetry_index": 0.177,
        "toroidal_radial_spread": 0.02,
    }
    expected = dp.predict_disruption_risk(signal, toroidal)

    risk, meta = dp.predict_disruption_risk_safe(
        signal,
        toroidal,
        model_path=model_path,
        train_if_missing=False,
    )

    assert abs(risk - expected) < 1e-12
    assert meta["mode"] == "fallback"
    assert meta["risk_source"] == "predict_disruption_risk"
    assert meta["reason"].startswith("checkpoint_load_failed:")


def test_disruption_transformer_uses_batch_first_attention() -> None:
    model = dp.DisruptionTransformer(seq_len=16)
    layer = model.transformer.layers[0]
    assert layer.self_attn.batch_first is True


def test_disruption_transformer_forward_preserves_output_contract() -> None:
    model = dp.DisruptionTransformer(seq_len=20)
    x = dp.torch.rand((3, 20, 1), dtype=dp.torch.float32)

    model.eval()
    with dp.torch.no_grad():
        y = model(x)

    assert y.shape == (3, 1)
    assert bool(dp.torch.isfinite(y).all())
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((20, 1), "shape"),
        ((2, 0, 1), "sequence length"),
        ((2, 20, 2), "feature dimension"),
    ],
)
def test_disruption_transformer_rejects_invalid_input_shapes(
    shape: tuple[int, ...], match: str
) -> None:
    model = dp.DisruptionTransformer(seq_len=20)
    bad_input = dp.torch.zeros(shape, dtype=dp.torch.float32)
    with pytest.raises(ValueError, match=match):
        model(bad_input)


def test_train_predictor_does_not_emit_nested_tensor_warning(tmp_path: Path) -> None:
    model_path = tmp_path / "warning_check.pth"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        dp.train_predictor(
            seq_len=24,
            n_shots=8,
            epochs=1,
            model_path=model_path,
            seed=19,
            save_plot=False,
        )

    assert model_path.exists()
    messages = [str(w.message) for w in captured]
    assert not any("enable_nested_tensor" in msg for msg in messages)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"seq_len": 7}, "seq_len"),
        ({"seq_len": 16.5}, "seq_len"),
        ({"n_shots": 7}, "n_shots"),
        ({"n_shots": 8.5}, "n_shots"),
        ({"epochs": 0}, "epochs"),
        ({"epochs": 1.5}, "epochs"),
        ({"seed": -1}, "seed"),
        ({"seed": 3.5}, "seed"),
    ],
)
def test_train_predictor_rejects_invalid_training_inputs(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    params: dict[str, object] = {
        "seq_len": 16,
        "n_shots": 8,
        "epochs": 1,
        "model_path": tmp_path / "invalid.pth",
        "seed": 3,
        "save_plot": False,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        dp.train_predictor(**params)
