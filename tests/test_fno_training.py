# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FNO Training Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from pathlib import Path
import warnings

import numpy as np
import pytest

from scpn_fusion.core.fno_training import train_fno
from scpn_fusion.core.fno_turbulence_suppressor import (
    FNO_Controller,
    SpectralTurbulenceGenerator,
    run_fno_simulation,
)

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = [pytest.mark.experimental, pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")]


def test_fno_training_smoke(tmp_path):
    output = tmp_path / "fno_smoke.npz"
    history = train_fno(
        n_samples=8,
        epochs=1,
        lr=1e-3,
        modes=4,
        width=4,
        save_path=output,
        batch_size=2,
        seed=7,
        patience=1,
    )
    assert output.exists()
    assert history["epochs_completed"] >= 1
    assert np.isfinite(float(history["best_val_loss"]))


def test_fno_controller_loads_saved_weights(tmp_path):
    from scpn_fusion.core.fno_jax_training import init_fno_params, MODES, WIDTH
    from jax import random as jrandom

    output = tmp_path / "fno_controller.npz"
    key = jrandom.PRNGKey(13)
    params = init_fno_params(key, MODES, WIDTH)
    np.savez(output, **{k: np.array(v) for k, v in params.items()})

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FNO turbulence surrogate.*",
            category=Warning,
        )
        controller = FNO_Controller(weights_path=str(output))
    suppression, prediction = controller.predict_and_suppress(np.zeros((64, 64), dtype=np.float64))

    assert controller.loaded_weights
    assert prediction.shape == (64, 64)
    assert np.isfinite(suppression)
    assert 0.0 <= suppression <= 1.0


def test_load_fno_params_rejects_object_array_payload(tmp_path: Path) -> None:
    from scpn_fusion.core.fno_jax_training import load_fno_params

    bad = tmp_path / "bad_fno_params.npz"
    np.savez(bad, w1_real=np.array([{"bad": 1}], dtype=object))
    with pytest.raises(ValueError):
        load_fno_params(str(bad))


def test_train_fno_jax_accepts_real_dataset_input(tmp_path: Path) -> None:
    from scpn_fusion.core.fno_jax_training import train_fno_jax

    rng = np.random.default_rng(17)
    data_path = tmp_path / "real_dataset.npz"
    np.savez(
        data_path,
        X=rng.normal(size=(2, 64, 64, 1)).astype(np.float32),
        Y=np.asarray([0.2, 0.35], dtype=np.float32),
    )
    weights_path = tmp_path / "fno_real_weights.npz"
    summary = train_fno_jax(
        data_path=str(data_path),
        epochs=1,
        batch_size=1,
        save_path=str(weights_path),
        seed=23,
    )

    assert weights_path.exists()
    assert summary["data_source"] == "real"
    assert summary["n_samples"] == 2
    assert summary["epochs_completed"] == 1
    assert np.isfinite(float(summary["final_loss"]))


def test_train_fno_jax_rejects_invalid_dataset_schema(tmp_path: Path) -> None:
    from scpn_fusion.core.fno_jax_training import train_fno_jax

    bad = tmp_path / "bad_dataset.npz"
    np.savez(bad, X=np.zeros((2, 64, 64, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="contain 'X' and 'Y'"):
        train_fno_jax(
            data_path=str(bad),
            epochs=1,
            batch_size=1,
            save_path=str(tmp_path / "unused.npz"),
            seed=23,
        )


def test_load_gene_binary_accepts_valid_npz(tmp_path: Path) -> None:
    from scpn_fusion.core.fno_jax_training import load_gene_binary

    rng = np.random.default_rng(123)
    path = tmp_path / "gene_like.npz"
    np.savez(
        path,
        X=rng.normal(size=(3, 64, 64, 1)).astype(np.float32),
        Y=np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32),
    )
    x_np, y_np = load_gene_binary(str(path))
    assert x_np.shape == (3, 64, 64, 1)
    assert y_np.shape == (3,)
    assert np.all(np.isfinite(x_np))
    assert np.all(np.isfinite(y_np))


def test_load_gene_binary_rejects_unsupported_extension(tmp_path: Path) -> None:
    from scpn_fusion.core.fno_jax_training import load_gene_binary

    path = tmp_path / "gene_like.npy"
    np.save(path, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="Unsupported GENE dataset format"):
        load_gene_binary(str(path))


def test_spectral_generator_is_deterministic_for_seed() -> None:
    g1 = SpectralTurbulenceGenerator(size=24, seed=77)
    g2 = SpectralTurbulenceGenerator(size=24, seed=77)
    np.testing.assert_allclose(g1.field, g2.field, rtol=0.0, atol=0.0)
    s1 = g1.step()
    s2 = g2.step()
    np.testing.assert_allclose(s1, s2, rtol=0.0, atol=0.0)


def test_spectral_generator_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(4242)
    state = np.random.get_state()

    g = SpectralTurbulenceGenerator(size=20, seed=12)
    _ = g.step()

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


def test_run_fno_simulation_returns_finite_summary_without_plot() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FNO turbulence surrogate.*",
            category=Warning,
        )
        summary = run_fno_simulation(time_steps=24, seed=7, save_plot=False, verbose=False)
    for key in (
        "seed",
        "steps",
        "loaded_weights",
        "final_energy",
        "mean_energy_last_20",
        "max_energy",
        "final_suppression",
        "plot_saved",
    ):
        assert key in summary
    assert summary["seed"] == 7
    assert summary["steps"] == 24
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["final_energy"])
    assert np.isfinite(summary["mean_energy_last_20"])
    assert np.isfinite(summary["max_energy"])


def test_run_fno_simulation_is_deterministic_for_seed() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FNO turbulence surrogate.*",
            category=Warning,
        )
        a = run_fno_simulation(time_steps=18, seed=19, save_plot=False, verbose=False)
        b = run_fno_simulation(time_steps=18, seed=19, save_plot=False, verbose=False)
    assert a["final_energy"] == b["final_energy"]
    assert a["mean_energy_last_20"] == b["mean_energy_last_20"]
    assert a["max_energy"] == b["max_energy"]
