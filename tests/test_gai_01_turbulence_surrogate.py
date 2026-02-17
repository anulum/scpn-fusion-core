# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GAI-01 Turbulence Surrogate Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GAI-01 synthetic turbulence surrogate validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from scpn_fusion.core.gyro_swin_surrogate import generate_synthetic_gyrokinetic_dataset


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gai_01_turbulence_surrogate.py"
SPEC = importlib.util.spec_from_file_location("gai_01_turbulence_surrogate", MODULE_PATH)
assert SPEC and SPEC.loader
gai_01_turbulence_surrogate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gai_01_turbulence_surrogate)


def test_dataset_is_deterministic_for_seed() -> None:
    a = generate_synthetic_gyrokinetic_dataset(seed=123, samples=32)
    b = generate_synthetic_gyrokinetic_dataset(seed=123, samples=32)
    np.testing.assert_allclose(a.features, b.features, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(a.chi_i, b.chi_i, rtol=0.0, atol=0.0)


def test_campaign_meets_thresholds_smoke() -> None:
    out = gai_01_turbulence_surrogate.run_campaign(
        seed=42, train_samples=1024, eval_samples=256, benchmark_samples=64
    )
    assert out["rmse_pct"] <= 10.0
    # Speedup is hardware-dependent; smoke test uses small benchmark_samples
    # so require only 50x (the campaign default threshold of 1000x targets
    # larger sample sizes on production hardware).
    assert out["speedup_vs_gene_proxy"] >= 50.0


def test_render_markdown_contains_key_sections() -> None:
    report = gai_01_turbulence_surrogate.generate_report(
        seed=11, train_samples=512, eval_samples=192, benchmark_samples=64
    )
    text = gai_01_turbulence_surrogate.render_markdown(report)
    assert "GAI-01 Turbulence Surrogate Validation" in text
    assert "RMSE (% of mean target)" in text
    assert "Speedup vs GENE-like proxy" in text


def test_campaign_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(6060)
    state = np.random.get_state()

    _ = gai_01_turbulence_surrogate.run_campaign(
        seed=5, train_samples=512, eval_samples=160, benchmark_samples=48
    )

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected
