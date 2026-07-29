# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for split-conformal surrogate certificates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_surrogate_conformal_certificates.py"
COMMITTED = ROOT / "validation" / "surrogate_conformal_certificates.json"
COMMITTED_WEIGHTS = ROOT / "weights" / "neural_transport_qlknn.npz"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "generate_surrogate_conformal_certificates", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_weights(path: Path, *, width: int = 10) -> None:
    hidden = 4
    np.savez(
        path,
        input_mean=np.zeros(width),
        input_std=np.ones(width),
        output_scale=np.ones(3),
        version=np.asarray(1),
        log_transform=np.asarray(0),
        gb_scale=np.asarray(0),
        gated=np.asarray(0),
        w1=np.zeros((width, hidden)),
        b1=np.zeros(hidden),
        w2=np.zeros((hidden, 3)),
        b2=np.zeros(3),
    )


def _write_split(
    path: Path,
    *,
    rows: int = 12,
    width: int = 10,
    offset: float = 0.0,
    gb_normalized: bool | None = None,
) -> None:
    x = np.ones((rows, width), dtype=np.float64)
    x[:, 0] = np.linspace(0.1, 0.9, rows)
    x[:, 1] = 2.0
    x[:, 2] = 3.0
    x[:, 4] = 6.0
    x[:, 5] = 7.0
    x[:, 7] = 2.0
    y = np.full((rows, 3), np.log(2.0) + offset, dtype=np.float64)
    payload: dict[str, Any] = {"X": x, "Y": y}
    if gb_normalized is not None:
        payload["gb_normalized"] = np.asarray(int(gb_normalized))
    np.savez(path, **payload)


def _build_fixture(tmp_path: Path) -> tuple[Any, Path, Path, Path, dict[str, Any]]:
    module = _load_module()
    weights = tmp_path / "weights.npz"
    calibration = tmp_path / "calibration.npz"
    holdout = tmp_path / "holdout.npz"
    _write_weights(weights)
    _write_split(calibration)
    _write_split(holdout, offset=0.05)
    payload = module.build_certificate(
        weights_path=weights,
        calibration_path=calibration,
        holdout_path=holdout,
        alpha=0.2,
    )
    return module, weights, calibration, holdout, payload


def test_committed_certificate_is_structurally_valid_and_weight_bound() -> None:
    """The committed radii bind the current weights and clear holdout coverage."""
    module = _load_module()
    payload = json.loads(COMMITTED.read_text(encoding="utf-8"))
    module.validate_certificate(payload, weights_path=COMMITTED_WEIGHTS)
    certificate = payload["certificates"][0]
    assert certificate["calibration"]["n"] == 47_826
    assert min(channel["holdout_empirical_coverage"] for channel in certificate["channels"]) >= 0.9


def test_finite_sample_rank_validation() -> None:
    """The corrected rank accepts valid inputs and rejects non-finite bounds."""
    module = _load_module()
    assert module.finite_sample_rank(12, 0.2) == 11
    with pytest.raises(ValueError, match="positive"):
        module.finite_sample_rank(0, 0.2)
    for alpha in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            module.finite_sample_rank(12, alpha)
    with pytest.raises(ValueError, match="too small"):
        module.finite_sample_rank(1, 0.1)


def test_model_feature_contracts() -> None:
    """Processed 10D inputs reproduce the runtime 12D/14D feature expansion."""
    module = _load_module()
    ten = np.ones((2, 10), dtype=np.float64)
    ten[:, 1] = 2.0
    ten[:, 2] = 4.0
    ten[:, 4] = 6.0
    ten[:, 5] = 7.0
    twelve = module._model_features(ten, 12)
    assert twelve.shape == (2, 12)
    assert np.allclose(twelve[:, 10], 2.0)
    fourteen = module._model_features(ten, 14)
    assert fourteen.shape == (2, 14)
    assert np.allclose(fourteen[:, 12:], [[3.0, 1.0], [3.0, 1.0]])
    assert module._model_features(fourteen, 14) is fourteen
    eleven = np.ones((2, 11), dtype=np.float64)
    with pytest.raises(ValueError, match="cannot satisfy"):
        module._model_features(eleven, 14)


def test_target_unit_conversion_matches_runtime_gyro_bohm_scale() -> None:
    """Normalized targets are scaled exactly when the runtime emits physical units."""
    module = _load_module()
    features = np.ones((2, 10), dtype=np.float64)
    features[:, 1] = [1.0, 4.0]
    targets = np.ones((2, 3), dtype=np.float64)
    unchanged_physical, physical_unit = module._targets_in_prediction_space(
        features, targets, gb_normalized=False, model_gb_scale=True
    )
    unchanged_normalized, normalized_unit = module._targets_in_prediction_space(
        features, targets, gb_normalized=True, model_gb_scale=False
    )
    scaled, scaled_unit = module._targets_in_prediction_space(
        features, targets, gb_normalized=True, model_gb_scale=True
    )
    assert unchanged_physical is targets
    assert unchanged_normalized is targets
    assert physical_unit == scaled_unit == "physical transport coefficient (m^2/s)"
    assert normalized_unit == "gyro-Bohm-normalized transport coefficient"
    assert np.all(scaled > 0.0)
    assert np.all(scaled[1] > scaled[0])


def test_build_certificate_and_external_path(tmp_path: Path) -> None:
    """Synthetic residuals produce deterministic channel certificates."""
    module, weights, calibration, holdout, payload = _build_fixture(tmp_path)
    module.validate_certificate(payload, weights_path=weights)
    module.validate_certificate(payload)
    certificate = payload["certificates"][0]
    assert certificate["calibration"]["quantile_rank_one_based"] == 11
    assert certificate["model"]["weights"].startswith("/")
    assert [channel["radius"] for channel in certificate["channels"]] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert [channel["holdout_empirical_coverage"] for channel in certificate["channels"]] == [
        0.0,
        0.0,
        0.0,
    ]
    assert module._sha256_file(calibration) == certificate["calibration"]["split_sha256"]
    assert len(module._sha256_array(np.ones(3, dtype=np.float64))) == 64
    assert holdout.is_file()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema="wrong"), "schema"),
        (lambda payload: payload.update(method=[]), "method"),
        (lambda payload: payload["method"].update(alpha=True), "alpha"),
        (lambda payload: payload.update(certificates=[]), "Exactly one"),
        (lambda payload: payload["certificates"].__setitem__(0, []), "entry"),
        (
            lambda payload: payload["certificates"][0].update(id="wrong"),
            "QLKNN",
        ),
        (
            lambda payload: payload["certificates"][0].update(calibration=[]),
            "Calibration metadata",
        ),
        (
            lambda payload: payload["certificates"][0]["calibration"].update(n=True),
            "Calibration n",
        ),
        (
            lambda payload: payload["certificates"][0]["calibration"].update(
                quantile_rank_one_based=1
            ),
            "quantile rank",
        ),
        (
            lambda payload: payload["certificates"][0].update(channels=[]),
            "channels",
        ),
        (
            lambda payload: payload["certificates"][0]["channels"][0].update(radius=-1.0),
            "radius",
        ),
        (
            lambda payload: payload["certificates"][0]["channels"][0].update(
                holdout_empirical_coverage=2.0
            ),
            "coverage",
        ),
        (
            lambda payload: payload["certificates"][0].update(model=[]),
            "weight digest",
        ),
    ],
)
def test_certificate_validation_rejects_mutations(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    """Malformed certificate fields fail closed before publication."""
    module, weights, _, _, payload = _build_fixture(tmp_path)
    mutation(payload)
    kwargs = {"weights_path": weights} if message == "weight digest" else {}
    with pytest.raises(ValueError, match=message):
        module.validate_certificate(payload, **kwargs)


def test_split_loader_rejects_bad_payloads(tmp_path: Path) -> None:
    """Bounded NPZ loading rejects absent, malformed, and non-finite arrays."""
    module = _load_module()
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="size"):
        module._load_split(empty)

    missing = tmp_path / "missing.npz"
    np.savez(missing, X=np.ones((2, 10)))
    with pytest.raises(ValueError, match="X and Y"):
        module._load_split(missing)

    misaligned = tmp_path / "misaligned.npz"
    np.savez(misaligned, X=np.ones((2, 10)), Y=np.ones((3, 3)))
    with pytest.raises(ValueError, match="aligned"):
        module._load_split(misaligned)

    width = tmp_path / "width.npz"
    np.savez(width, X=np.ones((2, 10)), Y=np.ones((2, 2)))
    with pytest.raises(ValueError, match="3 output"):
        module._load_split(width)

    nonfinite = tmp_path / "nonfinite.npz"
    np.savez(nonfinite, X=np.full((2, 10), np.nan), Y=np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        module._load_split(nonfinite)


def test_predict_rejects_bad_model_and_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Inference rejects unavailable weights and invalid output matrices."""
    module = _load_module()
    features = np.ones((2, 10), dtype=np.float64)
    with pytest.raises(ValueError, match="Could not load"):
        module._predict(tmp_path / "missing.npz", features)

    weights = tmp_path / "weights.npz"
    _write_weights(weights)
    monkeypatch.setattr(module, "_mlp_forward", lambda *_: np.ones((2, 2)))
    with pytest.raises(ValueError, match="unexpected shape"):
        module._predict(weights, features)
    monkeypatch.setattr(module, "_mlp_forward", lambda *_: np.full((2, 3), np.nan))
    with pytest.raises(ValueError, match="finite"):
        module._predict(weights, features)


def test_build_rejects_checksum_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A model-custody change between calibration and holdout fails closed."""
    module = _load_module()
    weights = tmp_path / "weights.npz"
    calibration = tmp_path / "cal.npz"
    holdout = tmp_path / "test.npz"
    _write_weights(weights)
    _write_split(calibration)
    _write_split(holdout)
    calls = iter(
        [
            (np.ones((12, 3), dtype=np.float64), "one", False),
            (np.ones((12, 3), dtype=np.float64), "two", False),
        ]
    )
    monkeypatch.setattr(module, "_predict", lambda *_: next(calls))
    with pytest.raises(ValueError, match="checksum changed"):
        module.build_certificate(
            weights_path=weights,
            calibration_path=calibration,
            holdout_path=holdout,
            alpha=0.2,
        )


def test_build_rejects_split_unit_mismatch(tmp_path: Path) -> None:
    """Calibration and holdout metadata cannot silently mix target units."""
    module = _load_module()
    weights = tmp_path / "weights.npz"
    calibration = tmp_path / "cal.npz"
    holdout = tmp_path / "test.npz"
    _write_weights(weights)
    _write_split(calibration, gb_normalized=True)
    _write_split(holdout, gb_normalized=False)
    with pytest.raises(ValueError, match="target-unit metadata"):
        module.build_certificate(
            weights_path=weights,
            calibration_path=calibration,
            holdout_path=holdout,
            alpha=0.2,
        )


def test_cli_write_and_check_modes(tmp_path: Path) -> None:
    """Write, online check, offline check, partial, stale, and absent paths work."""
    module = _load_module()
    weights = tmp_path / "weights.npz"
    calibration = tmp_path / "calibration.npz"
    holdout = tmp_path / "holdout.npz"
    output = tmp_path / "certificate.json"
    _write_weights(weights)
    _write_split(calibration)
    _write_split(holdout)
    common = [
        "--weights",
        str(weights),
        "--calibration",
        str(calibration),
        "--holdout",
        str(holdout),
        "--output",
        str(output),
        "--alpha",
        "0.2",
    ]
    assert module.main(common) == 0
    assert module.main([*common, "--check"]) == 0

    absent_cal = tmp_path / "absent-cal.npz"
    absent_holdout = tmp_path / "absent-holdout.npz"
    offline = [
        "--weights",
        str(weights),
        "--calibration",
        str(absent_cal),
        "--holdout",
        str(absent_holdout),
        "--output",
        str(output),
        "--alpha",
        "0.2",
        "--check",
    ]
    assert module.main(offline) == 0
    absent_cal.write_bytes(b"partial")
    assert module.main(offline) == 1

    absent_cal.unlink()
    output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert module.main([*common, "--check"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["certificates"][0]["channels"][0]["radius"] += 1.0
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert module.main([*common, "--check"]) == 1

    output.unlink()
    assert module.main([*common, "--check"]) == 1


def test_cli_rejects_invalid_json(tmp_path: Path) -> None:
    """Check mode reports malformed committed JSON without a traceback."""
    module = _load_module()
    weights = tmp_path / "weights.npz"
    output = tmp_path / "certificate.json"
    _write_weights(weights)
    output.write_text("{", encoding="utf-8")
    assert (
        module.main(
            [
                "--weights",
                str(weights),
                "--calibration",
                str(tmp_path / "none-cal.npz"),
                "--holdout",
                str(tmp_path / "none-test.npz"),
                "--output",
                str(output),
                "--check",
            ]
        )
        == 1
    )
