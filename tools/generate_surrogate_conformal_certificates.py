#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate finite-sample split-conformal certificates for surrogate outputs.

The QLKNN validation split supplies calibration residuals and the untouched
test split supplies an empirical coverage check.  The certificate uses the
finite-sample corrected order statistic
``ceil((n_calibration + 1) * (1 - alpha))`` independently for each output.
The resulting intervals are marginal guarantees under exchangeability; they
are deliberately separate from the input z-score OOD trigger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import numpy.typing as npt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.neural_transport import (  # noqa: E402
    NeuralTransportModel,
    _compute_nustar,
    _mlp_forward,
)


SCHEMA = "scpn-fusion-core.surrogate-conformal-certificates.v1"
DEFAULT_WEIGHTS = ROOT / "weights" / "neural_transport_qlknn.npz"
DEFAULT_CALIBRATION = ROOT / "data" / "qlknn10d_processed" / "val.npz"
DEFAULT_HOLDOUT = ROOT / "data" / "qlknn10d_processed" / "test.npz"
DEFAULT_OUTPUT = ROOT / "validation" / "surrogate_conformal_certificates.json"
OUTPUT_NAMES = ("chi_e", "chi_i", "D_e")
MAX_SPLIT_BYTES = 64 * 1024 * 1024
CRIT_ITG = 4.0
CRIT_TEM = 5.0
DEUTERIUM_MASS_KG = 3.344e-27
ELEMENTARY_CHARGE_C = 1.602e-19
REFERENCE_FIELD_T = 5.3
REFERENCE_MAJOR_RADIUS_M = 6.2

FloatArray = npt.NDArray[np.float64]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: FloatArray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def finite_sample_rank(n_calibration: int, alpha: float) -> int:
    """Return the one-based corrected split-conformal quantile rank."""
    if n_calibration <= 0:
        raise ValueError("n_calibration must be positive.")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1).")
    rank = int(math.ceil((n_calibration + 1) * (1.0 - alpha)))
    if rank > n_calibration:
        raise ValueError("Calibration set is too small for a finite bound at the requested alpha.")
    return rank


def _load_split(path: Path) -> tuple[FloatArray, FloatArray, bool]:
    size = int(path.stat().st_size)
    if size <= 0 or size > MAX_SPLIT_BYTES:
        raise ValueError(f"{path} size must be in 1..{MAX_SPLIT_BYTES} bytes.")
    with np.load(path, allow_pickle=False) as payload:
        if "X" not in payload or "Y" not in payload:
            raise ValueError(f"{path} must contain X and Y arrays.")
        x = np.asarray(payload["X"], dtype=np.float64)
        y = np.asarray(payload["Y"], dtype=np.float64)
        gb_normalized = bool(int(payload["gb_normalized"])) if "gb_normalized" in payload else False
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"{path} X/Y arrays must be aligned two-dimensional matrices.")
    if y.shape[1] != len(OUTPUT_NAMES):
        raise ValueError(f"{path} Y must contain {len(OUTPUT_NAMES)} output columns.")
    if x.shape[0] == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{path} X/Y arrays must be non-empty and finite.")
    return x, y, gb_normalized


def _model_features(features: FloatArray, expected_width: int) -> FloatArray:
    """Reproduce the training/runtime 12D-to-14D derived-feature contract."""
    if features.shape[1] == expected_width:
        return features
    augmented = features
    if features.shape[1] == 10 and expected_width >= 12:
        ti_te = features[:, 2] / np.maximum(features[:, 1], 1e-6)
        nustar = np.asarray(
            [_compute_nustar(row[1], row[3], row[7], row[0]) for row in features],
            dtype=np.float64,
        )
        augmented = np.asarray(np.column_stack([features, ti_te, nustar]), dtype=np.float64)
    if augmented.shape[1] == expected_width:
        return augmented
    if augmented.shape[1] == 12 and expected_width == 14:
        itg_excess = np.maximum(0.0, augmented[:, 5] - CRIT_ITG)
        tem_excess = np.maximum(0.0, augmented[:, 4] - CRIT_TEM)
        return np.asarray(np.column_stack([augmented, itg_excess, tem_excess]), dtype=np.float64)
    raise ValueError(
        f"Processed split width {features.shape[1]} cannot satisfy model width {expected_width}."
    )


def _predict(weights_path: Path, features: FloatArray) -> tuple[FloatArray, str, bool]:
    model = NeuralTransportModel(weights_path)
    if not model.is_neural or model._weights is None or model.weights_checksum is None:
        raise ValueError(f"Could not load neural QLKNN weights from {weights_path}.")
    expected_width = int(model._weights.layers_w[0].shape[0])
    model_features = _model_features(features, expected_width)
    predictions = np.asarray(_mlp_forward(model_features, model._weights), dtype=np.float64)
    if predictions.shape != (features.shape[0], len(OUTPUT_NAMES)):
        raise ValueError("QLKNN prediction matrix has an unexpected shape.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("QLKNN predictions must be finite.")
    return predictions, model.weights_checksum, bool(model._weights.gb_scale)


def _targets_in_prediction_space(
    features: FloatArray,
    targets: FloatArray,
    *,
    gb_normalized: bool,
    model_gb_scale: bool,
) -> tuple[FloatArray, str]:
    """Convert calibration targets into the model's declared output space."""
    if not gb_normalized or not model_gb_scale:
        unit = (
            "gyro-Bohm-normalized transport coefficient"
            if gb_normalized
            else "physical transport coefficient (m^2/s)"
        )
        return targets, unit
    te_j = features[:, 1] * 1e3 * ELEMENTARY_CHARGE_C
    sound_speed = np.sqrt(te_j / DEUTERIUM_MASS_KG)
    rho_s = np.sqrt(DEUTERIUM_MASS_KG * te_j) / (ELEMENTARY_CHARGE_C * REFERENCE_FIELD_T)
    chi_gb = rho_s**2 * sound_speed / REFERENCE_MAJOR_RADIUS_M
    return (
        np.asarray(targets * chi_gb[:, np.newaxis], dtype=np.float64),
        "physical transport coefficient (m^2/s)",
    )


def build_certificate(
    *,
    weights_path: Path,
    calibration_path: Path,
    holdout_path: Path,
    alpha: float,
) -> dict[str, Any]:
    """Build a deterministic per-output split-conformal certificate."""
    x_cal, y_cal_raw, calibration_gb_normalized = _load_split(calibration_path)
    x_test, y_test_raw, holdout_gb_normalized = _load_split(holdout_path)
    if calibration_gb_normalized != holdout_gb_normalized:
        raise ValueError("Calibration and holdout target-unit metadata must match.")
    rank = finite_sample_rank(int(x_cal.shape[0]), alpha)
    pred_cal, runtime_checksum, model_gb_scale = _predict(weights_path, x_cal)
    pred_test, holdout_runtime_checksum, holdout_model_gb_scale = _predict(weights_path, x_test)
    if holdout_runtime_checksum != runtime_checksum or holdout_model_gb_scale != model_gb_scale:
        raise ValueError(
            "Runtime weight checksum changed between calibration and holdout inference."
        )
    y_cal, output_unit = _targets_in_prediction_space(
        x_cal,
        y_cal_raw,
        gb_normalized=calibration_gb_normalized,
        model_gb_scale=model_gb_scale,
    )
    y_test, holdout_unit = _targets_in_prediction_space(
        x_test,
        y_test_raw,
        gb_normalized=holdout_gb_normalized,
        model_gb_scale=holdout_model_gb_scale,
    )
    cal_scores = np.abs(pred_cal - y_cal)
    test_scores = np.abs(pred_test - y_test)
    channels: list[dict[str, Any]] = []
    for index, name in enumerate(OUTPUT_NAMES):
        scores = np.asarray(cal_scores[:, index], dtype=np.float64)
        radius = float(np.partition(scores, rank - 1)[rank - 1])
        coverage = float(np.mean(test_scores[:, index] <= radius))
        channels.append(
            {
                "name": name,
                "unit": output_unit,
                "nonconformity": "absolute_error",
                "interval": "[max(0, prediction - radius), prediction + radius]",
                "radius": radius,
                "calibration_score_sha256": _sha256_array(scores),
                "holdout_empirical_coverage": coverage,
            }
        )

    return {
        "schema": SCHEMA,
        "method": {
            "name": "split_conformal_absolute_residual",
            "alpha": float(alpha),
            "target_marginal_coverage": float(1.0 - alpha),
            "quantile_rank_formula": "ceil((n_calibration + 1) * (1 - alpha))",
            "reference": {
                "title": "Distribution-Free Predictive Inference for Regression",
                "doi": "10.1080/01621459.2017.1307116",
            },
            "assumptions": [
                "Calibration and future examples are exchangeable.",
                "Coverage is marginal per output channel, not conditional on an input or OOD state.",
                "The z-score OOD trigger remains a separate escalation mechanism and is not an error bound.",
            ],
        },
        "certificates": [
            {
                "id": "qlknn10d_neural_transport",
                "status": "certified_split_conformal",
                "model": {
                    "weights": _display_path(weights_path),
                    "weights_file_sha256": _sha256_file(weights_path),
                    "runtime_weights_checksum": runtime_checksum,
                },
                "calibration": {
                    "split": _display_path(calibration_path),
                    "split_sha256": _sha256_file(calibration_path),
                    "source_targets_gb_normalized": calibration_gb_normalized,
                    "certificate_output_unit": output_unit,
                    "n": int(x_cal.shape[0]),
                    "quantile_rank_one_based": rank,
                },
                "holdout": {
                    "split": _display_path(holdout_path),
                    "split_sha256": _sha256_file(holdout_path),
                    "source_targets_gb_normalized": holdout_gb_normalized,
                    "certificate_output_unit": holdout_unit,
                    "n": int(x_test.shape[0]),
                    "role": "independent_empirical_coverage_check_only",
                },
                "channels": channels,
            }
        ],
    }


def validate_certificate(payload: Any, *, weights_path: Path | None = None) -> None:
    """Validate certificate structure, finite-sample rank, and model custody."""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("Unexpected conformal-certificate schema.")
    method = payload.get("method")
    if not isinstance(method, dict):
        raise ValueError("Certificate method must be an object.")
    alpha = method.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("Certificate alpha must be numeric.")
    certificates = payload.get("certificates")
    if not isinstance(certificates, list) or len(certificates) != 1:
        raise ValueError("Exactly one QLKNN certificate is required.")
    certificate = certificates[0]
    if not isinstance(certificate, dict):
        raise ValueError("Certificate entry must be an object.")
    if certificate.get("id") != "qlknn10d_neural_transport":
        raise ValueError("Certificate id must bind the QLKNN transport lane.")
    calibration = certificate.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Calibration metadata must be an object.")
    n_calibration = calibration.get("n")
    rank = calibration.get("quantile_rank_one_based")
    if isinstance(n_calibration, bool) or not isinstance(n_calibration, int):
        raise ValueError("Calibration n must be an integer.")
    expected_rank = finite_sample_rank(n_calibration, float(alpha))
    if rank != expected_rank:
        raise ValueError("Certificate quantile rank does not match the finite-sample correction.")
    channels = certificate.get("channels")
    if not isinstance(channels, list) or [
        item.get("name") for item in channels if isinstance(item, dict)
    ] != list(OUTPUT_NAMES):
        raise ValueError("Certificate channels must be chi_e, chi_i, and D_e in order.")
    for channel in channels:
        channel_data = cast(dict[str, Any], channel)
        radius = channel_data.get("radius")
        coverage = channel_data.get("holdout_empirical_coverage")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(float(radius))
            or float(radius) < 0.0
        ):
            raise ValueError("Every certificate radius must be finite and non-negative.")
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not 0.0 <= float(coverage) <= 1.0
        ):
            raise ValueError("Every empirical holdout coverage must lie in [0, 1].")
    if weights_path is not None:
        model = certificate.get("model")
        if not isinstance(model, dict) or model.get("weights_file_sha256") != _sha256_file(
            weights_path
        ):
            raise ValueError("Certificate weight digest does not match the current model artifact.")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Write or drift-check the committed split-conformal certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.check:
            if not args.output.is_file():
                raise ValueError(f"Certificate output is missing: {args.output}")
            committed = _load_json(args.output)
            validate_certificate(committed, weights_path=args.weights)
            calibration_exists = args.calibration.is_file()
            holdout_exists = args.holdout.is_file()
            if calibration_exists != holdout_exists:
                raise ValueError(
                    "Calibration and holdout inputs must either both exist or both be absent."
                )
            if calibration_exists:
                fresh = build_certificate(
                    weights_path=args.weights,
                    calibration_path=args.calibration,
                    holdout_path=args.holdout,
                    alpha=args.alpha,
                )
                if fresh != committed:
                    raise ValueError("Split-conformal certificate is stale.")
            print(f"Split-conformal certificate is valid: {args.output}")
            return 0

        certificate = build_certificate(
            weights_path=args.weights,
            calibration_path=args.calibration,
            holdout_path=args.holdout,
            alpha=args.alpha,
        )
        validate_certificate(certificate, weights_path=args.weights)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote split-conformal certificate: {args.output}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"CONFORMAL CERTIFICATE ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
