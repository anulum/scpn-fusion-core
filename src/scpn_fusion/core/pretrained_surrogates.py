# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pretrained Surrogates Bundle
"""Bundle and coverage facade for the pretrained MLP/FNO surrogates.

Composes the ITPA MLP surrogate (:mod:`~scpn_fusion.core.pretrained_mlp_surrogate`)
and the Eurofusion/JET FNO surrogate
(:mod:`~scpn_fusion.core.pretrained_fno_surrogate`) into a single reproducible
artifact bundle with a provenance manifest, and reports pretrained-surrogate
coverage. Re-exports the full surrogate surface for backward compatibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

from scpn_fusion._data_paths import artifact_root, data_root, default_artifact_path

from ._pretrained_surrogate_config import (
    DEFAULT_BUNDLE_FNO_PATH,
    DEFAULT_BUNDLE_MANIFEST_PATH,
    DEFAULT_BUNDLE_MLP_PATH,
    DEFAULT_ITPA_CSV,
    DEFAULT_JET_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MLP_PATH,
    DEFAULT_FNO_PATH,
    DEFAULT_WEIGHTS_DIR,
    FloatArray,
)
from .pretrained_fno_surrogate import (
    _train_fno_on_jet,
    evaluate_pretrained_fno,
)
from .pretrained_mlp_surrogate import (
    PretrainedMLPSurrogate,
    _train_itpa_mlp,
    evaluate_pretrained_mlp,
    load_pretrained_mlp,
    save_pretrained_mlp,
)

logger = logging.getLogger(__name__)

# Re-computed in this module body (not imported) so that reloading this facade
# after an ``SCPN_ARTIFACT_DIR`` override re-resolves the artifact root
# (see tests/test_data_paths.py::test_training_output_defaults_resolve_under_artifact_root).
DEFAULT_BUNDLE_WEIGHTS_DIR: Path = default_artifact_path("weights")

_REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "version",
    "artifacts",
    "datasets",
    "config",
    "metrics",
    "coverage",
)

__all__ = [
    "DEFAULT_BUNDLE_FNO_PATH",
    "DEFAULT_BUNDLE_MANIFEST_PATH",
    "DEFAULT_BUNDLE_MLP_PATH",
    "DEFAULT_BUNDLE_WEIGHTS_DIR",
    "DEFAULT_FNO_PATH",
    "DEFAULT_ITPA_CSV",
    "DEFAULT_JET_DIR",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_MLP_PATH",
    "DEFAULT_WEIGHTS_DIR",
    "FloatArray",
    "PretrainedMLPSurrogate",
    "bundle_pretrained_surrogates",
    "evaluate_pretrained_fno",
    "evaluate_pretrained_mlp",
    "get_pretrained_surrogate_coverage",
    "load_pretrained_mlp",
    "save_pretrained_mlp",
]


def _default_surrogate_coverage() -> dict[str, Any]:
    shipped = [
        "scpn_fusion.core.pretrained_surrogates:mlp_itpa",
        "scpn_fusion.core.pretrained_surrogates:fno_eurofusion_jet",
        "scpn_fusion.core.neural_equilibrium:sparc",
        "scpn_fusion.core.neural_transport:qlknn",
    ]
    requires_user_training = [
        "scpn_fusion.core.heat_ml_shadow_surrogate",
        "scpn_fusion.core.gyro_swin_surrogate",
        "scpn_fusion.core.turbulence_oracle",
    ]
    total = len(shipped) + len(requires_user_training)
    return {
        "pretrained_shipped": shipped,
        "requires_user_training": requires_user_training,
        "coverage_fraction": float(len(shipped) / max(total, 1)),
        "coverage_percent": float(100.0 * len(shipped) / max(total, 1)),
        "notes": (
            "Pretrained artifacts are bundled for MLP (ITPA), FNO (JET), "
            "neural equilibrium (SPARC GEQDSK), and QLKNN transport. "
            "Remaining surrogate lanes still require facility-specific "
            "user training."
        ),
    }


def get_pretrained_surrogate_coverage(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the coverage metadata for pretrained surrogate availability.

    Parameters
    ----------
    manifest
        Optional user-provided manifest payload. If present and valid, its
        ``coverage`` section is merged into the canonical shipped baseline.

    Returns
    -------
    dict[str, Any]
        Coverage report containing shipped models, user-training-only models,
        coverage fraction, and explanatory notes.
    """
    default_cov = _default_surrogate_coverage()
    if not manifest:
        return default_cov
    cov = manifest.get("coverage")
    if not isinstance(cov, dict):
        return default_cov
    merged = dict(default_cov)
    merged.update(cov)
    return merged


def _as_known_relative(path: Path) -> str:
    p = Path(path)
    resolved = p.resolve()
    for root in (data_root(), artifact_root()):
        try:
            return str(resolved.relative_to(root.resolve()).as_posix())
        except ValueError:
            continue
    return str(p.as_posix())


def _load_cached_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        raw = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cached manifest: {exc.__class__.__name__}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("invalid cached manifest: expected JSON object.")
    missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        missing_joined = ", ".join(missing)
        raise ValueError(f"invalid cached manifest: missing keys: {missing_joined}")
    if not isinstance(manifest.get("version"), str):
        raise ValueError("invalid cached manifest: version must be a string.")
    for k in ("artifacts", "datasets", "config", "metrics", "coverage"):
        if not isinstance(manifest.get(k), dict):
            raise ValueError(f"invalid cached manifest: {k} must be an object.")
    return manifest


def bundle_pretrained_surrogates(
    *,
    force_retrain: bool = False,
    seed: int = 42,
    itpa_csv_path: Path = DEFAULT_ITPA_CSV,
    jet_dir: Path = DEFAULT_JET_DIR,
    mlp_hidden: int = 32,
    mlp_epochs: int = 1200,
    mlp_lr: float = 1.2e-2,
    mlp_l2: float = 5.0e-4,
    fno_modes: int = 8,
    fno_width: int = 16,
    fno_epochs: int = 24,
    fno_batch_size: int = 8,
    fno_augment_per_file: int = 12,
    weights_dir: Path = DEFAULT_BUNDLE_WEIGHTS_DIR,
    manifest_path: Path = DEFAULT_BUNDLE_MANIFEST_PATH,
    mlp_path: Path = DEFAULT_BUNDLE_MLP_PATH,
    fno_path: Path = DEFAULT_BUNDLE_FNO_PATH,
) -> dict[str, Any]:
    """Train (optionally), serialize, and bundle pretrained surrogates.

    The helper emits an artifact manifest containing file paths, dataset
    provenance and run-time metrics for traceability and reproducibility.

    Parameters
    ----------
    force_retrain
        When ``True``, rebuild both artifacts regardless of existing cache.
    seed
        Global random seed for both training pipelines.
    itpa_csv_path
        Source ITPA CSV file for MLP training/evaluation.
    jet_dir
        Directory with GEQDSK files for FNO training/evaluation.
    mlp_hidden
        Hidden width of the single hidden MLP layer.
    mlp_epochs
        Number of optimization epochs for MLP training.
    mlp_lr
        MLP learning rate.
    mlp_l2
        L2 regularization coefficient applied in MLP updates.
    fno_modes
        Number of Fourier modes used by the FNO model.
    fno_width
        FNO channel width.
    fno_epochs
        Number of FNO training epochs.
    fno_batch_size
        Mini-batch size used for FNO optimization.
    fno_augment_per_file
        Number of augmented samples drawn from each GEQDSK input.
    weights_dir
        Output directory for generated artifact files.
    manifest_path
        Path where metadata manifest is persisted.
    mlp_path
        Target path for serialized MLP artifact.
    fno_path
        Target path for serialized FNO artifact.

    Returns
    -------
    dict[str, Any]
        Manifest payload with generated artefacts, datasets, run-time config,
        metric values, and coverage map.
    """
    if int(mlp_hidden) <= 0:
        raise ValueError("mlp_hidden must be > 0.")
    if int(mlp_epochs) <= 0:
        raise ValueError("mlp_epochs must be > 0.")
    if int(fno_modes) <= 0 or int(fno_width) <= 0:
        raise ValueError("fno_modes/fno_width must be > 0.")
    if int(fno_epochs) <= 0 or int(fno_batch_size) <= 0:
        raise ValueError("fno_epochs/fno_batch_size must be > 0.")
    if int(fno_augment_per_file) <= 0:
        raise ValueError("fno_augment_per_file must be > 0.")

    if not force_retrain and manifest_path.exists() and mlp_path.exists() and fno_path.exists():
        try:
            return _load_cached_manifest(manifest_path)
        except ValueError as exc:
            logger.warning(
                "Invalid cached pretrained-surrogates manifest at %s; rebuilding artifacts: %s",
                manifest_path,
                exc,
            )

    mlp_model, mlp_metrics = _train_itpa_mlp(
        seed=seed,
        hidden=mlp_hidden,
        epochs=mlp_epochs,
        lr=mlp_lr,
        l2=mlp_l2,
        csv_path=itpa_csv_path,
    )
    save_pretrained_mlp(mlp_model, path=mlp_path)

    fno_metrics = _train_fno_on_jet(
        save_path=fno_path,
        seed=seed + 73,
        modes=fno_modes,
        width=fno_width,
        epochs=fno_epochs,
        batch_size=fno_batch_size,
        augment_per_file=fno_augment_per_file,
        jet_dir=jet_dir,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "task2-pretrained-v1",
        "artifacts": {
            "mlp_itpa": _as_known_relative(mlp_path),
            "fno_eurofusion_jet": _as_known_relative(fno_path),
        },
        "datasets": {
            "itpa": _as_known_relative(itpa_csv_path),
            "eurofusion_proxy_jet": _as_known_relative(jet_dir),
        },
        "config": {
            "seed": int(seed),
            "mlp_hidden": int(mlp_hidden),
            "mlp_epochs": int(mlp_epochs),
            "fno_modes": int(fno_modes),
            "fno_width": int(fno_width),
            "fno_epochs": int(fno_epochs),
            "fno_batch_size": int(fno_batch_size),
            "fno_augment_per_file": int(fno_augment_per_file),
        },
        "metrics": {
            "mlp": mlp_metrics,
            "fno": fno_metrics,
        },
        "coverage": _default_surrogate_coverage(),
    }
    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
