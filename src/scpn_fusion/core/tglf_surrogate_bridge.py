# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TGLF Surrogate Bridge
"""TGLF dataset generation and a deterministic transport-surrogate fit.

``TGLFDatasetGenerator`` samples the TGLF binary to build a training set;
``TGLFSurrogate`` is a deterministic ridge-regularised polynomial regression that
maps the turbulence-drive inputs (``R/LTi``, ``R/LTe``, ``R/Lne``, ``q``,
``s_hat``, ``beta_e``, ``Z_eff``) to the TGLF transport outputs (``chi_i``,
``chi_e``, ``gamma_max``, ``q_i``, ``q_e``). The fit is a closed-form
``(ΦᵀΦ + ridge·I)⁻¹ Φᵀ Y`` solve over standardised, per-feature quadratic features,
so it is reproducible bit-for-bit from a given dataset (no RNG, no iterative
optimiser). ``train_surrogate_from_tglf`` fits, persists ``.npz`` weights, and
returns the per-target training RMSE so a caller can assess surrogate fidelity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)

#: Canonical turbulence-drive input features (the parameters the sampler varies).
DEFAULT_TGLF_FEATURES: tuple[str, ...] = (
    "R_LTi",
    "R_LTe",
    "R_Lne",
    "q",
    "s_hat",
    "beta_e",
    "Z_eff",
)

#: TGLF transport outputs the surrogate predicts.
DEFAULT_TGLF_TARGETS: tuple[str, ...] = (
    "chi_i",
    "chi_e",
    "gamma_max",
    "q_i",
    "q_e",
)


class TGLFDatasetGenerator:
    """Automated generation of TGLF datasets for surrogate training."""

    def __init__(self, tglf_binary_path: str | Path) -> None:
        """Store the TGLF executable path used by sampled dataset runs."""
        self.tglf_path = Path(tglf_binary_path)

    def generate_random_dataset(self, n_samples: int = 100) -> list[dict[str, Any]]:
        """Generate a randomized dataset of TGLF runs."""
        from scpn_fusion.core import tglf_interface as tglf

        rng = np.random.default_rng()
        dataset: list[dict[str, Any]] = []

        print(f"[TGLF] Generating {n_samples} samples for surrogate training...")
        for i in range(n_samples):
            deck = tglf.TGLFInputDeck(
                R_LTi=float(rng.uniform(0.0, 12.0)),
                R_LTe=float(rng.uniform(0.0, 12.0)),
                R_Lne=float(rng.uniform(0.0, 5.0)),
                q=float(rng.uniform(1.0, 5.0)),
                s_hat=float(rng.uniform(0.0, 3.0)),
                beta_e=float(rng.uniform(0.001, 0.05)),
                Z_eff=float(rng.uniform(1.0, 3.0)),
            )

            try:
                out = tglf.run_tglf_binary(deck, self.tglf_path, timeout_s=60.0)
                dataset.append({"input": deck.__dict__, "output": out.__dict__})
            except Exception as exc:
                logger.warning("Sample %s failed: %s", i, exc)

        return dataset


def _dataset_to_arrays(
    dataset: list[dict[str, Any]],
    features: tuple[str, ...],
    targets: tuple[str, ...],
) -> tuple[FloatArray, FloatArray]:
    """Return ``(X, Y)`` design/target matrices extracted from a TGLF dataset."""
    if len(dataset) < len(features) + 1:
        raise ValueError(
            f"dataset needs at least {len(features) + 1} samples to fit "
            f"{len(features)} features; got {len(dataset)}."
        )
    rows_x: list[list[float]] = []
    rows_y: list[list[float]] = []
    for sample in dataset:
        din = sample["input"]
        dout = sample["output"]
        try:
            rows_x.append([float(din[name]) for name in features])
            rows_y.append([float(dout[name]) for name in targets])
        except KeyError as exc:
            raise ValueError(f"dataset sample missing required key: {exc}") from exc
    x = np.asarray(rows_x, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.float64)
    if not (bool(np.all(np.isfinite(x))) and bool(np.all(np.isfinite(y)))):
        raise ValueError("dataset contains non-finite inputs or outputs.")
    return x, y


class TGLFSurrogate:
    """Deterministic ridge-regularised polynomial TGLF transport surrogate."""

    def __init__(
        self,
        *,
        features: tuple[str, ...] = DEFAULT_TGLF_FEATURES,
        targets: tuple[str, ...] = DEFAULT_TGLF_TARGETS,
        ridge: float = 1e-3,
    ) -> None:
        """Configure the feature/target vocabulary and ridge strength."""
        self.features = tuple(features)
        self.targets = tuple(targets)
        self.ridge = float(max(ridge, 1e-12))
        self._mean: FloatArray | None = None
        self._std: FloatArray | None = None
        self._weights: FloatArray | None = None

    def _standardise(self, x: FloatArray) -> FloatArray:
        """Return the z-scored inputs using the fitted per-feature statistics."""
        assert self._mean is not None and self._std is not None
        return (x - self._mean) / self._std

    def _design(self, x: FloatArray) -> FloatArray:
        """Return the standardised linear + per-feature quadratic design matrix."""
        z = self._standardise(x)
        ones = np.ones((z.shape[0], 1), dtype=np.float64)
        return np.concatenate([ones, z, z * z], axis=1)

    def fit(self, x: FloatArray, y: FloatArray) -> TGLFSurrogate:
        """Fit the surrogate weights by a closed-form ridge solve."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError(f"x must be (N, {len(self.features)}).")
        if y.shape[0] != x.shape[0] or y.shape[1] != len(self.targets):
            raise ValueError(f"y must be (N, {len(self.targets)}) aligned with x.")
        self._mean = x.mean(axis=0)
        std = x.std(axis=0)
        # Guard constant features so standardisation never divides by zero.
        self._std = np.where(std > 1e-12, std, 1.0)
        phi = self._design(x)
        gram = phi.T @ phi + self.ridge * np.eye(phi.shape[1], dtype=np.float64)
        self._weights = np.asarray(np.linalg.solve(gram, phi.T @ y), dtype=np.float64)
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict transport outputs for input rows ``x``."""
        if self._weights is None:
            raise RuntimeError("Surrogate is not fit. Call fit() first.")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != len(self.features):
            raise ValueError(f"x must have {len(self.features)} columns.")
        return np.asarray(self._design(x) @ self._weights, dtype=np.float64)

    def training_rmse(self, x: FloatArray, y: FloatArray) -> dict[str, float]:
        """Return the per-target RMSE of the surrogate on ``(x, y)``."""
        pred = self.predict(x)
        y = np.asarray(y, dtype=np.float64)
        rmse = np.sqrt(np.mean((pred - y) ** 2, axis=0))
        return {name: float(rmse[i]) for i, name in enumerate(self.targets)}

    def save(self, path: str | Path) -> None:
        """Persist the fitted surrogate to a ``.npz`` archive."""
        if self._weights is None or self._mean is None or self._std is None:
            raise RuntimeError("Surrogate is not fit. Call fit() first.")
        np.savez(
            path,
            features=np.array(self.features),
            targets=np.array(self.targets),
            ridge=np.array([self.ridge], dtype=np.float64),
            mean=self._mean,
            std=self._std,
            weights=self._weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> TGLFSurrogate:
        """Load a fitted surrogate from a ``.npz`` archive."""
        with np.load(path, allow_pickle=False) as archive:
            model = cls(
                features=tuple(str(name) for name in archive["features"]),
                targets=tuple(str(name) for name in archive["targets"]),
                ridge=float(archive["ridge"][0]),
            )
            model._mean = np.asarray(archive["mean"], dtype=np.float64)
            model._std = np.asarray(archive["std"], dtype=np.float64)
            model._weights = np.asarray(archive["weights"], dtype=np.float64)
        return model


def train_surrogate_from_tglf(
    dataset: list[dict[str, Any]],
    output_path: str | Path,
    *,
    features: tuple[str, ...] = DEFAULT_TGLF_FEATURES,
    targets: tuple[str, ...] = DEFAULT_TGLF_TARGETS,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    """Fit a deterministic TGLF transport surrogate and persist its weights.

    Extracts the design/target matrices from ``dataset`` (a list of
    ``{"input": deck, "output": run}`` dicts as produced by
    :meth:`TGLFDatasetGenerator.generate_random_dataset`), fits a
    :class:`TGLFSurrogate`, writes the ``.npz`` weights to ``output_path``, and
    returns a fit report with the per-target training RMSE.

    Parameters
    ----------
    dataset : list of dict
        TGLF samples, each with an ``"input"`` and ``"output"`` field mapping.
    output_path : str or Path
        Destination ``.npz`` path for the fitted weights.
    features, targets : tuple of str
        Input feature and output target keys to use.
    ridge : float
        Ridge regularisation strength.

    Returns
    -------
    dict
        ``{"n_samples", "features", "targets", "rmse", "output_path"}``.
    """
    model = TGLFSurrogate(features=features, targets=targets, ridge=ridge)
    x, y = _dataset_to_arrays(dataset, model.features, model.targets)
    model.fit(x, y)
    model.save(output_path)
    report = {
        "n_samples": int(x.shape[0]),
        "features": list(model.features),
        "targets": list(model.targets),
        "rmse": model.training_rmse(x, y),
        "output_path": str(output_path),
    }
    logger.info(
        "Fitted TGLF surrogate on %d samples; per-target RMSE %s", x.shape[0], report["rmse"]
    )
    return report


__all__ = [
    "DEFAULT_TGLF_FEATURES",
    "DEFAULT_TGLF_TARGETS",
    "TGLFDatasetGenerator",
    "TGLFSurrogate",
    "train_surrogate_from_tglf",
]
