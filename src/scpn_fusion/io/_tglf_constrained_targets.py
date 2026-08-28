# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Charge-Constrained TGLF Targets
"""Encode and reconstruct TGLF outputs with exact particle ambipolarity."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.io._tglf_model_selection_data import (
    FLUX_FIELDS,
    SPECIES_SLOTS,
    FloatArray,
    TGLFModelStudyData,
)
from scpn_fusion.io._tglf_model_selection_metrics import closure_ratio

PARTICLE_INDICES: Final = np.asarray(
    [1 + slot * len(FLUX_FIELDS) for slot in range(SPECIES_SLOTS)], dtype=np.int64
)
FIXED_CLOSURE_INDEX: Final = int(PARTICLE_INDICES[0])
PARTICLE_CLOSURE_P95_MAX: Final = 1.0e-12


def constrained_coordinate_names(data: TGLFModelStudyData) -> tuple[str, ...]:
    """Return physical target names with the dependent electron slot relabelled."""
    names = list(data.target_names)
    names[FIXED_CLOSURE_INDEX] = "charge_weighted_particle_residual.fixed_zero"
    return tuple(names)


def encode_constrained_targets(data: TGLFModelStudyData) -> FloatArray:
    """Replace the dependent electron particle coordinate with exact zero."""
    if data.targets.ndim != 2 or data.targets.shape[1] != len(data.target_names):
        raise ValueError("TGLF targets and target names have incompatible shapes")
    if data.charges.shape != (data.targets.shape[0], SPECIES_SLOTS):
        raise ValueError("TGLF charge matrix has an incompatible shape")
    encoded = data.targets.copy()
    encoded[:, FIXED_CLOSURE_INDEX] = 0.0
    return encoded


def reconstruct_constrained_prediction(
    data: TGLFModelStudyData,
    row_indices: NDArray[np.int64],
    coordinate_prediction: FloatArray,
) -> FloatArray:
    """Map candidate coordinates to physical outputs with exact charge closure."""
    if row_indices.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    expected_shape = (row_indices.size, data.targets.shape[1])
    if coordinate_prediction.shape != expected_shape:
        raise ValueError(
            f"coordinate prediction shape {coordinate_prediction.shape} != {expected_shape}"
        )
    if np.any(row_indices < 0) or np.any(row_indices >= data.targets.shape[0]):
        raise ValueError("row_indices contain an out-of-range row")
    charges = data.charges[row_indices]
    active = data.active_targets[row_indices]
    if np.any(charges[:, 0] >= 0.0):
        raise ValueError("species 0 must be the negatively charged electron")
    for slot, particle_index in enumerate(PARTICLE_INDICES[1:], start=1):
        present = active[:, particle_index]
        if np.any(charges[present, slot] <= 0.0) or np.any(charges[~present, slot] != 0.0):
            raise ValueError("ion charges and active particle coordinates are inconsistent")
    prediction = coordinate_prediction.copy()
    prediction[~active] = 0.0
    ion_particle = prediction[:, PARTICLE_INDICES[1:]]
    prediction[:, FIXED_CLOSURE_INDEX] = (
        -np.sum(charges[:, 1:] * ion_particle, axis=1) / charges[:, 0]
    )
    return prediction


def particle_closure_summary(
    data: TGLFModelStudyData,
    row_indices: NDArray[np.int64],
    prediction: FloatArray,
) -> dict[str, float]:
    """Summarise charge-weighted particle closure for reconstructed outputs."""
    expected_shape = (row_indices.size, data.targets.shape[1])
    if prediction.shape != expected_shape:
        raise ValueError(f"prediction shape {prediction.shape} != {expected_shape}")
    values = closure_ratio(prediction[:, PARTICLE_INDICES] * data.charges[row_indices])
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


__all__ = [
    "PARTICLE_CLOSURE_P95_MAX",
    "constrained_coordinate_names",
    "encode_constrained_targets",
    "particle_closure_summary",
    "reconstruct_constrained_prediction",
]
