# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Neural Transport Surrogate
"""Compatibility façade for the neural transport surrogate.

The implementation is split across focused helper modules while this module
retains the historical import surface used by callers and tests.
"""

from __future__ import annotations

from ._neural_transport_analytic import (
    _CHI_GB,
    _CRIT_ETG,
    _CRIT_ITG,
    _CRIT_TEM,
    _STIFFNESS,
    _STIFFNESS_MAX,
    _STIFFNESS_MIN,
    _TRANSPORT_FLOOR,
    _dominant_channel,
    _gyro_bohm_diffusivity,
    critical_gradient_model,
    reduced_gyrokinetic_profile_model,
)
from ._neural_transport_runtime import NeuralTransportModel, _append_derived
from ._neural_transport_types import (
    FloatArray,
    MLPWeights,
    TransportFluxes,
    TransportInputs,
    _MAX_WEIGHTS_FILE_BYTES,
    _WEIGHTS_FORMAT_VERSION,
)
from .neural_transport_math import _compute_nustar, _mlp_forward
from .neural_transport_math import _relu as _relu
from .neural_transport_math import _softplus as _softplus


# Backward-compatible class name used by older interop/parity tests.
NeuralTransportSurrogate = NeuralTransportModel


__all__ = [
    "FloatArray",
    "MLPWeights",
    "NeuralTransportModel",
    "NeuralTransportSurrogate",
    "TransportFluxes",
    "TransportInputs",
    "_CHI_GB",
    "_CRIT_ETG",
    "_CRIT_ITG",
    "_CRIT_TEM",
    "_MAX_WEIGHTS_FILE_BYTES",
    "_STIFFNESS",
    "_STIFFNESS_MAX",
    "_STIFFNESS_MIN",
    "_TRANSPORT_FLOOR",
    "_WEIGHTS_FORMAT_VERSION",
    "_append_derived",
    "_compute_nustar",
    "_dominant_channel",
    "_gyro_bohm_diffusivity",
    "_mlp_forward",
    "_relu",
    "_softplus",
    "critical_gradient_model",
    "reduced_gyrokinetic_profile_model",
]
