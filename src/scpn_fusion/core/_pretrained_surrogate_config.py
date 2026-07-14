# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pretrained Surrogates Shared Configuration
"""Shared paths and array type for the pretrained-surrogate modules.

Holds the default dataset / weight / bundle paths and the ``FloatArray`` alias
imported by the MLP surrogate, FNO surrogate, and bundle-orchestrator modules.

Note: ``pretrained_surrogates`` (the facade) re-computes
:data:`DEFAULT_BUNDLE_WEIGHTS_DIR` in its own body so that reloading the facade
after an ``SCPN_ARTIFACT_DIR`` override re-resolves the artifact root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_fusion._data_paths import data_root, default_artifact_path

FloatArray = NDArray[np.float64]

DEFAULT_ITPA_CSV = data_root() / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"
DEFAULT_JET_DIR = data_root() / "validation" / "reference_data" / "jet"
DEFAULT_WEIGHTS_DIR = data_root() / "weights"
DEFAULT_MLP_PATH = DEFAULT_WEIGHTS_DIR / "pretrained_mlp_itpa.npz"
DEFAULT_FNO_PATH = DEFAULT_WEIGHTS_DIR / "pretrained_fno_eurofusion_jet.npz"
DEFAULT_MANIFEST_PATH = DEFAULT_WEIGHTS_DIR / "pretrained_surrogates_manifest.json"
DEFAULT_BUNDLE_WEIGHTS_DIR: Path = default_artifact_path("weights")
DEFAULT_BUNDLE_MLP_PATH = DEFAULT_BUNDLE_WEIGHTS_DIR / "pretrained_mlp_itpa.npz"
DEFAULT_BUNDLE_FNO_PATH = DEFAULT_BUNDLE_WEIGHTS_DIR / "pretrained_fno_eurofusion_jet.npz"
DEFAULT_BUNDLE_MANIFEST_PATH = DEFAULT_BUNDLE_WEIGHTS_DIR / "pretrained_surrogates_manifest.json"
