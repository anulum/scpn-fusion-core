# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pretrained Surrogate Config Tests
"""Contract tests for the shared pretrained-surrogate path configuration."""

from __future__ import annotations

from scpn_fusion._data_paths import data_root, default_artifact_path
from scpn_fusion.core import _pretrained_surrogate_config as cfg


def test_data_root_anchored_paths_resolve_under_weights() -> None:
    weights = data_root() / "weights"
    assert weights == cfg.DEFAULT_WEIGHTS_DIR
    assert weights / "pretrained_mlp_itpa.npz" == cfg.DEFAULT_MLP_PATH
    assert weights / "pretrained_fno_eurofusion_jet.npz" == cfg.DEFAULT_FNO_PATH
    assert weights / "pretrained_surrogates_manifest.json" == cfg.DEFAULT_MANIFEST_PATH
    assert cfg.DEFAULT_ITPA_CSV.name == "hmode_confinement.csv"
    assert cfg.DEFAULT_JET_DIR.name == "jet"


def test_bundle_paths_anchored_under_artifact_root() -> None:
    bundle_weights = default_artifact_path("weights")
    assert bundle_weights == cfg.DEFAULT_BUNDLE_WEIGHTS_DIR
    assert bundle_weights / "pretrained_mlp_itpa.npz" == cfg.DEFAULT_BUNDLE_MLP_PATH
    assert bundle_weights / "pretrained_fno_eurofusion_jet.npz" == cfg.DEFAULT_BUNDLE_FNO_PATH
    assert (
        bundle_weights / "pretrained_surrogates_manifest.json" == cfg.DEFAULT_BUNDLE_MANIFEST_PATH
    )
