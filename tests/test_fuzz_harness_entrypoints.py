# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fuzz Harness Entrypoint Tests
"""Tests for executable fuzz harness entrypoints over malformed input seeds."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_harness(name: str) -> ModuleType:
    path = ROOT / "fuzz" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geqdsk_fuzz_harness_rejects_absurd_grid_seed() -> None:
    harness = _load_harness("fuzz_geqdsk")
    harness.TestOneInput(b"absurd 0 1000000 1000000\n")


def test_config_fuzz_harness_rejects_deeply_invalid_json_seed() -> None:
    harness = _load_harness("fuzz_fusion_config")
    harness.TestOneInput(b'{"grid_resolution": [1, 1], "dimensions": {}}')


def test_npz_fuzz_harness_rejects_truncated_zip_seed() -> None:
    harness = _load_harness("fuzz_disruption_npz")
    harness.TestOneInput(b"PK\x03\x04truncated")


def test_imas_fuzz_harness_rejects_deeply_invalid_json_seed() -> None:
    harness = _load_harness("fuzz_imas_ids")
    harness.TestOneInput(b'{"ids_properties": {"homogeneous_time": 1}, "time_slice": "bad"}')


def test_snn_artifact_fuzz_harness_rejects_invalid_artifact_seed() -> None:
    harness = _load_harness("fuzz_snn_artifact")
    harness.TestOneInput(b'{"meta": {"artifact_version": 1}, "topology": {"places": []}}')


def test_snn_artifact_fuzz_harness_rejects_non_utf8_seed() -> None:
    harness = _load_harness("fuzz_snn_artifact")
    harness.TestOneInput(b"\xff\xfe\x00\x01not-a-controller")


def test_checked_npz_loader_rejects_oversized_disruption_archive(tmp_path: Path) -> None:
    from scpn_fusion.io.tokamak_disruption_archive import load_disruption_shot
    from scpn_fusion.io.safe_loaders import MAX_NPZ_BYTES

    path = tmp_path / "shot.npz"
    with path.open("wb") as handle:
        handle.truncate(MAX_NPZ_BYTES + 1)

    try:
        load_disruption_shot(path, disruption_dir=tmp_path)
    except ValueError as exc:
        assert "NumPy archive file too large" in str(exc)
    else:
        raise AssertionError("oversized NPZ archive was accepted")


def test_checked_npz_loader_rejects_compressed_expansion_bomb(tmp_path: Path) -> None:
    import numpy as np

    from scpn_fusion.io.safe_loaders import MAX_NPZ_MEMBER_BYTES
    from scpn_fusion.io.tokamak_disruption_archive import load_disruption_shot

    path = tmp_path / "shot.npz"
    oversized_points = MAX_NPZ_MEMBER_BYTES // np.dtype(np.float64).itemsize + 1
    np.savez_compressed(
        path,
        time_s=np.zeros(oversized_points, dtype=np.float64),
        Ip_MA=np.zeros(1),
        BT_T=np.zeros(1),
        beta_N=np.zeros(1),
        q95=np.zeros(1),
        ne_1e19=np.zeros(1),
        n1_amp=np.zeros(1),
        n2_amp=np.zeros(1),
        locked_mode_amp=np.zeros(1),
        dBdt_gauss_per_s=np.zeros(1),
        vertical_position_m=np.zeros(1),
        is_disruption=np.array(False),
        disruption_time_idx=np.array(-1),
        disruption_type=np.array("none"),
    )
    assert path.stat().st_size < MAX_NPZ_MEMBER_BYTES

    with pytest.raises(ValueError, match="member too large"):
        load_disruption_shot(path, disruption_dir=tmp_path)
