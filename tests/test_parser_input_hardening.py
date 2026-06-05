# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Parser Input Hardening Tests
"""Malformed GEQDSK and IMAS inputs fail closed before solver use."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_fusion.core.eqdsk import read_geqdsk
from scpn_fusion.io.imas_connector_storage import read_ids, write_ids


def _minimal_geqdsk_tokens(*, scalar_override: str | None = None) -> list[str]:
    scalars = [
        "1.0",
        "1.0",
        "1.5",
        "1.0",
        "0.0",
        "1.4",
        "0.0",
        "0.1",
        "0.9",
        "5.0",
        "1000000.0",
        "0.1",
        "0.0",
        "1.4",
        "0.0",
        "0.0",
        "0.0",
        "0.9",
        "0.0",
        "0.0",
    ]
    if scalar_override is not None:
        scalars[0] = scalar_override
    profiles = ["1.0", "1.0"] * 4
    psirz = ["0.1", "0.2", "0.3", "0.9"]
    qpsi = ["1.0", "2.0"]
    counts = ["0", "0"]
    return scalars + profiles + psirz + qpsi + counts


def _write_geqdsk(path: Path, tokens: list[str]) -> None:
    path.write_text("case 0 2 2\n" + "\n".join(tokens) + "\n", encoding="utf-8")


def test_geqdsk_rejects_non_finite_scalar(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.geqdsk"
    _write_geqdsk(path, _minimal_geqdsk_tokens(scalar_override="1e309"))

    with pytest.raises(ValueError, match="must be finite"):
        read_geqdsk(path)


def test_geqdsk_rejects_truncated_contours(tmp_path: Path) -> None:
    path = tmp_path / "truncated-contour.geqdsk"
    tokens = _minimal_geqdsk_tokens()
    tokens[-2:] = ["1", "0"]
    _write_geqdsk(path, tokens)

    with pytest.raises(ValueError, match="contour values"):
        read_geqdsk(path)


def test_imas_read_rejects_malformed_equilibrium_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad-imas.json"
    path.write_text(
        json.dumps({"ids_properties": {"homogeneous_time": 1}, "time": [0.0], "time_slice": "bad"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="time_slice"):
        read_ids(path)


def test_imas_write_rejects_excessive_json_depth(tmp_path: Path) -> None:
    payload: dict[str, object] = {"ids_properties": {"homogeneous_time": 1}}
    cursor = payload
    for idx in range(70):
        nested: dict[str, object] = {}
        cursor[f"level_{idx}"] = nested
        cursor = nested

    with pytest.raises(ValueError, match="nesting exceeds safety limit"):
        write_ids(payload, tmp_path / "too-deep.json")
