# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — tests for the IMAS equilibrium IDS bridge
"""Self-contained tests for :mod:`scpn_fusion.core.imas_equilibrium_io`.

Covers both directions of the bridge (solver ``(NZ, NR)`` ⇄ IMAS ``[dim1, dim2] = [R, Z]``) with an
orientation test that cannot self-cancel, the fail-closed guards (shape mismatch, non-rectangular
grid), IMAS data-dictionary schema enforcement, honest absence of optional quantities, and the JSON
file round trip. Hermetic: skips when the ``omas`` optional dependency (full extra) is not
installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omas", reason="omas (full extra) not installed")

from scpn_fusion.core.imas_equilibrium_io import (
    EquilibriumSlice,
    equilibrium_to_ods,
    load_equilibrium_ids,
    ods_to_equilibrium,
    save_equilibrium_ids,
)

_R = np.linspace(1.0, 2.5, 7)
_Z = np.linspace(-1.4, 1.4, 5)
_NZ, _NR = _Z.size, _R.size


def _slice(**overrides: object) -> EquilibriumSlice:
    rng = np.random.default_rng(3)
    base: dict = {
        "psi": rng.standard_normal((_NZ, _NR)),
        "R_grid": _R,
        "Z_grid": _Z,
        "ip": 1.5e6,
        "r_axis": 1.68,
        "z_axis": 0.02,
        "psi_axis": 0.91,
        "psi_boundary": 0.14,
        "time": 2.5,
    }
    base.update(overrides)
    return EquilibriumSlice(**base)


# ── Round trip (solver → IDS → solver) ────────────────────────────────────


def test_round_trip_is_exact() -> None:
    eq = _slice()
    back = ods_to_equilibrium(equilibrium_to_ods(eq))
    assert np.array_equal(back.psi, eq.psi)
    assert np.array_equal(back.R_grid, eq.R_grid)
    assert np.array_equal(back.Z_grid, eq.Z_grid)
    assert back.ip == eq.ip and back.time == eq.time
    assert back.r_axis == eq.r_axis and back.z_axis == eq.z_axis
    assert back.psi_axis == eq.psi_axis and back.psi_boundary == eq.psi_boundary


def test_orientation_marker_lands_at_imas_r_z() -> None:
    """A one-hot marker at solver ``(iz, ir)`` must appear at IMAS ``[ir, iz]`` — checked directly
    on the stored IDS array, so a transposition bug cannot cancel through the round trip."""
    iz, ir = 3, 5
    psi = np.zeros((_NZ, _NR))
    psi[iz, ir] = 7.25
    ods = equilibrium_to_ods(_slice(psi=psi))
    stored = np.asarray(ods["equilibrium.time_slice.0.profiles_2d.0.psi"])
    assert stored.shape == (_NR, _NZ)  # IMAS [dim1, dim2] = [R, Z]
    assert stored[ir, iz] == 7.25
    assert np.count_nonzero(stored) == 1


# ── Fail-closed guards ────────────────────────────────────────────────────


def test_wrong_psi_shape_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not match solver convention"):
        equilibrium_to_ods(_slice(psi=np.zeros((_NR, _NZ))))  # transposed input rejected


def test_non_rectangular_grid_fails_closed() -> None:
    ods = equilibrium_to_ods(_slice())
    ods["equilibrium.time_slice.0.profiles_2d.0.grid_type.index"] = 91  # irregular grid type
    with pytest.raises(ValueError, match="rectangular"):
        ods_to_equilibrium(ods)


def test_corrupted_stored_shape_fails_closed() -> None:
    ods = equilibrium_to_ods(_slice())
    ods["equilibrium.time_slice.0.profiles_2d.0.grid.dim1"] = np.linspace(1.0, 2.5, 4)
    with pytest.raises(ValueError, match="stored psi shape"):
        ods_to_equilibrium(ods)


# ── IMAS schema enforcement + honest absence ──────────────────────────────


def test_ids_paths_are_schema_valid() -> None:
    """OMAS validates every assignment against the IMAS data dictionary — and rejects paths that
    are not in the schema, so the bridge cannot silently invent IDS structure."""
    ods = equilibrium_to_ods(_slice())
    assert ods.consistency_check
    with pytest.raises(LookupError):
        ods["equilibrium.time_slice.0.profiles_2d.0.not_a_real_field"] = 1.0


def test_optional_quantities_absent_not_fabricated() -> None:
    """Uncomputed optional quantities must be ABSENT from the IDS, not written as zeros."""
    ods = equilibrium_to_ods(
        _slice(ip=None, r_axis=None, z_axis=None, psi_axis=None, psi_boundary=None)
    )
    assert "global_quantities.ip" not in ods["equilibrium.time_slice.0"]
    back = ods_to_equilibrium(ods)
    assert back.ip is None and back.r_axis is None and back.psi_boundary is None
    assert np.array_equal(back.psi, ods_to_equilibrium(ods).psi)  # field itself intact


# ── File persistence ──────────────────────────────────────────────────────


def test_json_file_round_trip(tmp_path: Path) -> None:
    eq = _slice()
    path = str(tmp_path / "equilibrium_ids.json")
    save_equilibrium_ids(equilibrium_to_ods(eq), path)
    back = ods_to_equilibrium(load_equilibrium_ids(path))
    assert np.array_equal(back.psi, eq.psi)
    assert back.ip == eq.ip and back.time == eq.time


def test_load_rejects_a_non_ods_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loader that yields anything but an ODS is rejected, not passed through."""
    from scpn_fusion.core import imas_equilibrium_io as mod

    monkeypatch.setattr(mod, "load_omas_json", lambda _path: {"not": "an ods"})
    with pytest.raises(TypeError, match="expected an ODS"):
        mod.load_equilibrium_ids("whatever.json")
