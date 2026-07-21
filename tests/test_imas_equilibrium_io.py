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


def test_round_trip_is_faithful_to_rounding() -> None:
    """Round trip through the IDS is exact up to one COCOS rounding op each way (ψ-class values
    are stored ×(−2π) in the IMAS frame and divided back on read — a single float multiply and
    divide, so agreement is to ~1 ulp, asserted at rtol 1e-14). Grids and toroidal-class values
    are untransformed and stay bit-exact."""
    eq = _slice()
    back = ods_to_equilibrium(equilibrium_to_ods(eq))
    assert np.allclose(back.psi, eq.psi, rtol=1e-14, atol=0.0)
    assert np.array_equal(back.R_grid, eq.R_grid)
    assert np.array_equal(back.Z_grid, eq.Z_grid)
    assert back.ip == eq.ip and back.time == eq.time
    assert back.r_axis == eq.r_axis and back.z_axis == eq.z_axis
    assert back.psi_axis == pytest.approx(eq.psi_axis, rel=1e-14)
    assert back.psi_boundary == pytest.approx(eq.psi_boundary, rel=1e-14)


def test_orientation_marker_lands_at_imas_r_z() -> None:
    """A one-hot marker at solver ``(iz, ir)`` must appear at IMAS ``[ir, iz]`` — checked directly
    on the stored IDS array, so a transposition bug cannot cancel through the round trip.
    The stored value carries the COCOS 3 → 11 factor (−2π): the IDS holds IMAS-frame ψ."""
    iz, ir = 3, 5
    psi = np.zeros((_NZ, _NR))
    psi[iz, ir] = 7.25
    ods = equilibrium_to_ods(_slice(psi=psi))
    stored = np.asarray(ods["equilibrium.time_slice.0.profiles_2d.0.psi"])
    assert stored.shape == (_NR, _NZ)  # IMAS [dim1, dim2] = [R, Z]
    assert stored[ir, iz] == pytest.approx(-2.0 * np.pi * 7.25)
    assert np.count_nonzero(stored) == 1


# ── COCOS (solver frame 3 ⇄ IMAS internal 11, via OMAS's own machinery) ────


def test_cocos_transform_applied_to_stored_ids() -> None:
    """ψ-class quantities are stored in the IMAS frame (COCOS 11 = −2π × solver COCOS 3);
    ``ip`` is a toroidal-class quantity and is unchanged between 3 and 11 (same σ_RφZ)."""
    eq = _slice()
    ods = equilibrium_to_ods(eq)
    ts = "equilibrium.time_slice.0."
    assert float(ods[ts + "global_quantities.psi_axis"]) == pytest.approx(
        -2.0 * np.pi * eq.psi_axis
    )
    assert float(ods[ts + "global_quantities.psi_boundary"]) == pytest.approx(
        -2.0 * np.pi * eq.psi_boundary
    )
    assert float(ods[ts + "global_quantities.ip"]) == pytest.approx(eq.ip)


def test_round_trip_identity_for_the_opposite_handedness() -> None:
    """The φ-handedness is unobservable to the 2-D solver, so ``solver_cocos=4`` (the even
    partner) must also round-trip to identity when used consistently on both sides."""
    eq = _slice()
    back = ods_to_equilibrium(equilibrium_to_ods(eq, solver_cocos=4), solver_cocos=4)
    assert np.allclose(back.psi, eq.psi, rtol=1e-14, atol=0.0)
    assert back.ip == pytest.approx(eq.ip)
    assert back.psi_axis == pytest.approx(eq.psi_axis, rel=1e-14)


def test_cocos4_stored_frame_measured_explicitly() -> None:
    """Independent stored-frame pin for the handedness partner (no self-cancelling round trip):
    for solver COCOS 4 → IMAS 11 the measured transform is ψ ↦ +2π·ψ and ``ip ↦ −ip``
    (σ_RφZ differs between 4 and 11, so toroidal-class quantities flip sign)."""
    iz, ir = 3, 5
    psi = np.zeros((_NZ, _NR))
    psi[iz, ir] = 7.25
    ods = equilibrium_to_ods(_slice(psi=psi), solver_cocos=4)
    stored = np.asarray(ods["equilibrium.time_slice.0.profiles_2d.0.psi"])
    assert stored[ir, iz] == pytest.approx(+2.0 * np.pi * 7.25)
    ts = "equilibrium.time_slice.0."
    assert float(ods[ts + "global_quantities.ip"]) == pytest.approx(-1.5e6)
    assert float(ods[ts + "global_quantities.psi_axis"]) == pytest.approx(+2.0 * np.pi * 0.91)


# ── Fail-closed guards ────────────────────────────────────────────────────


def test_unaudited_solver_cocos_fails_closed() -> None:
    """Only the audited pair {3, 4} is accepted; 11 in particular would silently pass
    IMAS-frame ψ through untransformed and must be rejected in BOTH directions."""
    ods = equilibrium_to_ods(_slice())
    for bad in (11, 1, 2, 5, 0, -3):
        with pytest.raises(ValueError, match="solver_cocos must be one of"):
            equilibrium_to_ods(_slice(), solver_cocos=bad)
        with pytest.raises(ValueError, match="solver_cocos must be one of"):
            ods_to_equilibrium(ods, solver_cocos=bad)


def test_non_finite_psi_fails_closed() -> None:
    for poison in (np.nan, np.inf, -np.inf):
        psi = np.zeros((_NZ, _NR))
        psi[2, 2] = poison
        with pytest.raises(ValueError, match="psi contains non-finite"):
            equilibrium_to_ods(_slice(psi=psi))


def test_non_finite_scalars_and_time_fail_closed() -> None:
    with pytest.raises(ValueError, match="psi_axis must be finite"):
        equilibrium_to_ods(_slice(psi_axis=float("nan")))
    with pytest.raises(ValueError, match="ip must be finite"):
        equilibrium_to_ods(_slice(ip=float("inf")))
    with pytest.raises(ValueError, match="time must be finite"):
        equilibrium_to_ods(_slice(time=float("nan")))


def test_nonmonotone_or_duplicate_grid_fails_closed() -> None:
    r_dup = np.array([1.0, 1.25, 1.25, 1.75, 2.0, 2.25, 2.5])
    with pytest.raises(ValueError, match="R_grid must be strictly increasing"):
        equilibrium_to_ods(_slice(R_grid=r_dup))
    with pytest.raises(ValueError, match="Z_grid must be strictly increasing"):
        equilibrium_to_ods(_slice(Z_grid=np.asarray(_Z)[::-1].copy()))
    with pytest.raises(ValueError, match="at least 2 points"):
        equilibrium_to_ods(_slice(R_grid=np.array([1.7]), psi=np.zeros((_NZ, 1))))


def test_read_validates_a_corrupted_ids() -> None:
    """The read direction enforces the same invariants — a corrupted or foreign IDS is
    rejected, not passed through into the solvers."""
    ods = equilibrium_to_ods(_slice())
    ods["equilibrium.time_slice.0.profiles_2d.0.grid.dim1"] = np.array(
        [1.0, 1.2, 1.2, 1.8, 2.0, 2.2, 2.5]
    )
    with pytest.raises(ValueError, match="R_grid must be strictly increasing"):
        ods_to_equilibrium(ods)
    ods2 = equilibrium_to_ods(_slice())
    poisoned = np.asarray(ods2["equilibrium.time_slice.0.profiles_2d.0.psi"]).copy()
    poisoned[1, 1] = np.nan
    ods2["equilibrium.time_slice.0.profiles_2d.0.psi"] = poisoned
    with pytest.raises(ValueError, match="psi contains non-finite"):
        ods_to_equilibrium(ods2)


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
    assert np.allclose(back.psi, eq.psi, rtol=1e-14, atol=0.0)  # 1 COCOS rounding op each way
    assert back.ip == eq.ip and back.time == eq.time


def test_load_rejects_a_non_ods_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loader that yields anything but an ODS is rejected, not passed through."""
    from scpn_fusion.core import imas_equilibrium_io as mod

    monkeypatch.setattr(mod, "load_omas_json", lambda _path: {"not": "an ods"})
    with pytest.raises(TypeError, match="expected an ODS"):
        mod.load_equilibrium_ids("whatever.json")
