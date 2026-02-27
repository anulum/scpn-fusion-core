# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Synthetic Shot Database Tests
# ──────────────────────────────────────────────────────────────────────
"""Tests for generate_synthetic_shot_database / list / load round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.io.tokamak_archive import (
    generate_synthetic_shot_database,
    list_synthetic_shots,
    load_synthetic_shot,
)


# -- Machine parameter bounds for physics-realism checks ------------------

_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "ITER": {
        "Ip_MA": (14.5, 15.5),
        "BT_T": (5.0, 5.6),
        "ne0_1e19": (10.0, 12.0),
        "Te0_keV": (8.0, 25.0),
        "q95": (1.5, 3.5),          # disruptions may push q95 down
        "beta_N": (1.8, 3.75),      # disruptions may push beta_N up
    },
    "SPARC": {
        "Ip_MA": (8.5, 9.0),
        "BT_T": (12.0, 12.5),
        "ne0_1e19": (25.0, 40.0),
        "Te0_keV": (10.0, 22.0),
        "q95": (1.5, 4.0),
        "beta_N": (1.5, 4.2),
    },
    "DIII-D": {
        "Ip_MA": (0.95, 2.05),
        "BT_T": (2.0, 2.2),
        "ne0_1e19": (3.0, 8.0),
        "Te0_keV": (2.0, 8.0),
        "q95": (1.5, 6.0),
        "beta_N": (1.0, 4.5),
    },
    "EAST": {
        "Ip_MA": (0.35, 1.05),
        "BT_T": (2.4, 2.6),
        "ne0_1e19": (2.0, 5.0),
        "Te0_keV": (1.0, 5.0),
        "q95": (1.5, 7.0),
        "beta_N": (0.8, 3.3),
    },
}


def test_generate_creates_npz_files(tmp_path: Path) -> None:
    """generate_synthetic_shot_database writes NPZ files to output_dir."""
    catalogue = generate_synthetic_shot_database(output_dir=tmp_path, seed=42)
    npz_files = list(tmp_path.glob("*.npz"))
    assert len(npz_files) >= 50
    assert len(catalogue) == len(npz_files)


def test_at_least_50_shots(tmp_path: Path) -> None:
    """Default call must produce at least 50 shots."""
    catalogue = generate_synthetic_shot_database(output_dir=tmp_path)
    assert len(catalogue) >= 50


def test_npz_has_required_keys(tmp_path: Path) -> None:
    """Every NPZ file must contain the documented keys."""
    generate_synthetic_shot_database(output_dir=tmp_path, seed=7)
    required = {
        "time_s", "Ip_MA", "BT_T", "ne_1e19", "Te_keV", "Ti_keV",
        "q95", "beta_N", "disruption_label", "machine",
    }
    for npz_path in tmp_path.glob("*.npz"):
        with np.load(str(npz_path), allow_pickle=False) as data:
            present = set(data.files)
            missing = required - present
            assert not missing, f"{npz_path.name} missing keys: {missing}"


def test_array_shapes(tmp_path: Path) -> None:
    """Array traces/profiles must have 1000 points."""
    generate_synthetic_shot_database(output_dir=tmp_path, n_shots=10, seed=99)
    array_keys = {"time_s", "Ip_MA", "BT_T", "ne_1e19", "Te_keV", "Ti_keV", "q95", "beta_N"}
    for npz_path in tmp_path.glob("*.npz"):
        with np.load(str(npz_path), allow_pickle=False) as data:
            for k in array_keys:
                arr = np.asarray(data[k], dtype=np.float64)
                assert arr.shape == (1000,), f"{npz_path.name}[{k}] shape {arr.shape}"


def test_four_machines_represented(tmp_path: Path) -> None:
    """All four machines must appear in the catalogue."""
    catalogue = generate_synthetic_shot_database(output_dir=tmp_path, seed=123)
    machines = {entry["machine"] for entry in catalogue}
    assert machines == {"ITER", "SPARC", "DIII-D", "EAST"}


def test_parameter_ranges_are_realistic(tmp_path: Path) -> None:
    """Scalar parameters must fall within physically realistic ranges."""
    catalogue = generate_synthetic_shot_database(output_dir=tmp_path, seed=55)
    for entry in catalogue:
        machine = entry["machine"]
        bounds = _BOUNDS[machine]
        for key, (lo, hi) in bounds.items():
            val = entry[key]
            assert lo <= val <= hi, (
                f"{entry['shot_id']} ({machine}): {key}={val} out of [{lo}, {hi}]"
            )


def test_disruption_rate_approximately_20_percent(tmp_path: Path) -> None:
    """Roughly 20% of shots should be flagged as disruptions."""
    catalogue = generate_synthetic_shot_database(output_dir=tmp_path, n_shots=100, seed=314)
    n_disrupt = sum(1 for e in catalogue if e["disruption_label"])
    rate = n_disrupt / len(catalogue)
    assert 0.08 <= rate <= 0.35, f"Disruption rate {rate:.2%} outside [8%, 35%]"


def test_disruption_shots_have_higher_beta_n(tmp_path: Path) -> None:
    """On average, disruption shots should have elevated beta_N."""
    catalogue = generate_synthetic_shot_database(output_dir=tmp_path, n_shots=200, seed=271)
    disrupt = [e["beta_N"] for e in catalogue if e["disruption_label"]]
    stable = [e["beta_N"] for e in catalogue if not e["disruption_label"]]
    if disrupt and stable:
        assert np.mean(disrupt) >= np.mean(stable) * 0.9


def test_list_synthetic_shots_round_trip(tmp_path: Path) -> None:
    """list_synthetic_shots returns names matching generated files."""
    generate_synthetic_shot_database(output_dir=tmp_path, seed=11)
    names = list_synthetic_shots(synthetic_dir=tmp_path)
    npz_stems = sorted(p.stem for p in tmp_path.glob("*.npz"))
    assert names == npz_stems
    assert len(names) >= 50


def test_load_synthetic_shot_round_trip(tmp_path: Path) -> None:
    """load_synthetic_shot returns correct data for a generated shot."""
    generate_synthetic_shot_database(output_dir=tmp_path, seed=22)
    names = list_synthetic_shots(synthetic_dir=tmp_path)
    assert names, "No shots generated"
    shot = load_synthetic_shot(names[0], synthetic_dir=tmp_path)
    assert "time_s" in shot
    assert "machine" in shot
    assert isinstance(shot["disruption_label"], bool)
    assert shot["time_s"].shape == (1000,)
    assert np.all(np.isfinite(shot["Te_keV"]))


def test_load_synthetic_shot_by_path(tmp_path: Path) -> None:
    """load_synthetic_shot accepts an absolute path."""
    generate_synthetic_shot_database(output_dir=tmp_path, seed=33)
    first_npz = sorted(tmp_path.glob("*.npz"))[0]
    shot = load_synthetic_shot(first_npz)
    assert shot["Ip_MA"].shape == (1000,)


def test_load_synthetic_shot_missing_raises(tmp_path: Path) -> None:
    """FileNotFoundError for a shot that does not exist."""
    with pytest.raises(FileNotFoundError):
        load_synthetic_shot("nonexistent_shot_999", synthetic_dir=tmp_path)


def test_load_synthetic_shot_rejects_object_array_payload(tmp_path: Path) -> None:
    """Loader should reject object-array NPZ payloads under secure defaults."""
    bad_path = tmp_path / "bad_object_payload.npz"
    zeros = np.zeros(1000, dtype=np.float64)
    np.savez(
        bad_path,
        time_s=zeros,
        Ip_MA=zeros,
        BT_T=zeros,
        ne_1e19=zeros,
        Te_keV=zeros,
        Ti_keV=zeros,
        q95=zeros,
        beta_N=zeros,
        disruption_label=np.array([True], dtype=object),
        machine=np.array(["ITER"], dtype=object),
    )
    with pytest.raises(ValueError):
        load_synthetic_shot(bad_path)


def test_list_synthetic_shots_empty_dir(tmp_path: Path) -> None:
    """list_synthetic_shots returns [] for a non-existent directory."""
    empty = tmp_path / "no_such_dir"
    assert list_synthetic_shots(synthetic_dir=empty) == []


def test_profiles_are_parabolic(tmp_path: Path) -> None:
    """ne/Te profiles should be monotonically non-increasing (core > edge)."""
    generate_synthetic_shot_database(output_dir=tmp_path, n_shots=10, seed=77)
    for npz_path in list(tmp_path.glob("*.npz"))[:5]:
        shot = load_synthetic_shot(npz_path)
        ne = shot["ne_1e19"]
        te = shot["Te_keV"]
        # Core (index 0) should be >= edge (index 999)
        assert ne[0] >= ne[-1], f"ne not peaked: core={ne[0]}, edge={ne[-1]}"
        assert te[0] >= te[-1], f"Te not peaked: core={te[0]}, edge={te[-1]}"


def test_reproducible_with_same_seed(tmp_path: Path) -> None:
    """Same seed produces identical catalogues."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    cat_a = generate_synthetic_shot_database(output_dir=dir_a, seed=42)
    cat_b = generate_synthetic_shot_database(output_dir=dir_b, seed=42)
    assert len(cat_a) == len(cat_b)
    for a, b in zip(cat_a, cat_b):
        assert a == b
