# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Safe Loader Tests
"""Tests for bounded JSON and NumPy archive loader contracts."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import IO, Any, Callable, cast

import numpy as np
import pytest

from scpn_fusion.io.safe_loaders import (
    MAX_JSON_BYTES,
    MAX_NPZ_BYTES,
    checked_json_load,
    checked_np_load,
)

_write_array_header = cast(
    Callable[[IO[bytes], dict[str, object], tuple[int, int]], None],
    vars(np.lib.format)["_write_array_header"],
)


def _npy_header(
    *,
    shape: tuple[int, ...],
    descr: str = "|u1",
    version: tuple[int, int] = (1, 0),
) -> bytes:
    """Build a valid NPY header without materialising its declared array."""
    stream = io.BytesIO()
    _write_array_header(
        stream,
        {"descr": descr, "fortran_order": False, "shape": shape},
        version,
    )
    return stream.getvalue()


def _forbid_numpy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Replace NumPy loading with a call recorder that always fails."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _load(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        raise AssertionError("numpy.load must not run for a rejected header")

    monkeypatch.setattr(np, "load", _load)
    return calls


def test_checked_json_load_rejects_oversized_file(tmp_path: Path) -> None:
    """Reject JSON files above the stored-byte limit."""
    path = tmp_path / "huge.json"
    with path.open("wb") as handle:
        handle.truncate(MAX_JSON_BYTES + 1)

    with pytest.raises(ValueError, match="JSON file too large"):
        checked_json_load(path)


def test_checked_np_load_rejects_oversized_archive(tmp_path: Path) -> None:
    """Reject NumPy files above the stored-byte limit."""
    path = tmp_path / "huge.npz"
    with path.open("wb") as handle:
        handle.truncate(MAX_NPZ_BYTES + 1)

    with pytest.raises(ValueError, match="NumPy archive file too large"):
        checked_np_load(path)


def test_checked_np_load_rejects_oversized_expanded_member(tmp_path: Path) -> None:
    """Reject one NPZ member above its expanded-byte limit."""
    path = tmp_path / "compressed_member.npz"
    np.savez_compressed(path, payload=np.zeros(512, dtype=np.uint8))

    with pytest.raises(ValueError, match="member too large"):
        checked_np_load(path, max_member_bytes=512)


def test_checked_np_load_rejects_oversized_expanded_total(tmp_path: Path) -> None:
    """Reject NPZ members above their aggregate expanded-byte limit."""
    path = tmp_path / "compressed_total.npz"
    np.savez_compressed(
        path,
        first=np.zeros(128, dtype=np.uint8),
        second=np.zeros(128, dtype=np.uint8),
        third=np.zeros(128, dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="expands too large"):
        checked_np_load(path, max_member_bytes=512, max_total_bytes=700)


def test_checked_np_load_accepts_bounded_npz_and_plain_npy(tmp_path: Path) -> None:
    """Load bounded NPZ and plain NPY inputs without changing their arrays."""
    archive_path = tmp_path / "bounded.npz"
    array_path = tmp_path / "bounded.npy"
    expected = np.arange(16, dtype=np.float64)
    np.savez_compressed(archive_path, payload=expected)
    np.save(array_path, expected)

    with checked_np_load(
        archive_path,
        max_member_bytes=512,
        max_total_bytes=512,
    ) as archive:
        np.testing.assert_array_equal(archive["payload"], expected)
    np.testing.assert_array_equal(
        checked_np_load(array_path, max_member_bytes=512, max_total_bytes=512),
        expected,
    )


def test_checked_np_load_rejects_forged_npz_header_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a tiny ZIP member whose NPY header declares a 1 GiB array."""
    path = tmp_path / "forged.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload.npy", _npy_header(shape=(1_073_741_824,)))
    calls = _forbid_numpy_load(monkeypatch)

    with pytest.raises(ValueError, match="logical payload too large"):
        checked_np_load(path)

    assert calls == []
    assert path.stat().st_size < 256


def test_checked_np_load_rejects_forged_plain_npy_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the nested-header logical-size bound to plain NPY files too."""
    path = tmp_path / "forged.npy"
    path.write_bytes(_npy_header(shape=(1_073_741_824,)))
    calls = _forbid_numpy_load(monkeypatch)

    with pytest.raises(ValueError, match="logical payload too large"):
        checked_np_load(path)

    assert calls == []


def test_checked_np_load_rejects_overflow_scale_shape_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short-circuit multiplicative shape overflow before loading."""
    path = tmp_path / "overflow.npy"
    path.write_bytes(_npy_header(shape=(2**63, 2**63)))
    calls = _forbid_numpy_load(monkeypatch)

    with pytest.raises(ValueError, match="logical payload too large"):
        checked_np_load(path)

    assert calls == []


def test_checked_np_load_rejects_object_dtype_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject pickle-capable object dtypes during header validation."""
    path = tmp_path / "object.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload.npy", _npy_header(shape=(1,), descr="|O"))
    calls = _forbid_numpy_load(monkeypatch)

    with pytest.raises(ValueError, match="object dtype is not allowed"):
        checked_np_load(path)

    assert calls == []


def test_checked_np_load_rejects_logical_npz_aggregate_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound aggregate declared arrays independently of ZIP member sizes."""
    path = tmp_path / "logical_total.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("first.npy", _npy_header(shape=(400,)))
        archive.writestr("second.npy", _npy_header(shape=(400,)))
    calls = _forbid_numpy_load(monkeypatch)

    with pytest.raises(ValueError, match="logical payload too large"):
        checked_np_load(path, max_member_bytes=512, max_total_bytes=700)

    assert calls == []


def test_checked_np_load_rejects_invalid_shape_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject negative dimensions before NumPy sees a malformed header."""
    path = tmp_path / "negative.npy"
    path.write_bytes(_npy_header(shape=(-1,)))
    calls = _forbid_numpy_load(monkeypatch)

    with pytest.raises(ValueError, match="shape is invalid"):
        checked_np_load(path)

    assert calls == []


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_checked_np_load_accepts_supported_header_versions(
    tmp_path: Path,
    version: tuple[int, int],
) -> None:
    """Parse every NPY header version supported by the pinned NumPy."""
    path = tmp_path / "zero.npy"
    path.write_bytes(_npy_header(shape=(0, 4), descr="<f8", version=version))

    loaded = checked_np_load(path, max_member_bytes=0, max_total_bytes=0)

    assert loaded.shape == (0, 4)
    assert loaded.dtype == np.dtype("<f8")


def test_checked_np_load_accepts_zero_width_dtype(tmp_path: Path) -> None:
    """Treat a zero-width dtype as zero logical bytes without a payload."""
    path = tmp_path / "zero_width.npy"
    path.write_bytes(_npy_header(shape=(1,), descr="|S0"))

    loaded = checked_np_load(path, max_member_bytes=0, max_total_bytes=0)

    assert loaded.shape == (1,)
    assert loaded.dtype == np.dtype("S0")
    assert loaded.nbytes == 0


def test_checked_np_load_rejects_scalar_over_member_limit(tmp_path: Path) -> None:
    """Bound the implicit one element represented by a scalar shape."""
    path = tmp_path / "scalar.npy"
    np.save(path, np.array(3.0, dtype=np.float64))

    with pytest.raises(ValueError, match="logical payload too large"):
        checked_np_load(path, max_member_bytes=1)


def test_checked_np_load_rejects_plain_npy_over_total_limit(tmp_path: Path) -> None:
    """Apply the aggregate logical limit to a single plain NPY array."""
    path = tmp_path / "plain_total.npy"
    np.save(path, np.arange(16, dtype=np.float64))

    with pytest.raises(ValueError, match="aggregate limit"):
        checked_np_load(path, max_member_bytes=512, max_total_bytes=64)


def test_checked_np_load_preserves_bounded_non_npy_zip_member(tmp_path: Path) -> None:
    """Count ancillary ZIP members without attempting to parse them as NPY."""
    path = tmp_path / "ancillary.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload.npy", _npy_header(shape=(0,)))
        archive.writestr("README.txt", b"bounded metadata")

    with checked_np_load(path, max_member_bytes=256, max_total_bytes=512) as archive:
        assert archive.files == ["payload", "README.txt"]


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"max_bytes": -1}, "max_bytes"),
        ({"max_member_bytes": -1}, "max_member_bytes"),
        ({"max_total_bytes": -1}, "max_total_bytes"),
        ({"max_header_bytes": -1}, "max_header_bytes"),
        ({"max_bytes": True}, "max_bytes"),
    ],
)
def test_checked_np_load_rejects_invalid_limits(
    tmp_path: Path,
    kwargs: dict[str, Any],
    name: str,
) -> None:
    """Reject negative and boolean byte limits before file inspection."""
    with pytest.raises(ValueError, match=rf"{name} must be a non-negative integer"):
        checked_np_load(tmp_path / "missing.npy", **kwargs)
