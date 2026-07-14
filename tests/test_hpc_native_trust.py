# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
"""Direct contract tests for the native-trust primitive layer.

The bridge-level behaviour (HPCBridge lifecycle, compile_cpp, trust
verification through env/manifest/sidecar) is exercised in ``test_hpc_bridge``
and ``test_hpc_bridge_runtime_contracts`` via the ``scpn_fusion.hpc.hpc_bridge``
facade; this module pins the extracted leaf primitives directly through their
own dotted path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scpn_fusion.hpc._hpc_native_trust as native_trust
from scpn_fusion.hpc._hpc_native_trust import (
    _as_contiguous_f64,
    _normalise_sha256,
    _require_c_contiguous_f64,
    _sanitize_convergence_params,
    _sha256_file,
    _sidecar_digest,
    _write_sha256_sidecar,
)


def test_module_exposes_trust_primitive_surface() -> None:
    """The leaf owns the trust/build/validation primitives."""
    for name in (
        "_verify_native_library_trust",
        "_expected_library_digest",
        "_manifest_digest",
        "_resolve_cpp_compiler",
        "_validate_cpp_source",
        "_cpp_build_env",
        "_require_explicit_library_path",
    ):
        assert hasattr(native_trust, name)
    assert native_trust._SHA256_HEX_LEN == 64


def test_as_contiguous_f64_returns_same_array_when_already_conformant() -> None:
    """A C-contiguous float64 array is returned without a copy."""
    arr = np.ascontiguousarray(np.arange(6.0).reshape(2, 3), dtype=np.float64)
    assert _as_contiguous_f64(arr) is arr


def test_as_contiguous_f64_copies_non_contiguous_input() -> None:
    """A non-contiguous or non-float64 array is coerced to a fresh buffer."""
    base = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    view = base[:, ::2]  # non-contiguous slice
    out = _as_contiguous_f64(view)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, view)


def test_require_c_contiguous_f64_accepts_conformant_buffer() -> None:
    """A conformant output buffer of the expected shape is returned unchanged."""
    buf = np.zeros((3, 2), dtype=np.float64)
    assert _require_c_contiguous_f64(buf, (3, 2), "psi_out") is buf


@pytest.mark.parametrize(
    ("buf", "message"),
    [
        (np.zeros((2, 2), dtype=np.float32), "dtype float64"),
        (np.asfortranarray(np.zeros((2, 2), dtype=np.float64)), "C-contiguous"),
        (np.zeros((2, 3), dtype=np.float64), "shape mismatch"),
    ],
)
def test_require_c_contiguous_f64_rejects_nonconformant(buf: object, message: str) -> None:
    """Wrong dtype, layout, or shape are rejected with a descriptive error."""
    with pytest.raises(ValueError, match=message):
        _require_c_contiguous_f64(buf, (2, 2), "psi_out")  # type: ignore[arg-type]


def test_require_c_contiguous_f64_rejects_non_ndarray() -> None:
    """A non-ndarray output buffer is rejected before any numpy access."""
    with pytest.raises(ValueError, match="numpy.ndarray"):
        _require_c_contiguous_f64([[1.0, 2.0]], (1, 2), "psi_out")  # type: ignore[arg-type]


def test_sanitize_convergence_params_normalises_valid_inputs() -> None:
    """Valid convergence parameters are coerced to their runtime types."""
    assert _sanitize_convergence_params(10, 1e-6, 1.8) == (10, 1e-6, 1.8)


@pytest.mark.parametrize(
    ("max_iters", "tol", "omega", "message"),
    [
        (0, 1e-6, 1.8, "max_iterations"),
        (10, -1.0, 1.8, "tolerance"),
        (10, 1e-6, 0.0, "omega"),
        (10, 1e-6, 2.0, "omega"),
        (10, float("nan"), 1.8, "tolerance"),
    ],
)
def test_sanitize_convergence_params_rejects_invalid(
    max_iters: int, tol: float, omega: float, message: str
) -> None:
    """Out-of-range convergence parameters raise a descriptive ValueError."""
    with pytest.raises(ValueError, match=message):
        _sanitize_convergence_params(max_iters, tol, omega)


def test_normalise_sha256_accepts_and_lowercases_a_hex_digest() -> None:
    """A 64-char hex digest (with a trailing filename field) is normalised."""
    digest = "A" * 64
    assert _normalise_sha256(f"{digest}  libscpn_solver.so") == "a" * 64


@pytest.mark.parametrize("bad", ["deadbeef", "z" * 64, "", "g" * 64])
def test_normalise_sha256_rejects_malformed(bad: str) -> None:
    """Anything that is not a 64-char hex string is rejected."""
    with pytest.raises(ValueError, match="SHA-256 hex string"):
        _normalise_sha256(bad)


def test_sidecar_digest_roundtrip(tmp_path: Path) -> None:
    """A sidecar written by the writer is read back as the same digest."""
    lib = tmp_path / "libscpn_solver.so"
    lib.write_bytes(b"native-bytes")
    assert _sidecar_digest(lib) is None  # no sidecar yet
    _write_sha256_sidecar(lib)
    assert _sidecar_digest(lib) == _sha256_file(lib)
