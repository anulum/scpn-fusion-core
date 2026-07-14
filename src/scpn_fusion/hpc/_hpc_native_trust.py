# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HPC Native Trust Primitives
"""Trust, build, and buffer-validation primitives for the native HPC bridge.

These helpers back :class:`scpn_fusion.hpc.hpc_bridge.HPCBridge` and
:func:`scpn_fusion.hpc.hpc_bridge.compile_cpp`: SHA-256 trust verification of an
optional native solver library, allowlisted compiler resolution, a minimal
build environment, and C-contiguous ``float64`` buffer validation. They hold no
reference to the module ``__file__``, so package-relative path resolution
(``_default_library_path``, ``compile_cpp``) deliberately stays in the bridge
module where tests can monkeypatch ``__file__``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import platform
import shutil
import stat
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_CPP_ALLOWED_COMPILERS = frozenset({"g++", "g++.exe", "clang++", "clang++.exe"})
_SHA256_HEX_LEN = 64
_SOLVER_LIB_ENV = "SCPN_SOLVER_LIB"


def _as_contiguous_f64(array: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Return ``array`` as C-contiguous ``float64`` with minimal copying."""
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def _require_c_contiguous_f64(
    array: NDArray[np.floating[Any]],
    expected_shape: tuple[int, int],
    name: str,
) -> NDArray[np.float64]:
    """Validate that an output buffer can be written into without copying."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray")
    if array.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} shape mismatch: expected {expected_shape}, received {tuple(array.shape)}"
        )
    return array


def _sanitize_convergence_params(
    max_iterations: int,
    tolerance: float,
    omega: float,
) -> tuple[int, float, float]:
    """Validate convergence parameters for native calls."""
    max_iters = int(max_iterations)
    if max_iters < 1:
        raise ValueError("max_iterations must be >= 1.")

    tol_safe = float(tolerance)
    if not math.isfinite(tol_safe) or tol_safe < 0.0:
        raise ValueError("tolerance must be finite and >= 0.")

    omega_safe = float(omega)
    if not math.isfinite(omega_safe) or omega_safe <= 0.0 or omega_safe >= 2.0:
        raise ValueError("omega must be finite and in (0, 2).")

    return max_iters, tol_safe, omega_safe


def _normalise_sha256(value: str) -> str:
    fields = value.strip().split()
    digest = fields[0].lower() if fields else ""
    if len(digest) != _SHA256_HEX_LEN or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("trusted native library digest must be a SHA-256 hex string")
    return digest


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_cpp_compiler() -> Path | None:
    """Return a trusted C++ compiler path without consulting arbitrary flags."""
    compiler = shutil.which("g++", path=os.defpath)
    if compiler is None:
        logger.error("Native build requires g++ on PATH.")
        return None
    compiler_path = Path(compiler).resolve()
    if compiler_path.name not in _CPP_ALLOWED_COMPILERS:
        logger.error("Rejected unsupported C++ compiler executable: %s", compiler_path)
        return None
    if not compiler_path.is_file():
        logger.error("Resolved C++ compiler is not a regular file: %s", compiler_path)
        return None
    mode = compiler_path.stat().st_mode
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        logger.error("Rejected group/world-writable C++ compiler executable: %s", compiler_path)
        return None
    return compiler_path


def _validate_cpp_source(src: Path, script_dir: Path) -> bool:
    """Fail closed unless the build source is the bundled, regular solver file."""
    expected = script_dir / "solver.cpp"
    if src.resolve() != expected.resolve():
        logger.error("Rejected native build source outside bundled solver.cpp: %s", src)
        return False
    if src.is_symlink():
        logger.error("Rejected symlinked native build source: %s", src)
        return False
    if not src.is_file():
        logger.error("Native build source is missing or not a regular file: %s", src)
        return False
    mode = src.stat().st_mode
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        logger.error("Rejected group/world-writable native build source: %s", src)
        return False
    return True


def _cpp_build_env() -> dict[str, str]:
    """Return a minimal build environment, excluding attacker-controlled flags."""
    env: dict[str, str] = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    if platform.system() == "Windows":
        for key in ("SystemRoot", "TEMP", "TMP"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    return env


def _sidecar_digest(path: Path) -> str | None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        return None
    return _normalise_sha256(sidecar.read_text(encoding="utf-8"))


def _manifest_digest(path: Path) -> str | None:
    manifest_path = os.environ.get("SCPN_SOLVER_TRUST_MANIFEST")
    if not manifest_path:
        return None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and "libraries" in manifest:
        manifest = manifest["libraries"]
    if not isinstance(manifest, dict):
        raise ValueError("native library trust manifest must be a JSON object")
    keys = (str(path), str(path.resolve()), path.name)
    for key in keys:
        if key in manifest:
            value = manifest[key]
            if not isinstance(value, str):
                raise ValueError("native library trust manifest digest must be a string")
            return _normalise_sha256(value)
    return None


def _expected_library_digest(path: Path) -> str | None:
    env_digest = os.environ.get("SCPN_SOLVER_LIB_SHA256")
    if env_digest:
        return _normalise_sha256(env_digest)
    manifest = _manifest_digest(path)
    if manifest:
        return manifest
    return _sidecar_digest(path)


def _verify_native_library_trust(path: Path) -> str:
    digest = _sha256_file(path)
    expected = _expected_library_digest(path)
    if expected is None:
        raise ValueError(
            "native solver library trust metadata is required; provide "
            "SCPN_SOLVER_LIB_SHA256, SCPN_SOLVER_TRUST_MANIFEST, or a .sha256 sidecar"
        )
    if not hmac.compare_digest(digest, expected):
        raise ValueError("native solver library SHA-256 does not match trusted metadata")
    return digest


def _require_explicit_library_path(path: str | os.PathLike[str]) -> Path:
    """Validate an explicit native-library override before ``ctypes`` loading."""
    lib_path = Path(path).expanduser()
    if not lib_path.is_absolute():
        raise ValueError(f"{_SOLVER_LIB_ENV}/lib_path must be an absolute path")
    if lib_path.is_symlink():
        raise ValueError(f"explicit native solver library must not be a symlink: {lib_path}")
    resolved = lib_path.resolve(strict=False)
    if not resolved.exists():
        raise ValueError(f"explicit native solver library does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"explicit native solver library must be a regular file: {resolved}")
    return resolved


def _write_sha256_sidecar(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{_sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )
