# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HPC Bridge
"""Trust-checked Python bridge to optional native HPC Grad-Shafranov solver."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import logging
import math
import numpy as np
import os
import platform
import shutil
import stat
import subprocess
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

from numpy.typing import NDArray

logger = logging.getLogger(__name__)
_CPP_BUILD_TIMEOUT_SECONDS = 300.0
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
    digest = value.strip().split()[0].lower()
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


def _default_library_path(lib_name: str) -> Path:
    """Return a package-local native-library path without consulting cwd."""
    here = Path(__file__).resolve().parent
    candidates = (
        here / lib_name,
        here / "bin" / lib_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (here / lib_name).resolve()


def _write_sha256_sidecar(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{_sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )


class HPCBridge:
    """Interface between Python and the compiled C++ Grad-Shafranov solver.

    Loads the shared library (``libscpn_solver.so`` / ``scpn_solver.dll``)
    at construction time.  If the library is not found the bridge
    gracefully degrades — :meth:`is_available` returns ``False`` and the
    caller falls back to Python.

    Parameters
    ----------
    lib_path : str, optional
        Explicit path to the shared library.  When *None* (default) the
        bridge searches trusted package-local locations only, unless
        ``SCPN_SOLVER_LIB`` is set explicitly.
    """

    def __init__(self, lib_path: Optional[str] = None) -> None:
        self.lib: Any | None = None
        self.solver_ptr = None
        self.loaded: bool = False
        self.load_error: Optional[str] = None
        self.close_error: Optional[str] = None
        self.close_error_count: int = 0
        self.lib_sha256: Optional[str] = None
        self._destroy_symbol: Optional[str] = None
        self._has_converged_api: bool = False
        self._has_boundary_api: bool = False

        lib_name = "scpn_solver.dll" if platform.system() == "Windows" else "libscpn_solver.so"
        env_path = os.environ.get(_SOLVER_LIB_ENV)
        explicit_override = lib_path is not None or bool(env_path)
        if lib_path is None and env_path:
            lib_path = env_path

        try:
            if explicit_override:
                resolved_lib_path = _require_explicit_library_path(str(lib_path))
            else:
                resolved_lib_path = _default_library_path(lib_name)
        except ValueError as exc:
            self.lib_path = str(lib_path)
            self.load_error = str(exc)
            logger.warning("Refusing native C++ accelerator override: %s", exc)
            return

        self.lib_path = str(resolved_lib_path)

        try:
            lib_file = resolved_lib_path
            if lib_file.exists():
                self.lib_sha256 = _verify_native_library_trust(lib_file)
            self.lib = ctypes.CDLL(self.lib_path)
            self._setup_signatures()
            logger.info("Loaded C++ accelerator: %s", self.lib_path)
            self.loaded = True
        except ValueError as exc:
            self.load_error = str(exc)
            logger.warning("Refusing untrusted C++ accelerator at %s: %s", self.lib_path, exc)
        except OSError as exc:
            self.load_error = str(exc)
            logger.debug(
                "C++ accelerator unavailable at %s; falling back to Python solver: %s",
                self.lib_path,
                self.load_error,
            )

    def is_available(self) -> bool:
        """Return *True* if the compiled solver library was loaded."""
        return self.loaded

    def close(self) -> None:
        """Release the C++ solver instance, if one was created."""
        if self.solver_ptr is not None and self.loaded:
            try:
                if self.lib is not None and self._destroy_symbol is not None:
                    getattr(self.lib, self._destroy_symbol)(self.solver_ptr)
            except Exception as exc:
                self.close_error = str(exc)
                self.close_error_count = int(getattr(self, "close_error_count", 0)) + 1
                logger.warning(
                    "Failed to release C++ solver instance cleanly: %s", self.close_error
                )
            self.solver_ptr = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "HPCBridge":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _setup_signatures(self) -> None:
        if self.lib is None:
            return
        # void* create_solver(int nr, int nz, double rmin, double rmax, double zmin, double zmax)
        self.lib.create_solver.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self.lib.create_solver.restype = ctypes.c_void_p

        # void run_step(void* solver, double* j, double* psi, int size, int iter)
        self.lib.run_step.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
        ]

        # int run_step_converged(void* solver, const double* j, double* psi,
        #                        int size, int max_iter, double omega,
        #                        double tol, double* final_delta)
        if hasattr(self.lib, "run_step_converged"):
            self.lib.run_step_converged.argtypes = [
                ctypes.c_void_p,
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
            ]
            self.lib.run_step_converged.restype = ctypes.c_int
            self._has_converged_api = True
        else:
            self._has_converged_api = False

        # void set_boundary_dirichlet(void* solver, double boundary_value)
        if hasattr(self.lib, "set_boundary_dirichlet"):
            self.lib.set_boundary_dirichlet.argtypes = [ctypes.c_void_p, ctypes.c_double]
            self.lib.set_boundary_dirichlet.restype = None
            self._has_boundary_api = True
        else:
            self._has_boundary_api = False

        # void destroy_solver(void* solver) or void delete_solver(void* solver)
        if hasattr(self.lib, "destroy_solver"):
            self.lib.destroy_solver.argtypes = [ctypes.c_void_p]
            self.lib.destroy_solver.restype = None
            self._destroy_symbol = "destroy_solver"
        elif hasattr(self.lib, "delete_solver"):
            self.lib.delete_solver.argtypes = [ctypes.c_void_p]
            self.lib.delete_solver.restype = None
            self._destroy_symbol = "delete_solver"
        else:
            self._destroy_symbol = None

    def initialize(
        self,
        nr: int,
        nz: int,
        r_range: tuple[float, float],
        z_range: tuple[float, float],
        boundary_value: float = 0.0,
    ) -> None:
        """Create the C++ solver instance for the given grid dimensions."""
        if not self.loaded:
            return
        if nr < 2 or nz < 2:
            raise ValueError("nr and nz must be >= 2.")
        if r_range[0] >= r_range[1] or z_range[0] >= z_range[1]:
            raise ValueError("r_range/z_range must have min < max.")
        if self.lib is None:
            return
        self.nr = nr
        self.nz = nz
        self.solver_ptr = self.lib.create_solver(
            nr, nz, r_range[0], r_range[1], z_range[0], z_range[1]
        )
        self.set_boundary_dirichlet(boundary_value)

    def set_boundary_dirichlet(self, boundary_value: float = 0.0) -> None:
        """Set a fixed Dirichlet boundary value for psi edges, if supported."""
        if (
            not self.loaded
            or self.solver_ptr is None
            or self.lib is None
            or not self._has_boundary_api
        ):
            return
        self.lib.set_boundary_dirichlet(self.solver_ptr, float(boundary_value))

    def solve(
        self,
        j_phi: NDArray[np.float64],
        iterations: int = 100,
    ) -> Optional[NDArray[np.float64]]:
        """Run the C++ solver for *iterations* sweeps.

        Returns *None* if the library is not loaded (caller should
        fall back to a Python solver).
        """
        prepared = self._prepare_inputs(j_phi)
        if prepared is None:
            return None
        _, expected_shape = prepared

        psi_out = np.zeros(expected_shape, dtype=np.float64)
        solved = self.solve_into(j_phi, psi_out, iterations=iterations)
        if solved is None:
            return None
        return solved

    def solve_into(
        self,
        j_phi: NDArray[np.float64],
        psi_out: NDArray[np.float64],
        iterations: int = 100,
    ) -> Optional[NDArray[np.float64]]:
        """Run the C++ solver and write results into ``psi_out`` in-place."""
        prepared = self._prepare_inputs(j_phi)
        if prepared is None:
            return None
        j_input, expected_shape = prepared
        psi_target = _require_c_contiguous_f64(psi_out, expected_shape, "psi_out")
        if self.lib is None:
            return None

        self.lib.run_step(
            self.solver_ptr,
            j_input,
            psi_target,
            int(j_input.size),
            int(iterations),
        )
        return psi_target

    def solve_neural(
        self, config_path: Optional[str | Path] = None
    ) -> Optional[NDArray[np.float64]]:
        """
        Run the O(1) Neural Equilibrium Surrogate.
        Requires NeuralEquilibriumKernel (JAX/NPZ weights).
        """
        try:
            from scpn_fusion.core.neural_equilibrium_kernel import NeuralEquilibriumKernel
        except ImportError:
            logger.warning("NeuralEquilibriumKernel not available (ImportError).")
            return None

        try:
            # Note: NeuralEquilibriumKernel needs a config for grid sizing
            # default to iter_config.json in root if not provided
            if config_path is None:
                config_path = Path(__file__).resolve().parents[3] / "iter_config.json"

            kernel = NeuralEquilibriumKernel(config_path)
            res = kernel.solve_equilibrium()
            psi = res.get("Psi")
            if psi is None:
                return None
            return np.asarray(psi, dtype=np.float64)
        except Exception as exc:
            logger.error("Neural surrogate inference failed: %s", exc)
            return None

    def solve_until_converged(
        self,
        j_phi: NDArray[np.float64],
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        omega: float = 1.8,
    ) -> Optional[tuple[NDArray[np.float64], int, float]]:
        """Run solver until convergence, if native API is available.

        Returns ``(psi, iterations_used, final_delta)``. If the library is
        unavailable or uninitialized, returns ``None``.
        """
        prepared = self._prepare_inputs(j_phi)
        if prepared is None:
            return None
        _, expected_shape = prepared

        psi_out = np.zeros(expected_shape, dtype=np.float64)
        converged = self.solve_until_converged_into(
            j_phi,
            psi_out,
            max_iterations=max_iterations,
            tolerance=tolerance,
            omega=omega,
        )
        if converged is None:
            return None
        iterations_used, final_delta = converged
        return psi_out, iterations_used, final_delta

    def solve_until_converged_into(
        self,
        j_phi: NDArray[np.float64],
        psi_out: NDArray[np.float64],
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        omega: float = 1.8,
    ) -> Optional[tuple[int, float]]:
        """Run convergence API and write results into ``psi_out`` in-place."""
        prepared = self._prepare_inputs(j_phi)
        if prepared is None:
            return None
        j_input, expected_shape = prepared
        psi_target = _require_c_contiguous_f64(psi_out, expected_shape, "psi_out")
        if self.lib is None:
            return None
        max_iters, tol_safe, omega_safe = _sanitize_convergence_params(
            max_iterations, tolerance, omega
        )

        if not self._has_converged_api:
            self.lib.run_step(
                self.solver_ptr,
                j_input,
                psi_target,
                int(j_input.size),
                int(max_iters),
            )
            return int(max_iters), float("nan")

        final_delta = ctypes.c_double(0.0)
        iterations_used = int(
            self.lib.run_step_converged(
                self.solver_ptr,
                j_input,
                psi_target,
                int(j_input.size),
                int(max_iters),
                float(omega_safe),
                float(tol_safe),
                ctypes.byref(final_delta),
            )
        )
        return iterations_used, float(final_delta.value)

    def _prepare_inputs(
        self, j_phi: NDArray[np.float64]
    ) -> Optional[tuple[NDArray[np.float64], tuple[int, int]]]:
        if not self.loaded or self.solver_ptr is None:
            return None

        j_input = _as_contiguous_f64(j_phi)
        if j_input.ndim != 2:
            raise ValueError(f"j_phi must be a 2D array, received ndim={j_input.ndim}")
        if j_input.size == 0:
            raise ValueError("j_phi must be non-empty")
        if not np.all(np.isfinite(j_input)):
            raise ValueError("j_phi must contain only finite values")
        expected_shape = (
            getattr(self, "nz", j_input.shape[0]),
            getattr(self, "nr", j_input.shape[-1]),
        )
        if tuple(j_input.shape) != tuple(expected_shape):
            raise ValueError(
                f"j_phi shape mismatch: expected {expected_shape}, received {tuple(j_input.shape)}"
            )
        return j_input, expected_shape


def compile_cpp() -> Optional[str]:
    """Compile the C++ solver from source.

    Looks for the bundled ``solver.cpp`` in the same directory as this module,
    resolves an allowlisted C++ compiler, and invokes it with fixed arguments
    and a minimal environment. The build is intentionally opt-in because local
    native compilation is code execution.

    Returns
    -------
    str or None
        Path to the compiled library, or *None* on failure.
    """
    if os.environ.get("SCPN_ALLOW_NATIVE_BUILD") != "1":
        logger.warning("Native build disabled. Set SCPN_ALLOW_NATIVE_BUILD=1 to enable.")
        return None

    script_dir = Path(__file__).resolve().parent
    src = script_dir / "solver.cpp"
    if not _validate_cpp_source(src, script_dir):
        return None

    compiler = _resolve_cpp_compiler()
    if compiler is None:
        return None

    logger.info("Compiling C++ solver kernel from trusted bundled source.")
    out_dir = script_dir / "bin"
    out_dir.mkdir(exist_ok=True)

    if platform.system() == "Windows":
        out = out_dir / "scpn_solver.dll"
        cmd = [str(compiler), "-shared", "-o", str(out), str(src), "-O3", "-mavx2"]
    else:
        out = out_dir / "libscpn_solver.so"
        cmd = [
            str(compiler),
            "-shared",
            "-fPIC",
            "-o",
            str(out),
            str(src),
            "-O3",
            "-march=native",
        ]

    logger.info("Executing trusted native build command: %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=_CPP_BUILD_TIMEOUT_SECONDS,
            env=_cpp_build_env(),
        )
    except subprocess.TimeoutExpired as exc:
        logger.error(
            "Compilation timed out after %.1fs: %s",
            _CPP_BUILD_TIMEOUT_SECONDS,
            exc,
        )
        return None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.error("Compilation failed: %s", exc)
        return None

    logger.info("Compilation succeeded: %s", out)
    _write_sha256_sidecar(out)
    return str(out)


if __name__ == "__main__":
    # Test sequence
    lib_file = compile_cpp()

    if lib_file:
        bridge = HPCBridge(lib_file)

        # Test Grid
        N = 100
        bridge.initialize(N, N, (2.0, 10.0), (-5.0, 5.0))

        # Dummy Current (Gaussian)
        J = np.random.rand(N, N)

        # Run
        Psi = bridge.solve(J, iterations=500)
        if Psi is not None:
            print(f"Max Flux: {np.max(Psi)}")
