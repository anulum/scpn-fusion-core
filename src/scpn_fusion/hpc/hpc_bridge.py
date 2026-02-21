# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — HPC Bridge
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import ctypes
import logging
import math
import numpy as np
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


try:
    from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumKernel
except ImportError:
    NeuralEquilibriumKernel = None


def _as_contiguous_f64(array: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return ``array`` as C-contiguous ``float64`` with minimal copying."""
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def _require_c_contiguous_f64(
    array: NDArray[np.floating],
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
        self.lib = None
        self.solver_ptr = None
        self.loaded: bool = False
        self._destroy_symbol: Optional[str] = None
        self._has_converged_api: bool = False
        self._has_boundary_api: bool = False

        lib_name = (
            "scpn_solver.dll"
            if platform.system() == "Windows"
            else "libscpn_solver.so"
        )
        env_path = os.environ.get("SCPN_SOLVER_LIB")
        if lib_path is None and env_path:
            lib_path = env_path

        if lib_path is None:
            here = Path(__file__).resolve().parent
            candidates = [
                here / lib_name,
                here / "bin" / lib_name,
            ]
            for c in candidates:
                if c.exists():
                    lib_path = str(c)
                    break
            if lib_path is None:
                lib_path = str(here / lib_name)

        self.lib_path = str(lib_path)

        try:
            self.lib = ctypes.CDLL(self.lib_path)
            self._setup_signatures()
            logger.info("Loaded C++ accelerator: %s", self.lib_path)
            self.loaded = True
        except OSError:
            pass

    def is_available(self) -> bool:
        """Return *True* if the compiled solver library was loaded."""
        return self.loaded

    def close(self) -> None:
        """Release the C++ solver instance, if one was created."""
        if self.solver_ptr is not None and self.loaded:
            try:
                if self.lib is not None and self._destroy_symbol is not None:
                    getattr(self.lib, self._destroy_symbol)(self.solver_ptr)
            except Exception:
                pass
            self.solver_ptr = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "HPCBridge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _setup_signatures(self):
        # void* create_solver(int nr, int nz, double rmin, double rmax, double zmin, double zmax)
        self.lib.create_solver.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.create_solver.restype = ctypes.c_void_p

        # void run_step(void* solver, double* j, double* psi, int size, int iter)
        self.lib.run_step.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int
        ]

        # int run_step_converged(void* solver, const double* j, double* psi,
        #                        int size, int max_iter, double omega,
        #                        double tol, double* final_delta)
        if hasattr(self.lib, "run_step_converged"):
            self.lib.run_step_converged.argtypes = [
                ctypes.c_void_p,
                np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
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

        self.lib.run_step(
            self.solver_ptr,
            j_input,
            psi_target,
            int(j_input.size),
            int(iterations),
        )
        return psi_target

    def solve_neural(self, config_path: Optional[str | Path] = None) -> Optional[NDArray[np.float64]]:
        """
        Run the O(1) Neural Equilibrium Surrogate.
        Requires NeuralEquilibriumKernel (JAX/NPZ weights).
        """
        if NeuralEquilibriumKernel is None:
            logger.warning("NeuralEquilibriumKernel not available (ImportError).")
            return None
            
        try:
            # Note: NeuralEquilibriumKernel needs a config for grid sizing
            # default to iter_config.json in root if not provided
            if config_path is None:
                config_path = Path(__file__).resolve().parents[3] / "iter_config.json"
            
            kernel = NeuralEquilibriumKernel(config_path)
            res = kernel.solve_equilibrium()
            return res.get("Psi")
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
            raise ValueError(
                f"j_phi must be a 2D array, received ndim={j_input.ndim}"
            )
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

    Looks for ``solver.cpp`` in the same directory as this module and
    invokes ``g++`` to produce a shared library.

    Returns
    -------
    str or None
        Path to the compiled library, or *None* on failure.
    """
    if os.environ.get("SCPN_ALLOW_NATIVE_BUILD") != "1":
        logger.warning(
            "Native build disabled. Set SCPN_ALLOW_NATIVE_BUILD=1 to enable."
        )
        return None

    logger.info("Compiling C++ solver kernel…")
    script_dir = Path(__file__).resolve().parent
    src = script_dir / "solver.cpp"
    out_dir = script_dir / "bin"
    out_dir.mkdir(exist_ok=True)

    if platform.system() == "Windows":
        out = out_dir / "scpn_solver.dll"
        cmd = ["g++", "-shared", "-o", str(out), str(src), "-O3", "-mavx2"]
    else:
        out = out_dir / "libscpn_solver.so"
        cmd = [
            "g++",
            "-shared",
            "-fPIC",
            "-o",
            str(out),
            str(src),
            "-O3",
            "-march=native",
        ]

    logger.info("Executing: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("Compilation failed: %s", exc)
        return None

    logger.info("Compilation succeeded: %s", out)
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
        print(f"Max Flux: {np.max(Psi)}")
