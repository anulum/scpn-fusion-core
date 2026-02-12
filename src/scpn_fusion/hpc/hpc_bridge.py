# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — HPC Bridge
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import ctypes
import logging
import numpy as np
import os
import platform
from typing import Optional

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
        bridge searches the current directory and its own package
        directory.
    """

    def __init__(self, lib_path: Optional[str] = None) -> None:
        self.lib = None
        self.solver_ptr = None
        self.loaded: bool = False

        if lib_path is None:
            lib_name = (
                "scpn_solver.dll"
                if platform.system() == "Windows"
                else "libscpn_solver.so"
            )
            candidates = [
                f"./{lib_name}",
                os.path.join(os.path.dirname(__file__), lib_name),
                os.path.join(
                    os.getcwd(), "03_CODE", "SCPN-Fusion-Core", lib_name
                ),
            ]
            for c in candidates:
                if os.path.exists(c):
                    lib_path = c
                    break
            if lib_path is None:
                lib_path = f"./{lib_name}"

        self.lib_path = lib_path

        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_signatures()
            logger.info("Loaded C++ accelerator: %s", lib_path)
            self.loaded = True
        except OSError:
            pass

    def is_available(self) -> bool:
        """Return *True* if the compiled solver library was loaded."""
        return self.loaded

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

    def initialize(
        self,
        nr: int,
        nz: int,
        r_range: tuple[float, float],
        z_range: tuple[float, float],
    ) -> None:
        """Create the C++ solver instance for the given grid dimensions."""
        if not self.loaded:
            return
        self.nr = nr
        self.nz = nz
        self.solver_ptr = self.lib.create_solver(
            nr, nz, r_range[0], r_range[1], z_range[0], z_range[1]
        )

    def solve(
        self,
        j_phi: NDArray[np.float64],
        iterations: int = 100,
    ) -> Optional[NDArray[np.float64]]:
        """Run the C++ solver for *iterations* sweeps.

        Returns *None* if the library is not loaded (caller should
        fall back to a Python solver).
        """
        if not self.loaded:
            return None

        psi_out = np.zeros_like(j_phi)
        self.lib.run_step(
            self.solver_ptr,
            np.ascontiguousarray(j_phi),
            psi_out,
            j_phi.size,
            iterations,
        )
        return psi_out

def compile_cpp() -> Optional[str]:
    """Compile the C++ solver from source.

    Looks for ``solver.cpp`` in the same directory as this module and
    invokes ``g++`` to produce a shared library.

    Returns
    -------
    str or None
        Path to the compiled library, or *None* on failure.
    """
    logger.info("Compiling C++ solver kernel…")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(script_dir, "solver.cpp")

    if platform.system() == "Windows":
        out = "scpn_solver.dll"
        cmd = f'g++ -shared -o {out} "{src}" -O3 -mavx2'
    else:
        out = "libscpn_solver.so"
        cmd = f'g++ -shared -fPIC -o {out} "{src}" -O3 -march=native'

    logger.info("Executing: %s", cmd)
    ret = os.system(cmd)

    if ret == 0:
        logger.info("Compilation succeeded: %s", out)
        return out

    logger.error("Compilation failed (exit code %d). Is g++ installed?", ret)
    return None

if __name__ == "__main__":
    # Test sequence
    lib_file = compile_cpp()
    
    if lib_file:
        bridge = HPCBridge(f"./{lib_file}")
        
        # Test Grid
        N = 100
        bridge.initialize(N, N, (2.0, 10.0), (-5.0, 5.0))
        
        # Dummy Current (Gaussian)
        J = np.random.rand(N, N)
        
        # Run
        Psi = bridge.solve(J, iterations=500)
        print(f"Max Flux: {np.max(Psi)}")
