import ctypes
import numpy as np
import os
import time
import platform

class HPCBridge:
    """
    Interface between Python Logic and C++ Muscle.
    Loads the compiled shared library (.so/.dll) and dispatches computation.
    """
    def __init__(self, lib_path=None):
        self.lib = None
        self.solver_ptr = None
        self.loaded = False
        
        # Auto-detect library name if not provided
        if lib_path is None:
            lib_name = "scpn_solver.dll" if platform.system() == "Windows" else "libscpn_solver.so"
            # Look in current directory and hpc directory
            candidates = [
                f"./{lib_name}",
                os.path.join(os.path.dirname(__file__), lib_name),
                os.path.join(os.getcwd(), "03_CODE", "SCPN-Fusion-Core", lib_name)
            ]
            for c in candidates:
                if os.path.exists(c):
                    lib_path = c
                    break
            if lib_path is None:
                # Default to a relative path for message
                lib_path = f"./{lib_name}"

        self.lib_path = lib_path
        
        # Try load
        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_signatures()
            print(f"[HPC] Loaded Accelerator: {lib_path}")
            self.loaded = True
        except OSError:
            # Silent fail on init is okay, we check is_available() later
            pass
            # print(f"[HPC] WARNING: Accelerator binary not found at {lib_path}")

    def is_available(self):
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

    def initialize(self, nr, nz, r_range, z_range):
        if not self.loaded: return
        self.nr = nr
        self.nz = nz
        self.solver_ptr = self.lib.create_solver(nr, nz, r_range[0], r_range[1], z_range[0], z_range[1])

    def solve(self, j_phi, iterations=100):
        if not self.loaded: 
            # Fallback to Python logic or return None to signal caller
            return None
            
        psi_out = np.zeros_like(j_phi)
        
        # Call C++
        # t0 = time.time()
        self.lib.run_step(
            self.solver_ptr, 
            np.ascontiguousarray(j_phi), 
            psi_out, 
            j_phi.size, 
            iterations
        )
        # dt = time.time() - t0
        # print(f"[HPC] C++ Kernel finished in {dt*1000:.2f} ms ({iterations} iters)")
        
        return psi_out

# Compilation Helper (For user convenience)
def compile_cpp():
    print("--- COMPILING C++ KERNEL ---")
    
    # Locate source relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(script_dir, "solver.cpp")
    
    if platform.system() == "Windows":
        out = "scpn_solver.dll"
        cmd = f"g++ -shared -o {out} \"{src}\" -O3 -mavx2"
    else:
        out = "libscpn_solver.so"
        cmd = f"g++ -shared -fPIC -o {out} \"{src}\" -O3 -march=native"
        
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    
    if ret == 0:
        print("Compilation SUCCESS.")
        return out
    else:
        print("Compilation FAILED. Ensure g++ is installed and in PATH.")
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
