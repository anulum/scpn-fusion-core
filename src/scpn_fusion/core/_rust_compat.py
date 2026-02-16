# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Rust Compat
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Backward compatibility layer: imports from Rust (scpn_fusion_rs) if available,
falls back to pure-Python implementations.

Usage:
    from scpn_fusion.core._rust_compat import FusionKernel, RUST_BACKEND
"""
import numpy as np
from typing import Any, Dict, Tuple, Optional, Union, TYPE_CHECKING, Protocol, runtime_checkable
from pathlib import Path

@runtime_checkable
class FusionKernelProtocol(Protocol):
    """Structural interface for FusionKernel implementations."""
    Psi: np.ndarray[Any, Any]
    J_phi: np.ndarray[Any, Any]
    R: np.ndarray[Any, Any]
    Z: np.ndarray[Any, Any]
    RR: np.ndarray[Any, Any]
    ZZ: np.ndarray[Any, Any]
    NR: int
    NZ: int
    dR: float
    dZ: float
    cfg: Dict[str, Any]
    external_profile_mode: bool = False

    def solve_equilibrium(self) -> Dict[str, Any]: ...
    def compute_b_field(self) -> None: ...
    def find_x_point(self, Psi: np.ndarray[Any, Any]) -> Tuple[Tuple[float, float], float]: ...
    def calculate_thermodynamics(self, p_aux_mw: float) -> Dict[str, Any]: ...
    def save_results(self, filename: Union[str, Path] = "equilibrium_nonlinear.npz") -> None: ...

# Type aliases for Mypy
if TYPE_CHECKING:
    from scpn_fusion.core.fusion_kernel import FusionKernel as _PyFKImpl
    # FusionKernel base class type for inheritance
    FusionKernelBase = _PyFKImpl
else:
    FusionKernelBase = object # type: ignore

try:
    from scpn_fusion_rs import (
        PyFusionKernel,
        PyEquilibriumResult,
        PyThermodynamicsResult,
        shafranov_bv,
        solve_coil_currents,
        measure_magnetics,
        simulate_tearing_mode,
    )
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def _rust_available() -> bool:
    """Check if the Rust backend is loadable."""
    return _RUST_AVAILABLE


class RustAcceleratedKernel:
    """
    Drop-in wrapper around Rust PyFusionKernel that mirrors the Python
    FusionKernel attribute interface (.Psi, .R, .Z, .RR, .ZZ, .cfg, etc.).

    Delegates equilibrium solve to Rust for ~20x speedup while keeping
    all attribute accesses compatible with downstream code.
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        # Load via Rust
        self._rust = PyFusionKernel(str(config_path))

        # Also load JSON config for attribute access (bridges read .cfg directly)
        import json
        with open(config_path, 'r') as f:
            self.cfg: Dict[str, Any] = json.load(f)

        # Mirror grid attributes
        nr, nz = self._rust.grid_shape()
        self.NR: int = nr
        self.NZ: int = nz
        self.R: np.ndarray[Any, Any] = np.asarray(self._rust.get_r())
        self.Z: np.ndarray[Any, Any] = np.asarray(self._rust.get_z())
        self.dR: float = float(self.R[1] - self.R[0])
        self.dZ: float = float(self.Z[1] - self.Z[0])
        self.RR: np.ndarray[Any, Any]
        self.ZZ: np.ndarray[Any, Any]
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        # Initialize Psi and J_phi from Rust state
        self.Psi: np.ndarray[Any, Any] = np.asarray(self._rust.get_psi())
        self.J_phi: np.ndarray[Any, Any] = np.asarray(self._rust.get_j_phi())

        # B-field placeholders (computed after solve)
        self.B_R: np.ndarray[Any, Any] = np.zeros((self.NZ, self.NR))
        self.B_Z: np.ndarray[Any, Any] = np.zeros((self.NZ, self.NR))
        self.external_profile_mode: bool = False

    def solve_equilibrium(self) -> Dict[str, Any]:
        """Solve Grad-Shafranov equilibrium via Rust backend."""
        result: Dict[str, Any] = self._rust.solve_equilibrium() # type: ignore

        # Sync arrays back to Python attributes
        self.Psi = np.asarray(self._rust.get_psi())
        self.J_phi = np.asarray(self._rust.get_j_phi())

        # Compute B-field from Psi (matching Python FusionKernel.compute_b_field)
        self.compute_b_field()

        return result

    def compute_b_field(self) -> None:
        """Compute magnetic field components from Psi gradient."""
        dPsi_dR, dPsi_dZ = np.gradient(self.Psi, self.dR, self.dZ)
        R_safe = np.maximum(self.RR, 1e-6)
        self.B_R = -(1.0 / R_safe) * dPsi_dZ
        self.B_Z = (1.0 / R_safe) * dPsi_dR

    def find_x_point(self, Psi: np.ndarray[Any, Any]) -> Tuple[Tuple[float, float], float]:
        """
        Locate the null point (B=0) using local minimization.
        Matches Python FusionKernel.find_x_point() interface.
        """
        dPsi_dR, dPsi_dZ = np.gradient(Psi, self.dR, self.dZ)
        B_mag = np.sqrt(dPsi_dR**2 + dPsi_dZ**2)

        mask_divertor = self.ZZ < (self.cfg['dimensions']['Z_min'] * 0.5)

        if np.any(mask_divertor):
            masked_B = np.where(mask_divertor, B_mag, 1e9)
            idx_min = np.argmin(masked_B)
            iz, ir = np.unravel_index(idx_min, Psi.shape)
            return (float(self.R[ir]), float(self.Z[iz])), float(Psi[iz, ir])
        else:
            return (0.0, 0.0), float(np.min(Psi))

    def calculate_thermodynamics(self, p_aux_mw: float) -> Dict[str, Any]:
        """Calculate thermodynamics via Rust backend."""
        return self._rust.calculate_thermodynamics(p_aux_mw) # type: ignore

    def set_solver_method(self, method: str) -> None:
        """Set inner linear solver: 'sor' or 'multigrid'."""
        self._rust.set_solver_method(method) # type: ignore

    def solver_method(self) -> str:
        """Get current solver method name."""
        return str(self._rust.solver_method()) # type: ignore

    def save_results(self, filename: Union[str, Path] = "equilibrium_nonlinear.npz") -> None:
        """Save current state to .npz file."""
        np.savez(str(filename), R=self.R, Z=self.Z, Psi=self.Psi, J_phi=self.J_phi)


# ─── Public API ─────────────────────────────────────────────────────

if _RUST_AVAILABLE:
    FusionKernel: Any = RustAcceleratedKernel
    RUST_BACKEND = True
else:
    from scpn_fusion.core.fusion_kernel import FusionKernel as _PyFK  # type: ignore[attr-defined]
    FusionKernel: Any = _PyFK  # type: ignore[no-redef]
    RUST_BACKEND = False


# Re-export Rust-only helpers (with fallback stubs)
if _RUST_AVAILABLE:
    # These are pure Rust functions with no Python equivalent
    rust_shafranov_bv = shafranov_bv
    rust_solve_coil_currents = solve_coil_currents
    rust_measure_magnetics = measure_magnetics
    rust_simulate_tearing_mode = simulate_tearing_mode
else:
    def rust_shafranov_bv(*args, **kwargs):
        raise ImportError("scpn_fusion_rs not installed. Run: maturin develop")

    def rust_solve_coil_currents(*args, **kwargs):
        raise ImportError("scpn_fusion_rs not installed. Run: maturin develop")

    def rust_measure_magnetics(*args, **kwargs):
        raise ImportError("scpn_fusion_rs not installed. Run: maturin develop")

    def rust_simulate_tearing_mode(*args, **kwargs):
        raise ImportError("scpn_fusion_rs not installed. Run: maturin develop")
