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
import os
import logging
from collections import deque
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

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


def _require_monotonic_axis(name: str, values: np.ndarray, expected_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size != int(expected_len):
        raise ValueError(f"{name} must be 1-D with length {expected_len}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    delta = np.diff(arr)
    if delta.size == 0 or not np.all(delta > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _require_state_grid(
    name: str,
    values: np.ndarray,
    *,
    nz: int,
    nr: int,
    require_finite: bool,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    expected = (int(nz), int(nr))
    if arr.ndim != 2 or tuple(arr.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    if require_finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _rust_available():
    """Check if the Rust backend is loadable."""
    return _RUST_AVAILABLE


class RustAcceleratedKernel:
    """
    Drop-in wrapper around Rust PyFusionKernel that mirrors the Python
    FusionKernel attribute interface (.Psi, .R, .Z, .RR, .ZZ, .cfg, etc.).

    Delegates equilibrium solve to Rust for ~20x speedup while keeping
    all attribute accesses compatible with downstream code.
    """

    def __init__(self, config_path):
        self._config_path = str(config_path)
        self.state_sync_failures = 0
        self.last_state_sync_error: Optional[str] = None
        # Load via Rust (PyO3 expects str, not Path)
        self._rust = PyFusionKernel(self._config_path)

        # Also load JSON config for attribute access (bridges read .cfg directly)
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        # Mirror grid attributes
        nr, nz = self._rust.grid_shape()
        self.NR = int(nr)
        self.NZ = int(nz)
        if self.NR < 2 or self.NZ < 2:
            raise ValueError(f"Rust grid shape must be >= 2x2, got {(self.NR, self.NZ)}")
        self.R = _require_monotonic_axis("R", np.asarray(self._rust.get_r()), self.NR)
        self.Z = _require_monotonic_axis("Z", np.asarray(self._rust.get_z()), self.NZ)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        # Initialize and validate state from Rust arrays.
        self.Psi = np.zeros((self.NZ, self.NR), dtype=np.float64)
        self.J_phi = np.zeros((self.NZ, self.NR), dtype=np.float64)
        self.B_R = np.zeros((self.NZ, self.NR), dtype=np.float64)
        self.B_Z = np.zeros((self.NZ, self.NR), dtype=np.float64)
        self._sync_state_from_rust(context="init", require_finite=True)
        self.compute_b_field()

    def solve_equilibrium(self):
        """Solve Grad-Shafranov equilibrium via Rust backend."""
        result = self._rust.solve_equilibrium()

        # Sync arrays back to Python attributes
        self._sync_state_from_rust(context="solve_equilibrium", require_finite=True)

        # Compute B-field from Psi (matching Python FusionKernel.compute_b_field)
        self.compute_b_field()

        return result

    def compute_b_field(self):
        """Compute magnetic field components from Psi gradient."""
        if tuple(self.Psi.shape) != (self.NZ, self.NR):
            raise ValueError(
                f"Psi shape mismatch for B-field computation: expected {(self.NZ, self.NR)}, "
                f"got {self.Psi.shape}"
            )
        if not np.all(np.isfinite(self.Psi)):
            raise ValueError("Psi must contain finite values before B-field computation")
        dPsi_dR, dPsi_dZ = np.gradient(self.Psi, self.dR, self.dZ)
        R_safe = np.maximum(self.RR, 1e-6)
        self.B_R = -(1.0 / R_safe) * dPsi_dZ
        self.B_Z = (1.0 / R_safe) * dPsi_dR

    def _sync_state_from_rust(self, *, context: str, require_finite: bool) -> None:
        """Synchronize Psi/J_phi arrays from Rust and enforce shape/finite contracts."""
        try:
            psi = _require_state_grid(
                "Psi",
                np.asarray(self._rust.get_psi()),
                nz=self.NZ,
                nr=self.NR,
                require_finite=require_finite,
            )
            j_phi = _require_state_grid(
                "J_phi",
                np.asarray(self._rust.get_j_phi()),
                nz=self.NZ,
                nr=self.NR,
                require_finite=require_finite,
            )
        except ValueError as exc:
            self.state_sync_failures += 1
            self.last_state_sync_error = str(exc)
            logger.warning("Rust state sync failed during %s: %s", context, exc)
            raise RuntimeError(f"Rust state sync failed during {context}: {exc}") from exc
        self.Psi = psi
        self.J_phi = j_phi

    def find_x_point(self, Psi):
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
            return (self.R[ir], self.Z[iz]), Psi[iz, ir]
        else:
            return (0, 0), np.min(Psi)

    def calculate_thermodynamics(self, p_aux_mw):
        """Calculate thermodynamics via Rust backend."""
        return self._rust.calculate_thermodynamics(p_aux_mw)

    def calculate_vacuum_field(self):
        """Compute vacuum field with Python reference implementation."""
        from scpn_fusion.core.fusion_kernel import FusionKernel as _PyFusionKernel

        fk = _PyFusionKernel(self._config_path)
        return fk.calculate_vacuum_field()

    def set_solver_method(self, method: str) -> None:
        """Set inner linear solver: 'sor' or 'multigrid'."""
        self._rust.set_solver_method(method)

    def solver_method(self) -> str:
        """Get current solver method name."""
        return self._rust.solver_method()

    def save_results(self, filename="equilibrium_nonlinear.npz"):
        """Save current state to .npz file."""
        np.savez(filename, R=self.R, Z=self.Z, Psi=self.Psi, J_phi=self.J_phi)


if _RUST_AVAILABLE:
    FusionKernel = RustAcceleratedKernel
    RUST_BACKEND = True
else:
    from scpn_fusion.core.fusion_kernel import FusionKernel  # noqa: F811
    RUST_BACKEND = False


# Re-export Rust-only helpers (with compatibility shims where needed)
if _RUST_AVAILABLE:
    def rust_shafranov_bv(*args, **kwargs):
        """Compatibility wrapper for legacy config-path invocation.

        Supported call forms:
        - rust_shafranov_bv(r_geo, a_min, ip_ma) -> tuple[float, float, float]
        - rust_shafranov_bv(config_path) -> vacuum Psi array
        """
        if len(args) == 1 and not kwargs and isinstance(args[0], (str, os.PathLike)):
            from scpn_fusion.core.fusion_kernel import FusionKernel as _PyFusionKernel

            fk = _PyFusionKernel(str(args[0]))
            return fk.calculate_vacuum_field()
        return shafranov_bv(*args, **kwargs)

    rust_solve_coil_currents = solve_coil_currents
    rust_measure_magnetics = measure_magnetics

    def rust_simulate_tearing_mode(steps: int, seed: Optional[int] = None):
        """Rust tearing mode with optional deterministic seed compatibility."""
        if seed is None:
            return simulate_tearing_mode(int(steps))

        from scpn_fusion.control.disruption_predictor import (
            simulate_tearing_mode as _py_tearing,
        )

        rng = np.random.default_rng(seed=int(seed))
        return _py_tearing(steps=int(steps), rng=rng)
else:
    def rust_shafranov_bv(*args, **kwargs):
        raise ImportError("scpn_fusion_rs not installed. Run: maturin develop")

    def rust_solve_coil_currents(*args, **kwargs):
        raise ImportError("scpn_fusion_rs not installed. Run: maturin develop")

    def rust_measure_magnetics(*args, **kwargs):
        raise ImportError("scpn_fusion_rs not installed. Run: maturin develop")

    def rust_simulate_tearing_mode(*args, **kwargs):
        raise ImportError("scpn_fusion_rs not installed. Run: maturin develop")


class RustSnnPool:
    """Compatibility wrapper for Rust SpikingControllerPool.

    Uses the Rust implementation when available and falls back to a
    deterministic NumPy LIF population otherwise.

    Parameters
    ----------
    n_neurons : int
        Number of LIF neurons per sub-population (positive/negative).
    gain : float
        Output scaling factor.
    window_size : int
        Sliding window length for rate-code averaging.
    allow_numpy_fallback : bool
        When ``False``, raise :class:`ImportError` if Rust extension is
        unavailable.
    seed : int
        Seed used by deterministic NumPy fallback backend.
    """

    def __init__(
        self,
        n_neurons: int = 50,
        gain: float = 10.0,
        window_size: int = 20,
        *,
        allow_numpy_fallback: bool = True,
        seed: int = 42,
    ):
        self._backend = "rust"
        if _RUST_AVAILABLE:
            from scpn_fusion_rs import PySnnPool  # type: ignore[import-untyped]

            self._inner = PySnnPool(n_neurons, gain, window_size)
            return

        if not allow_numpy_fallback:
            raise ImportError(
                "scpn_fusion_rs not installed and allow_numpy_fallback=False."
            )
        self._backend = "numpy_fallback"
        self._inner = _NumpySnnPoolFallback(
            n_neurons=n_neurons,
            gain=gain,
            window_size=window_size,
            seed=seed,
        )

    def step(self, error: float) -> float:
        """Process *error* through SNN pool and return scalar control output."""
        return self._inner.step(error)

    @property
    def n_neurons(self) -> int:
        return self._inner.n_neurons

    @property
    def gain(self) -> float:
        return self._inner.gain

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self) -> str:
        return (
            f"RustSnnPool(n_neurons={self.n_neurons}, gain={self.gain}, "
            f"backend='{self.backend}')"
        )


class RustSnnController:
    """Compatibility wrapper for Rust NeuroCyberneticController.

    Uses the Rust implementation when available and falls back to paired
    deterministic NumPy LIF pools otherwise.

    Parameters
    ----------
    target_r : float
        Target major-radius position [m].
    target_z : float
        Target vertical position [m].
    allow_numpy_fallback : bool
        When ``False``, raise :class:`ImportError` if Rust extension is
        unavailable.
    seed : int
        Seed used by deterministic NumPy fallback backend.
    """

    def __init__(
        self,
        target_r: float = 6.2,
        target_z: float = 0.0,
        *,
        allow_numpy_fallback: bool = True,
        seed: int = 42,
    ):
        self._backend = "rust"
        if _RUST_AVAILABLE:
            from scpn_fusion_rs import PySnnController  # type: ignore[import-untyped]

            self._inner = PySnnController(target_r, target_z)
            return

        if not allow_numpy_fallback:
            raise ImportError(
                "scpn_fusion_rs not installed and allow_numpy_fallback=False."
            )
        self._backend = "numpy_fallback"
        self._inner = _NumpySnnControllerFallback(
            target_r=target_r,
            target_z=target_z,
            seed=seed,
        )

    def step(self, measured_r: float, measured_z: float) -> tuple[float, float]:
        """Process measured (R, Z) position and return (ctrl_R, ctrl_Z)."""
        return self._inner.step(measured_r, measured_z)

    @property
    def target_r(self) -> float:
        return self._inner.target_r

    @property
    def target_z(self) -> float:
        return self._inner.target_z

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self) -> str:
        return (
            f"RustSnnController(target_r={self.target_r}, target_z={self.target_z}, "
            f"backend='{self.backend}')"
        )


class _NumpySnnPoolFallback:
    """Deterministic local fallback matching the Rust SNN pool interface."""

    def __init__(
        self,
        n_neurons: int,
        gain: float,
        window_size: int,
        *,
        seed: int,
    ) -> None:
        self.n_neurons = int(n_neurons)
        self.gain = float(gain)
        self.window_size = int(window_size)
        if self.n_neurons < 1:
            raise ValueError("n_neurons must be >= 1.")
        if not np.isfinite(self.gain):
            raise ValueError("gain must be finite.")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1.")

        self._rng_pos = np.random.default_rng(int(seed))
        self._rng_neg = np.random.default_rng(int(seed) + 100003)
        self._v_pos = np.zeros(self.n_neurons, dtype=np.float64)
        self._v_neg = np.zeros(self.n_neurons, dtype=np.float64)
        self._history_pos: deque[int] = deque([0] * self.window_size, maxlen=self.window_size)
        self._history_neg: deque[int] = deque([0] * self.window_size, maxlen=self.window_size)
        self._alpha = 1.0e-3 / 15.0e-3
        self._noise_std = 0.02
        self._i_scale = 5.0
        self._i_bias = 0.1
        self._v_threshold = 0.35
        self._v_reset = 0.0

    def _step_pop(self, v: np.ndarray, rng: np.random.Generator, input_current: float) -> int:
        noise = rng.normal(0.0, self._noise_std, size=v.shape)
        v += self._alpha * (-v + float(input_current) + noise)
        fired = v >= self._v_threshold
        n_fired = int(np.count_nonzero(fired))
        if n_fired > 0:
            v[fired] = self._v_reset
        return n_fired

    def step(self, error_signal: float) -> float:
        err = float(error_signal)
        if not np.isfinite(err):
            raise ValueError("error_signal must be finite.")
        input_pos = max(0.0, err) * self._i_scale
        input_neg = max(0.0, -err) * self._i_scale

        spikes_pos = self._step_pop(self._v_pos, self._rng_pos, self._i_bias + input_pos)
        spikes_neg = self._step_pop(self._v_neg, self._rng_neg, self._i_bias + input_neg)
        self._history_pos.append(spikes_pos)
        self._history_neg.append(spikes_neg)

        rate_pos = float(sum(self._history_pos) / (self.window_size * self.n_neurons))
        rate_neg = float(sum(self._history_neg) / (self.window_size * self.n_neurons))
        return float((rate_pos - rate_neg) * self.gain)


class _NumpySnnControllerFallback:
    """Deterministic local fallback matching the Rust SNN controller interface."""

    def __init__(self, target_r: float, target_z: float, *, seed: int) -> None:
        self.target_r = float(target_r)
        self.target_z = float(target_z)
        if not np.isfinite(self.target_r) or not np.isfinite(self.target_z):
            raise ValueError("target_r and target_z must be finite.")
        self._pool_r = _NumpySnnPoolFallback(50, 10.0, 20, seed=int(seed) + 1)
        self._pool_z = _NumpySnnPoolFallback(50, 20.0, 20, seed=int(seed) + 2)

    def step(self, measured_r: float, measured_z: float) -> tuple[float, float]:
        mr = float(measured_r)
        mz = float(measured_z)
        if not np.isfinite(mr) or not np.isfinite(mz):
            raise ValueError("measured_r and measured_z must be finite.")
        err_r = self.target_r - mr
        err_z = self.target_z - mz
        return self._pool_r.step(err_r), self._pool_z.step(err_z)


def rust_multigrid_vcycle(
    source: np.ndarray,
    psi_bc: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    nr: int,
    nz: int,
    tol: float = 1e-6,
    max_cycles: int = 500,
) -> tuple[np.ndarray, float, int, bool]:
    """Call Rust multigrid V-cycle if available, else raise ImportError.

    Returns
    -------
    tuple of (psi, residual, n_cycles, converged)
    """
    if not _RUST_AVAILABLE:
        raise ImportError(
            "scpn_fusion_rs not installed — Rust multigrid unavailable. "
            "Use Python multigrid fallback in FusionKernel._multigrid_vcycle()."
        )
    # Delegate to Rust PyO3 binding (when available in fusion-python crate)
    try:
        from scpn_fusion_rs import multigrid_vcycle as _rust_mg  # type: ignore
        return _rust_mg(source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol, max_cycles)
    except ImportError:
        raise ImportError(
            "Rust multigrid_vcycle not exposed via PyO3. "
            "Use Python multigrid fallback."
        )
