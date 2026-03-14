# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Free-Boundary Tracking Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for free-boundary target tracking control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.control.free_boundary_tracking import (
    FreeBoundaryTrackingController,
    run_free_boundary_tracking,
)
from scpn_fusion.core.fusion_kernel import CoilSet, FusionKernel


class _DummyFreeBoundaryKernel:
    """Linear free-boundary plant with explicit boundary/X-point/divertor observables."""

    def __init__(self, _config_file: str) -> None:
        self._boundary_points = np.array(
            [
                [3.4, -0.1],
                [4.0, 0.3],
                [4.6, -0.4],
            ],
            dtype=np.float64,
        )
        self._divertor_points = np.array(
            [
                [3.1, -2.6],
                [4.9, -2.6],
            ],
            dtype=np.float64,
        )
        self._x_target = np.array([4.2, -1.4], dtype=np.float64)
        self._target_vector = np.array(
            [0.12, 0.18, 0.10, 4.2, -1.4, 0.15, 0.15, 0.15],
            dtype=np.float64,
        )
        self._response_matrix = np.array(
            [
                [0.60, -0.20, 0.15, 0.05],
                [0.10, 0.55, -0.15, 0.05],
                [-0.20, 0.10, 0.50, 0.10],
                [0.12, -0.08, 0.03, 0.01],
                [-0.04, 0.02, 0.11, -0.09],
                [0.22, 0.10, -0.05, 0.04],
                [0.14, -0.12, 0.18, 0.05],
                [0.11, 0.09, -0.04, 0.16],
            ],
            dtype=np.float64,
        )
        self._bias = self._response_matrix @ np.array([0.45, -0.35, 0.30, -0.20], dtype=np.float64)
        self._drift_vector = self._response_matrix @ np.array(
            [0.10, -0.06, 0.08, -0.04], dtype=np.float64
        )
        self.cfg = {
            "physics": {"drift_scale": 0.0},
            "coils": [
                {"name": "PF1", "current": 0.0},
                {"name": "PF2", "current": 0.0},
                {"name": "PF3", "current": 0.0},
                {"name": "PF4", "current": 0.0},
            ],
            "free_boundary": {
                "objective_tolerances": {
                    "shape_rms": 0.025,
                    "x_point_position": 0.08,
                    "x_point_flux": 0.03,
                    "divertor_rms": 0.025,
                }
            },
        }
        self.R = np.linspace(3.0, 5.2, 8)
        self.Z = np.linspace(-3.0, 1.0, 8)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._state = self._target_vector + self._bias
        self.solve()

    def build_coilset_from_config(self) -> CoilSet:
        return CoilSet(
            positions=[(3.0, 2.2), (3.6, -2.1), (4.4, 2.0), (5.0, -2.2)],
            currents=np.zeros(4, dtype=np.float64),
            turns=[12, 12, 12, 12],
            current_limits=np.full(4, 3.0, dtype=np.float64),
            target_flux_points=self._boundary_points.copy(),
            target_flux_values=self._target_vector[:3].copy(),
            x_point_target=self._x_target.copy(),
            x_point_flux_target=float(self._target_vector[5]),
            divertor_strike_points=self._divertor_points.copy(),
            divertor_flux_values=self._target_vector[6:].copy(),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        drift = float(self.cfg.get("physics", {}).get("drift_scale", 0.0))
        disturbance = drift * self._drift_vector
        self._state = (
            self._target_vector + self._bias + disturbance + self._response_matrix @ currents
        )
        for idx, current in enumerate(currents):
            self.cfg["coils"][idx]["current"] = float(current)
        self.Psi.fill(0.0)
        return {
            "boundary_variant": (
                "free_boundary" if boundary_variant is None else str(boundary_variant)
            ),
            "converged": True,
            "outer_iterations": 1,
            "final_diff": float(np.linalg.norm(self._response_matrix @ currents)),
        }

    def _sample_flux_at_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:3].copy()
        if pts.shape == self._divertor_points.shape and np.allclose(pts, self._divertor_points):
            return self._state[6:].copy()
        raise ValueError("Unexpected probe points for dummy free-boundary kernel.")

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (float(self._state[3]), float(self._state[4])), float(self._state[5])

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[5])
        return float(np.mean(self._state[:3]))


class _NoTargetKernel(_DummyFreeBoundaryKernel):
    def build_coilset_from_config(self) -> CoilSet:
        coils = super().build_coilset_from_config()
        coils.target_flux_values = None
        coils.x_point_target = None
        coils.x_point_flux_target = None
        coils.divertor_flux_values = None
        return coils


class _FallbackKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "fallback_currents": [0.20, -0.20, 0.10, -0.10],
        }


class _ObserverKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "observer_gain": 0.45,
            "observer_max_abs": 0.35,
        }


class _MeasurementDistortionKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_bias": {
                "shape_flux": [0.04, -0.02, 0.03],
                "x_point_position": [0.06, -0.05],
                "x_point_flux": 0.025,
                "divertor_flux": [0.03, -0.02],
            },
            "measurement_drift_per_step": {
                "shape_flux": [0.006, -0.003, 0.004],
                "x_point_position": [0.01, -0.008],
                "x_point_flux": 0.004,
                "divertor_flux": [0.005, -0.003],
            },
        }


class _MeasurementCorrectedKernel(_MeasurementDistortionKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        tracking_cfg = self.cfg["free_boundary_tracking"]
        tracking_cfg["measurement_correction_bias"] = {
            "shape_flux": [0.04, -0.02, 0.03],
            "x_point_position": [0.06, -0.05],
            "x_point_flux": 0.025,
            "divertor_flux": [0.03, -0.02],
        }
        tracking_cfg["measurement_correction_drift_per_step"] = {
            "shape_flux": [0.006, -0.003, 0.004],
            "x_point_position": [0.01, -0.008],
            "x_point_flux": 0.004,
            "divertor_flux": [0.005, -0.003],
        }


class _MeasurementLatencyKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_latency_steps": 2,
        }


class _MeasurementLatencyCompensatedKernel(_MeasurementLatencyKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        tracking_cfg = self.cfg["free_boundary_tracking"]
        tracking_cfg["latency_compensation_gain"] = 1.0
        tracking_cfg["latency_rate_max_abs"] = 0.5


class _InvalidMeasurementKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_bias": {
                "x_point_position": [0.1, 0.2, 0.3],
            }
        }


class _PriorityConflictKernel:
    """Single-coil plant where shape and X-point flux objectives conflict."""

    def __init__(self, _config_file: str) -> None:
        self._boundary_points = np.array([[3.8, 0.0]], dtype=np.float64)
        self._x_target = np.array([4.2, -1.4], dtype=np.float64)
        self._target_vector = np.array([0.0, 4.2, -1.4, 0.0], dtype=np.float64)
        self._response_matrix = np.array([[1.0], [0.0], [0.0], [-1.0]], dtype=np.float64)
        self._bias = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.cfg = {
            "coils": [{"name": "PF1", "current": 0.0}],
            "free_boundary": {
                "objective_tolerances": {
                    "shape_rms": 2.0,
                    "x_point_flux": 0.05,
                }
            },
        }
        self.R = np.linspace(3.0, 5.0, 6)
        self.Z = np.linspace(-2.0, 1.0, 6)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._state = self._target_vector + self._bias
        self.solve()

    def build_coilset_from_config(self) -> CoilSet:
        return CoilSet(
            positions=[(3.2, 2.0)],
            currents=np.zeros(1, dtype=np.float64),
            turns=[12],
            current_limits=np.array([3.0], dtype=np.float64),
            target_flux_points=self._boundary_points.copy(),
            target_flux_values=self._target_vector[:1].copy(),
            x_point_target=self._x_target.copy(),
            x_point_flux_target=float(self._target_vector[3]),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        del max_outer_iter, tol, optimize_shape, tikhonov_alpha
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        self._state = self._target_vector + self._bias + self._response_matrix @ currents
        self.Psi.fill(0.0)
        return {
            "boundary_variant": (
                "free_boundary" if boundary_variant is None else str(boundary_variant)
            ),
            "converged": True,
            "outer_iterations": 1,
            "final_diff": float(np.linalg.norm(self._response_matrix @ currents)),
        }

    def _sample_flux_at_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:1].copy()
        raise ValueError("Unexpected probe points for priority-conflict kernel.")

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (float(self._x_target[0]), float(self._x_target[1])), float(self._state[3])

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[3])
        return float(self._state[0])


class _ProtectedObjectiveKernel:
    """Plant where shape improvement would violate an already-met X-point flux target."""

    def __init__(self, _config_file: str) -> None:
        self._boundary_points = np.array([[3.8, 0.0]], dtype=np.float64)
        self._x_target = np.array([4.2, -1.4], dtype=np.float64)
        self._target_vector = np.array([0.0, 4.2, -1.4, 0.0], dtype=np.float64)
        self._response_matrix = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=np.float64)
        self._bias = np.array([10.0, 0.0, 0.0, 0.01], dtype=np.float64)
        self.cfg = {
            "coils": [{"name": "PF1", "current": 0.0}],
            "free_boundary": {
                "objective_tolerances": {
                    "shape_rms": 0.05,
                    "x_point_flux": 0.05,
                }
            },
        }
        self.R = np.linspace(3.0, 5.0, 6)
        self.Z = np.linspace(-2.0, 1.0, 6)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._state = self._target_vector + self._bias
        self.solve()

    def build_coilset_from_config(self) -> CoilSet:
        return CoilSet(
            positions=[(3.2, 2.0)],
            currents=np.zeros(1, dtype=np.float64),
            turns=[12],
            current_limits=np.array([6.0], dtype=np.float64),
            target_flux_points=self._boundary_points.copy(),
            target_flux_values=self._target_vector[:1].copy(),
            x_point_target=self._x_target.copy(),
            x_point_flux_target=float(self._target_vector[3]),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        del max_outer_iter, tol, optimize_shape, tikhonov_alpha
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        self._state = self._target_vector + self._bias + self._response_matrix @ currents
        self.Psi.fill(0.0)
        return {
            "boundary_variant": (
                "free_boundary" if boundary_variant is None else str(boundary_variant)
            ),
            "converged": True,
            "outer_iterations": 1,
            "final_diff": float(np.linalg.norm(self._response_matrix @ currents)),
        }

    def _sample_flux_at_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:1].copy()
        raise ValueError("Unexpected probe points for protected-objective kernel.")

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (float(self._x_target[0]), float(self._x_target[1])), float(self._state[3])

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[3])
        return float(self._state[0])


class _ZeroAuthorityKernel:
    """Plant with no coil authority over any tracked objective."""

    def __init__(self, _config_file: str) -> None:
        self._boundary_points = np.array([[3.8, 0.0]], dtype=np.float64)
        self._x_target = np.array([4.2, -1.4], dtype=np.float64)
        self._target_vector = np.array([0.0, 4.2, -1.4, 0.0], dtype=np.float64)
        self._response_matrix = np.zeros((4, 1), dtype=np.float64)
        self._bias = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float64)
        self.cfg = {
            "coils": [{"name": "PF1", "current": 0.0}],
            "free_boundary": {
                "objective_tolerances": {
                    "shape_rms": 0.1,
                    "x_point_flux": 0.1,
                }
            },
        }
        self.R = np.linspace(3.0, 5.0, 6)
        self.Z = np.linspace(-2.0, 1.0, 6)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._state = self._target_vector + self._bias
        self.solve()

    def build_coilset_from_config(self) -> CoilSet:
        return CoilSet(
            positions=[(3.2, 2.0)],
            currents=np.zeros(1, dtype=np.float64),
            turns=[12],
            current_limits=np.array([3.0], dtype=np.float64),
            target_flux_points=self._boundary_points.copy(),
            target_flux_values=self._target_vector[:1].copy(),
            x_point_target=self._x_target.copy(),
            x_point_flux_target=float(self._target_vector[3]),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        del max_outer_iter, tol, optimize_shape, tikhonov_alpha
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        self._state = self._target_vector + self._bias + self._response_matrix @ currents
        self.Psi.fill(0.0)
        return {
            "boundary_variant": (
                "free_boundary" if boundary_variant is None else str(boundary_variant)
            ),
            "converged": True,
            "outer_iterations": 1,
            "final_diff": float(np.linalg.norm(self._response_matrix @ currents)),
        }

    def _sample_flux_at_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:1].copy()
        raise ValueError("Unexpected probe points for zero-authority kernel.")

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (float(self._x_target[0]), float(self._x_target[1])), float(self._state[3])

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[3])
        return float(self._state[0])


class _WithinToleranceKernel:
    """Plant with small residuals already inside tolerance deadbands."""

    def __init__(self, _config_file: str) -> None:
        self._boundary_points = np.array([[3.8, 0.0]], dtype=np.float64)
        self._x_target = np.array([4.2, -1.4], dtype=np.float64)
        self._target_vector = np.array([0.0, 4.2, -1.4, 0.0], dtype=np.float64)
        self._response_matrix = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=np.float64)
        self._bias = np.array([0.03, 0.0, 0.0, 0.02], dtype=np.float64)
        self.cfg = {
            "coils": [{"name": "PF1", "current": 0.0}],
            "free_boundary": {
                "objective_tolerances": {
                    "shape_rms": 0.05,
                    "x_point_flux": 0.05,
                }
            },
        }
        self.R = np.linspace(3.0, 5.0, 6)
        self.Z = np.linspace(-2.0, 1.0, 6)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._state = self._target_vector + self._bias
        self.solve()

    def build_coilset_from_config(self) -> CoilSet:
        return CoilSet(
            positions=[(3.2, 2.0)],
            currents=np.zeros(1, dtype=np.float64),
            turns=[12],
            current_limits=np.array([3.0], dtype=np.float64),
            target_flux_points=self._boundary_points.copy(),
            target_flux_values=self._target_vector[:1].copy(),
            x_point_target=self._x_target.copy(),
            x_point_flux_target=float(self._target_vector[3]),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        del max_outer_iter, tol, optimize_shape, tikhonov_alpha
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        self._state = self._target_vector + self._bias + self._response_matrix @ currents
        self.Psi.fill(0.0)
        return {
            "boundary_variant": (
                "free_boundary" if boundary_variant is None else str(boundary_variant)
            ),
            "converged": True,
            "outer_iterations": 1,
            "final_diff": float(np.linalg.norm(self._response_matrix @ currents)),
        }

    def _sample_flux_at_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:1].copy()
        raise ValueError("Unexpected probe points for within-tolerance kernel.")

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (float(self._x_target[0]), float(self._x_target[1])), float(self._state[3])

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[3])
        return float(self._state[0])


class _HeadroomAllocationKernel:
    """Plant where two coils are equally effective but one starts near its current limit."""

    def __init__(self, _config_file: str) -> None:
        self._boundary_points = np.array([[3.8, 0.0]], dtype=np.float64)
        self._target_vector = np.array([0.0], dtype=np.float64)
        self._nominal_currents = np.array([9.8, 0.0], dtype=np.float64)
        self._response_matrix = np.array([[1.0, 1.0]], dtype=np.float64)
        self._bias = np.array([0.4], dtype=np.float64)
        self.cfg = {
            "coils": [
                {"name": "PF1", "current": float(self._nominal_currents[0])},
                {"name": "PF2", "current": float(self._nominal_currents[1])},
            ],
        }
        self.R = np.linspace(3.0, 5.0, 6)
        self.Z = np.linspace(-2.0, 1.0, 6)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._state = self._target_vector + self._bias
        self.solve()

    def build_coilset_from_config(self) -> CoilSet:
        return CoilSet(
            positions=[(3.2, 2.0), (4.8, -2.0)],
            currents=self._nominal_currents.copy(),
            turns=[12, 12],
            current_limits=np.array([10.0, 10.0], dtype=np.float64),
            target_flux_points=self._boundary_points.copy(),
            target_flux_values=self._target_vector.copy(),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        del boundary_variant, max_outer_iter, tol, optimize_shape, tikhonov_alpha
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        delta_currents = currents - self._nominal_currents
        self._state = self._target_vector + self._bias + self._response_matrix @ delta_currents
        self.Psi.fill(0.0)
        return {
            "boundary_variant": "free_boundary",
            "converged": True,
            "outer_iterations": 1,
            "final_diff": float(np.linalg.norm(self._response_matrix @ delta_currents)),
        }

    def _sample_flux_at_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state.copy()
        raise ValueError("Unexpected probe points for headroom-allocation kernel.")

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (4.2, -1.4), 0.0

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        del r_pt, z_pt
        return float(self._state[0])


def _write_real_kernel_tracking_config(path: Path) -> Path:
    cfg = {
        "reactor_name": "Real-Free-Boundary-Tracking-Test",
        "grid_resolution": [12, 12],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": {
            "max_iterations": 10,
            "convergence_threshold": 1e-3,
            "relaxation_factor": 0.15,
            "solver_method": "sor",
            "boundary_variant": "free_boundary",
        },
        "free_boundary": {
            "current_limits": [5.0e4, 5.0e4],
            "target_flux_points": [[3.5, 0.0], [4.0, 0.5]],
            "objective_tolerances": {"shape_rms": 0.25, "shape_max_abs": 0.35},
        },
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")

    kernel = FusionKernel(path)
    coils = kernel.build_coilset_from_config()
    kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=1e-2,
        optimize_shape=False,
    )
    flux_targets = kernel._sample_flux_at_points(coils.target_flux_points)
    cfg["free_boundary"]["target_flux_values"] = [float(v) for v in flux_targets]
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_run_free_boundary_tracking_returns_bounded_converged_summary() -> None:
    summary = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=8,
        gain=0.9,
        verbose=False,
        kernel_factory=_DummyFreeBoundaryKernel,
        stop_on_convergence=False,
    )
    for key in (
        "config_path",
        "steps",
        "runtime_seconds",
        "boundary_variant",
        "final_tracking_error_norm",
        "mean_tracking_error_norm",
        "final_control_error_norm",
        "mean_control_error_norm",
        "max_abs_delta_i",
        "max_abs_coil_current",
        "objective_convergence_active",
        "objective_converged",
        "objective_checks",
    ):
        assert key in summary
    assert summary["config_path"] == "dummy.json"
    assert summary["boundary_variant"] == "free_boundary"
    assert summary["steps"] == 8
    assert np.isfinite(summary["runtime_seconds"])
    assert summary["max_abs_coil_current"] <= 3.0 + 1e-9
    assert summary["objective_convergence_active"] is True
    assert summary["objective_converged"] is True
    assert summary["final_tracking_error_norm"] < 0.15


def test_free_boundary_tracking_is_deterministic_for_fixed_inputs() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_steps=6,
        gain=0.85,
        verbose=False,
        kernel_factory=_DummyFreeBoundaryKernel,
        stop_on_convergence=False,
    )
    a = run_free_boundary_tracking(**kwargs)
    b = run_free_boundary_tracking(**kwargs)
    for key in (
        "final_tracking_error_norm",
        "mean_tracking_error_norm",
        "max_abs_delta_i",
        "max_abs_coil_current",
        "shape_rms",
        "x_point_position_error",
        "x_point_flux_error",
        "divertor_rms",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_controller_reduces_error_under_disturbance_callback() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
        response_refresh_steps=1,
    )
    controller._solve_free_boundary_state()
    initial_metrics = controller.evaluate_objectives(controller._observe_objectives())

    def disturbance(kernel: _DummyFreeBoundaryKernel, _coils: CoilSet, step: int) -> None:
        kernel.cfg.setdefault("physics", {})["drift_scale"] = 0.8 if step < 2 else 0.2

    summary = controller.run_tracking_shot(
        shot_steps=7,
        gain=0.9,
        disturbance_callback=disturbance,
        stop_on_convergence=False,
    )

    assert summary["final_tracking_error_norm"] < initial_metrics["tracking_error_norm"]
    assert summary["objective_converged"] is True
    assert summary["max_abs_coil_current"] <= 3.0 + 1e-9


def test_controller_rejects_invalid_runtime_inputs() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
    )
    with pytest.raises(ValueError, match="shot_steps"):
        controller.run_tracking_shot(shot_steps=0)
    with pytest.raises(ValueError, match="gain"):
        controller.run_tracking_shot(shot_steps=2, gain=0.0)
    with pytest.raises(ValueError, match="perturbation"):
        controller.identify_response_matrix(perturbation=0.0)


def test_controller_rejects_missing_explicit_targets() -> None:
    with pytest.raises(ValueError, match="explicit target values"):
        FreeBoundaryTrackingController(
            "dummy.json",
            kernel_factory=_NoTargetKernel,
            verbose=False,
        )


def test_controller_backtracks_aggressive_gain() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
    )

    summary = controller.run_tracking_shot(
        shot_steps=3,
        gain=12.0,
        stop_on_convergence=False,
    )

    assert summary["mean_accepted_gain"] > 0.0
    assert summary["min_accepted_gain"] > 0.0
    assert summary["mean_accepted_gain"] < 12.0
    assert max(controller.history["accepted_gain"]) < 12.0
    assert summary["final_tracking_error_norm"] < 0.25


def test_controller_enforces_coil_slew_limits() -> None:
    summary = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=1,
        gain=8.0,
        verbose=False,
        kernel_factory=_DummyFreeBoundaryKernel,
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=0.5,
        stop_on_convergence=False,
    )

    assert summary["max_abs_coil_current"] <= 0.05 + 1e-9
    assert summary["max_abs_actuator_lag"] > 0.01
    assert summary["mean_abs_actuator_lag"] > 0.01


def test_controller_supervisor_rejects_and_holds_unsafe_updates() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=0.5,
        supervisor_limits={"max_abs_actuator_lag": 0.0},
        hold_steps_after_reject=2,
    )
    controller._solve_free_boundary_state()
    initial_metrics = controller.evaluate_objectives(controller._observe_objectives())

    summary = controller.run_tracking_shot(
        shot_steps=4,
        gain=8.0,
        stop_on_convergence=False,
    )

    assert summary["supervisor_active"] is True
    assert summary["supervisor_intervention_count"] >= 3
    assert summary["hold_steps_after_reject"] == 2
    assert controller.history["supervisor_intervened"][0] is True
    assert controller.history["supervisor_hold_steps_remaining"][0] == 2
    assert controller.history["supervisor_hold_steps_remaining"][1] == 1
    assert controller.history["supervisor_hold_steps_remaining"][2] == 0
    assert summary["final_tracking_error_norm"] == pytest.approx(
        initial_metrics["tracking_error_norm"]
    )
    assert summary["max_abs_coil_current"] == pytest.approx(0.0)


def test_controller_uses_safe_current_fallback_when_configured() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_FallbackKernel,
        verbose=False,
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=0.5,
        supervisor_limits={"max_abs_actuator_lag": 0.0},
        hold_steps_after_reject=2,
    )

    summary = controller.run_tracking_shot(
        shot_steps=4,
        gain=8.0,
        stop_on_convergence=False,
    )

    assert summary["fallback_configured"] is True
    assert summary["fallback_active_steps"] >= 4
    assert np.allclose(
        controller.coils.currents,
        np.array([0.20, -0.20, 0.10, -0.10], dtype=np.float64),
        atol=5.0e-2,
    )
    assert controller.history["fallback_active"] == [True, True, True, True]
    assert summary["max_abs_coil_current"] > 0.0


def test_objective_observer_reduces_persistent_disturbance_error() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_steps=4,
        gain=0.18,
        verbose=False,
        stop_on_convergence=False,
    )

    def disturbance(kernel: _DummyFreeBoundaryKernel, _coils: CoilSet, _step: int) -> None:
        kernel.cfg.setdefault("physics", {})["drift_scale"] = 1.0

    baseline = run_free_boundary_tracking(
        kernel_factory=_DummyFreeBoundaryKernel,
        disturbance_callback=disturbance,
        **kwargs,
    )
    observed = run_free_boundary_tracking(
        kernel_factory=_ObserverKernel,
        disturbance_callback=disturbance,
        **kwargs,
    )

    assert baseline["observer_enabled"] is False
    assert observed["observer_enabled"] is True
    assert observed["max_abs_objective_bias_estimate"] > 0.0
    assert observed["final_tracking_error_norm"] < baseline["final_tracking_error_norm"]


def test_controller_reports_hidden_true_metrics_under_measurement_distortion() -> None:
    summary = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=4,
        gain=0.18,
        verbose=False,
        kernel_factory=_MeasurementDistortionKernel,
        stop_on_convergence=False,
    )

    assert summary["measurement_distortion_enabled"] is True
    assert summary["measurement_compensation_enabled"] is False
    assert summary["max_abs_measurement_offset"] > 0.0
    assert summary["mean_abs_measurement_offset"] > 0.0
    assert (
        abs(summary["final_tracking_error_norm"] - summary["final_true_tracking_error_norm"])
        > 1.0e-4
    )
    for key in (
        "final_true_tracking_error_norm",
        "mean_true_tracking_error_norm",
        "true_shape_rms",
        "true_x_point_position_error",
        "true_x_point_flux_error",
        "true_divertor_rms",
    ):
        assert key in summary


def test_measurement_compensation_restores_true_tracking_baseline() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_steps=4,
        gain=0.18,
        verbose=False,
        stop_on_convergence=False,
    )
    baseline = run_free_boundary_tracking(
        kernel_factory=_DummyFreeBoundaryKernel,
        **kwargs,
    )
    distorted = run_free_boundary_tracking(
        kernel_factory=_MeasurementDistortionKernel,
        **kwargs,
    )
    corrected = run_free_boundary_tracking(
        kernel_factory=_MeasurementCorrectedKernel,
        **kwargs,
    )

    assert distorted["measurement_compensation_enabled"] is False
    assert distorted["final_true_tracking_error_norm"] != pytest.approx(
        baseline["final_true_tracking_error_norm"],
        rel=0.0,
        abs=1.0e-4,
    )
    assert distorted["final_tracking_error_norm"] != pytest.approx(
        distorted["final_true_tracking_error_norm"],
        rel=0.0,
        abs=1.0e-4,
    )
    assert corrected["measurement_compensation_enabled"] is True
    for key in (
        "final_tracking_error_norm",
        "mean_tracking_error_norm",
        "final_true_tracking_error_norm",
        "mean_true_tracking_error_norm",
        "true_shape_rms",
        "true_x_point_position_error",
        "true_x_point_flux_error",
        "true_divertor_rms",
    ):
        assert corrected[key] == pytest.approx(baseline[key], rel=0.0, abs=1.0e-12)


def test_latency_compensation_reduces_delayed_observation_error() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_steps=4,
        gain=0.18,
        verbose=False,
        stop_on_convergence=False,
    )

    def disturbance(kernel: _DummyFreeBoundaryKernel, _coils: CoilSet, step: int) -> None:
        drift_schedule = (0.0, 1.0, 0.45, 0.10)
        kernel.cfg.setdefault("physics", {})["drift_scale"] = drift_schedule[
            min(step, len(drift_schedule) - 1)
        ]

    delayed = run_free_boundary_tracking(
        kernel_factory=_MeasurementLatencyKernel,
        disturbance_callback=disturbance,
        **kwargs,
    )
    compensated = run_free_boundary_tracking(
        kernel_factory=_MeasurementLatencyCompensatedKernel,
        disturbance_callback=disturbance,
        **kwargs,
    )

    assert delayed["measurement_latency_enabled"] is True
    assert delayed["latency_compensation_enabled"] is False
    assert delayed["measurement_latency_steps"] == 2
    assert delayed["max_delayed_observation_error_norm"] > 0.0
    assert delayed["max_estimated_observation_error_norm"] == pytest.approx(
        delayed["max_delayed_observation_error_norm"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert compensated["measurement_latency_enabled"] is True
    assert compensated["latency_compensation_enabled"] is True
    assert compensated["max_abs_objective_rate_estimate"] > 0.0
    assert (
        compensated["mean_estimated_observation_error_norm"]
        < delayed["max_delayed_observation_error_norm"]
    )
    assert (
        compensated["max_estimated_observation_error_norm"]
        <= delayed["max_delayed_observation_error_norm"] + 0.02
    )
    assert compensated["final_true_tracking_error_norm"] < delayed["final_true_tracking_error_norm"]


def test_controller_rejects_invalid_measurement_bias_shape() -> None:
    with pytest.raises(
        ValueError, match="measurement_bias.x_point_position must be a scalar or contain exactly 2"
    ):
        FreeBoundaryTrackingController(
            "dummy.json",
            kernel_factory=_InvalidMeasurementKernel,
            verbose=False,
        )


def test_controller_prioritizes_tighter_x_point_flux_tolerance_under_conflict() -> None:
    weighted = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=1,
        gain=1.0,
        verbose=False,
        kernel_factory=_PriorityConflictKernel,
        stop_on_convergence=False,
    )
    flat = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=1,
        gain=1.0,
        verbose=False,
        kernel_factory=_PriorityConflictKernel,
        objective_tolerances={"shape_rms": 2.0, "x_point_flux": 1.0},
        stop_on_convergence=False,
    )

    assert weighted["x_point_flux_error"] < flat["x_point_flux_error"]
    assert weighted["shape_rms"] > flat["shape_rms"]
    assert weighted["max_abs_delta_i"] > flat["max_abs_delta_i"]


def test_controller_does_not_sacrifice_already_met_tolerance() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_ProtectedObjectiveKernel,
        verbose=False,
    )
    controller._solve_free_boundary_state()
    initial_metrics = controller.evaluate_objectives(controller._observe_objectives())

    assert initial_metrics["shape_rms"] == pytest.approx(10.0)
    assert initial_metrics["x_point_flux_error"] == pytest.approx(0.01)

    summary = controller.run_tracking_shot(
        shot_steps=1,
        gain=1.0,
        stop_on_convergence=False,
    )

    assert summary["shape_rms"] == pytest.approx(initial_metrics["shape_rms"])
    assert summary["x_point_flux_error"] == pytest.approx(initial_metrics["x_point_flux_error"])
    assert summary["tolerance_regression_blocked_count"] == 1
    assert controller.history["tolerance_regression_blocked"] == [True]
    assert summary["max_abs_coil_current"] == pytest.approx(0.0)


def test_controller_flags_zero_response_authority() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_ZeroAuthorityKernel,
        verbose=False,
    )

    summary = controller.run_tracking_shot(
        shot_steps=1,
        gain=1.0,
        stop_on_convergence=False,
    )

    assert summary["response_degenerate_count"] == 1
    assert summary["min_response_rank"] == 0
    assert summary["supervisor_intervention_count"] == 1
    assert controller.history["response_degenerate"] == [True]
    assert controller.history["response_rank"] == [0]
    assert summary["max_abs_coil_current"] == pytest.approx(0.0)
    assert summary["final_tracking_error_norm"] > 0.0


def test_controller_uses_tolerance_deadband_to_avoid_chatter() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_WithinToleranceKernel,
        verbose=False,
    )

    summary = controller.run_tracking_shot(
        shot_steps=1,
        gain=1.0,
        stop_on_convergence=False,
    )

    assert summary["final_tracking_error_norm"] > 0.0
    assert summary["final_control_error_norm"] == pytest.approx(0.0)
    assert summary["max_abs_delta_i"] == pytest.approx(0.0)
    assert summary["max_abs_coil_current"] == pytest.approx(0.0)
    assert summary["final_active_control_rows"] == 0
    assert controller.history["active_control_rows"] == [0]


def test_controller_prefers_coils_with_remaining_headroom() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_HeadroomAllocationKernel,
        verbose=False,
        response_regularization=0.5,
    )
    initial_currents = controller.coils.currents.copy()

    summary = controller.run_tracking_shot(
        shot_steps=1,
        gain=1.0,
        stop_on_convergence=False,
    )
    delta_currents = controller.coils.currents - initial_currents

    assert delta_currents.shape == (2,)
    assert abs(delta_currents[0]) > abs(delta_currents[1])
    assert summary["max_coil_penalty"] > 1.0
    assert summary["max_abs_coil_current"] <= 10.0 + 1e-9


def test_run_free_boundary_tracking_with_real_kernel_smoke(tmp_path: Path) -> None:
    cfg_path = _write_real_kernel_tracking_config(tmp_path / "real_tracking.json")

    summary = run_free_boundary_tracking(
        config_file=str(cfg_path),
        shot_steps=2,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        stop_on_convergence=False,
    )

    assert summary["boundary_variant"] == "free_boundary"
    assert summary["steps"] == 2
    assert summary["objective_convergence_active"] is True
    assert summary["shape_rms"] is not None
    assert np.isfinite(summary["final_tracking_error_norm"])
    assert np.isfinite(summary["mean_tracking_error_norm"])
    assert summary["max_abs_coil_current"] <= 5.0e4 + 1e-9
