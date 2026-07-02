# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Configuration Tests
"""Configuration-contract tests for free-boundary tracking control."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_fusion.control.free_boundary_tracking as tracking_api
from scpn_fusion.control._free_boundary_tracking_types import _ObjectiveBlock
from scpn_fusion.control.free_boundary_tracking import FreeBoundaryTrackingController
from scpn_fusion.control.state_estimator import ExtendedKalmanFilter
from scpn_fusion.core.fusion_kernel import CoilSet

FloatArray = NDArray[np.float64]


class _ConfigKernel:
    """Free-boundary kernel stand-in backed by a supplied coil set and config."""

    cfg: dict[str, Any]
    R: FloatArray
    Z: FloatArray
    RR: FloatArray
    ZZ: FloatArray
    Psi: FloatArray

    def __init__(self, _config_file: str, cfg: dict[str, Any], coilset: CoilSet) -> None:
        self.cfg = copy.deepcopy(cfg)
        self._coilset = _copy_coilset(coilset)
        self.R = np.linspace(3.0, 5.0, 4, dtype=np.float64)
        self.Z = np.linspace(-2.0, 1.0, 4, dtype=np.float64)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((self.Z.size, self.R.size), dtype=np.float64)

    def build_coilset_from_config(self) -> CoilSet:
        """Return an isolated copy of the configured coil set."""
        return _copy_coilset(self._coilset)

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
    ) -> dict[str, float | bool | str]:
        """Accept the production free-boundary solve call shape."""
        del coils, max_outer_iter, tol, optimize_shape
        return {
            "boundary_variant": "free_boundary" if boundary_variant is None else boundary_variant,
            "converged": True,
            "outer_iterations": 1,
            "final_diff": 0.0,
        }

    def _sample_flux_at_points(self, points: FloatArray) -> FloatArray:
        """Return target fluxes for the configured diagnostic point set."""
        pts = np.asarray(points, dtype=np.float64)
        if (
            self._coilset.target_flux_points is not None
            and self._coilset.target_flux_values is not None
            and pts.shape == self._coilset.target_flux_points.shape
            and np.allclose(pts, self._coilset.target_flux_points)
        ):
            return np.asarray(self._coilset.target_flux_values.copy(), dtype=np.float64)
        if (
            self._coilset.divertor_strike_points is not None
            and self._coilset.divertor_flux_values is not None
            and pts.shape == self._coilset.divertor_strike_points.shape
            and np.allclose(pts, self._coilset.divertor_strike_points)
        ):
            return np.asarray(self._coilset.divertor_flux_values.copy(), dtype=np.float64)
        raise ValueError("unexpected diagnostic points")

    def find_x_point(self, _psi: FloatArray) -> tuple[tuple[float, float], float]:
        """Return the configured X-point target as the current X-point."""
        if self._coilset.x_point_target is None:
            return (4.0, -1.0), 0.0
        target = np.asarray(self._coilset.x_point_target, dtype=np.float64).reshape(2)
        flux = (
            0.0 if self._coilset.x_point_flux_target is None else self._coilset.x_point_flux_target
        )
        return (float(target[0]), float(target[1])), float(flux)

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        """Return the configured X-point flux at any requested point."""
        del r_pt, z_pt
        return (
            0.0 if self._coilset.x_point_flux_target is None else self._coilset.x_point_flux_target
        )


class _KernelWithoutCoilBuilder:
    """Kernel stand-in that lacks the required coil-set builder hook."""

    cfg: dict[str, Any]

    def __init__(self, _config_file: str) -> None:
        self.cfg = {}


class _KernelWithoutSolver:
    """Kernel stand-in that can build coils but cannot solve free-boundary state."""

    cfg: dict[str, Any]

    def __init__(self, _config_file: str) -> None:
        self.cfg = {}

    def build_coilset_from_config(self) -> CoilSet:
        """Return a valid coil set so constructor setup can finish."""
        return _coilset()


def _copy_array(array: FloatArray | None) -> FloatArray | None:
    """Return a copied ``float64`` array or ``None``."""
    if array is None:
        return None
    return np.asarray(array, dtype=np.float64).copy()


def _copy_coilset(coilset: CoilSet) -> CoilSet:
    """Return a deep-enough copy of a ``CoilSet`` for controller tests."""
    return CoilSet(
        positions=list(coilset.positions),
        currents=np.asarray(coilset.currents, dtype=np.float64).copy(),
        turns=list(coilset.turns),
        current_limits=_copy_array(coilset.current_limits),
        target_flux_points=_copy_array(coilset.target_flux_points),
        target_flux_values=_copy_array(coilset.target_flux_values),
        x_point_target=_copy_array(coilset.x_point_target),
        x_point_flux_target=coilset.x_point_flux_target,
        divertor_strike_points=_copy_array(coilset.divertor_strike_points),
        divertor_flux_values=_copy_array(coilset.divertor_flux_values),
    )


def _coilset(
    *,
    x_point_target: FloatArray | None | bool = True,
    x_point_flux_target: float | None = 0.30,
) -> CoilSet:
    """Return a representative free-boundary tracking coil set."""
    resolved_x_point: FloatArray | None
    if x_point_target is True:
        resolved_x_point = np.array([4.0, -1.0], dtype=np.float64)
    elif x_point_target is False:
        resolved_x_point = None
    else:
        resolved_x_point = x_point_target
    return CoilSet(
        positions=[(3.2, 2.0), (4.8, -2.0)],
        currents=np.array([0.1, -0.1], dtype=np.float64),
        turns=[12, 12],
        current_limits=np.array([1.0, 1.5], dtype=np.float64),
        target_flux_points=np.array([[3.6, 0.0], [4.2, 0.2]], dtype=np.float64),
        target_flux_values=np.array([0.10, 0.20], dtype=np.float64),
        x_point_target=resolved_x_point,
        x_point_flux_target=x_point_flux_target,
        divertor_strike_points=np.array([[3.1, -2.4]], dtype=np.float64),
        divertor_flux_values=np.array([0.40], dtype=np.float64),
    )


def _coilset_with_current_limits(current_limits: FloatArray | None) -> CoilSet:
    """Return a representative coil set with explicit current limits."""
    coilset = _coilset()
    coilset.current_limits = current_limits
    return coilset


def _empty_coilset() -> CoilSet:
    """Return an intentionally empty coil set for constructor guard coverage."""
    return CoilSet(
        positions=[],
        currents=np.array([], dtype=np.float64),
        turns=[],
        current_limits=None,
    )


def _coilset_without_targets() -> CoilSet:
    """Return a valid coil set without explicit tracking objectives."""
    return CoilSet(
        positions=[(3.2, 2.0), (4.8, -2.0)],
        currents=np.array([0.1, -0.1], dtype=np.float64),
        turns=[12, 12],
        current_limits=np.array([1.0, 1.5], dtype=np.float64),
    )


def _factory(
    cfg: dict[str, Any] | None = None,
    *,
    coilset: CoilSet | None = None,
) -> Callable[[str], _ConfigKernel]:
    """Return a kernel factory compatible with ``FreeBoundaryTrackingController``."""
    resolved_cfg: dict[str, Any] = {} if cfg is None else cfg
    resolved_coilset = _coilset() if coilset is None else coilset

    def _build(config_file: str) -> _ConfigKernel:
        return _ConfigKernel(config_file, resolved_cfg, resolved_coilset)

    return _build


def _controller(
    cfg: dict[str, Any] | None = None,
    *,
    coilset: CoilSet | None = None,
    control_dt_s: float | None = None,
    coil_slew_limits: float | list[float] | None = None,
    hold_steps_after_reject: int | None = None,
    state_estimator: ExtendedKalmanFilter | None = None,
) -> FreeBoundaryTrackingController:
    """Construct the production controller against a deterministic kernel."""
    return FreeBoundaryTrackingController(
        "config.json",
        kernel_factory=_factory(cfg, coilset=coilset),
        verbose=False,
        control_dt_s=control_dt_s,
        coil_slew_limits=coil_slew_limits,
        hold_steps_after_reject=hold_steps_after_reject,
        state_estimator=state_estimator,
    )


def test_controller_accepts_vector_slew_limits_and_broadcast_measurement_bias() -> None:
    """Vector slew limits and single-entry block vectors resolve to controller state."""
    controller = _controller(
        {
            "free_boundary": {
                "objective_tolerances": {
                    "shape_rms": 0.25,
                    "shape_max_abs": 0.40,
                    "x_point_position": 0.05,
                    "x_point_flux": 0.02,
                    "divertor_rms": 0.10,
                }
            },
            "free_boundary_tracking": {
                "coil_slew_limits": [0.2, 0.3],
                "measurement_bias": {"shape_flux": [0.05]},
                "supervisor_limits": {"shape_rms": 1.0},
            },
        }
    )

    assert controller.coil_slew_limits.tolist() == [0.2, 0.3]
    assert controller.measurement_bias_vector[:2].tolist() == [0.05, 0.05]
    assert controller.control_objective_weights[:2].tolist() == [4.0, 4.0]
    assert controller.control_objective_weights[2:4].tolist() == [20.0, 20.0]
    assert controller.control_objective_weights[4] == 50.0
    assert controller.control_objective_weights[5] == 10.0


def test_controller_logs_when_verbose(caplog: pytest.LogCaptureFixture) -> None:
    """Verbose controller logging emits through the tracking configuration logger."""
    controller = FreeBoundaryTrackingController(
        "config.json",
        kernel_factory=_factory(),
        verbose=True,
    )

    with caplog.at_level(logging.INFO, logger="scpn_fusion.control._free_boundary_tracking_config"):
        controller._log("tracking config log")

    assert "tracking config log" in caplog.text


@pytest.mark.parametrize(
    ("cfg", "match"),
    (
        ({"free_boundary": {"objective_tolerances": ["shape_rms"]}}, "objective_tolerances"),
        ({"free_boundary": {"objective_tolerances": {"bad_key": 0.1}}}, "Unknown"),
        ({"free_boundary": {"objective_tolerances": {"shape_rms": -0.1}}}, "shape_rms"),
        ({"free_boundary_tracking": {"observer_max_abs": float("nan")}}, "finite or infinity"),
        ({"free_boundary_tracking": {"observer_gain": -0.1}}, "observer_gain"),
        ({"free_boundary_tracking": {"observer_forgetting": 1.1}}, "observer_forgetting"),
        ({"free_boundary_tracking": {"supervisor_limits": ["shape_rms"]}}, "supervisor_limits"),
        (
            {"free_boundary_tracking": {"supervisor_limits": {"bad_key": 1.0}}},
            "Unknown",
        ),
        (
            {"free_boundary_tracking": {"supervisor_limits": {"shape_rms": -1.0}}},
            "shape_rms",
        ),
        ({"free_boundary_tracking": {"fallback_currents": [0.1]}}, "fallback_currents"),
        (
            {"free_boundary_tracking": {"fallback_currents": [0.1, float("nan")]}},
            "fallback_currents",
        ),
        ({"free_boundary_tracking": {"fallback_currents": [2.0, 0.0]}}, "fallback_currents"),
        ({"free_boundary_tracking": {"measurement_bias": ["shape_flux"]}}, "measurement_bias"),
        (
            {"free_boundary_tracking": {"measurement_bias": {"bad_key": 0.1}}},
            "Unknown",
        ),
        (
            {"free_boundary_tracking": {"measurement_bias": {"shape_flux": [0.1, 0.2, 0.3]}}},
            "shape_flux",
        ),
        (
            {"free_boundary_tracking": {"measurement_bias": {"shape_flux": [0.1, float("inf")]}}},
            "finite",
        ),
    ),
)
def test_controller_rejects_invalid_tracking_configuration(
    cfg: dict[str, Any],
    match: str,
) -> None:
    """Invalid tracking configuration is rejected during controller construction."""
    with pytest.raises(ValueError, match=match):
        _controller(cfg)


def test_controller_rejects_invalid_constructor_overrides() -> None:
    """Invalid explicit constructor overrides fail before a control shot starts."""
    with pytest.raises(ValueError, match="identification_perturbation"):
        FreeBoundaryTrackingController(
            "config.json",
            kernel_factory=_factory(),
            verbose=False,
            identification_perturbation=0.0,
        )
    with pytest.raises(ValueError, match="correction_limit"):
        FreeBoundaryTrackingController(
            "config.json",
            kernel_factory=_factory(),
            verbose=False,
            correction_limit=0.0,
        )
    with pytest.raises(ValueError, match="response_regularization"):
        FreeBoundaryTrackingController(
            "config.json",
            kernel_factory=_factory(),
            verbose=False,
            response_regularization=-1.0,
        )
    with pytest.raises(ValueError, match="response_refresh_steps"):
        FreeBoundaryTrackingController(
            "config.json",
            kernel_factory=_factory(),
            verbose=False,
            response_refresh_steps=0,
        )
    with pytest.raises(ValueError, match="solve_max_outer_iter"):
        FreeBoundaryTrackingController(
            "config.json",
            kernel_factory=_factory(),
            verbose=False,
            solve_max_outer_iter=0,
        )
    with pytest.raises(ValueError, match="solve_tol"):
        FreeBoundaryTrackingController(
            "config.json",
            kernel_factory=_factory(),
            verbose=False,
            solve_tol=0.0,
        )
    with pytest.raises(ValueError, match="control_dt_s"):
        _controller(control_dt_s=0.0)
    with pytest.raises(ValueError, match="hold_steps_after_reject"):
        _controller(hold_steps_after_reject=-1)
    with pytest.raises(ValueError, match="coil_slew_limits"):
        _controller(coil_slew_limits=[0.2])
    with pytest.raises(ValueError, match="coil_slew_limits"):
        _controller(coil_slew_limits=[0.2, 0.0])


def test_controller_requires_kernel_coil_builder() -> None:
    """The production controller rejects kernels without the coil builder hook."""
    with pytest.raises(TypeError, match="build_coilset_from_config"):
        FreeBoundaryTrackingController(
            "config.json",
            kernel_factory=_KernelWithoutCoilBuilder,
            verbose=False,
        )


def test_controller_rejects_malformed_coil_sets() -> None:
    """Coil-set cardinality and limit validation fail during construction."""
    with pytest.raises(ValueError, match="at least one external coil"):
        _controller(coilset=_empty_coilset())
    with pytest.raises(ValueError, match="current_limits must match"):
        _controller(coilset=_coilset_with_current_limits(np.array([1.0], dtype=np.float64)))
    with pytest.raises(ValueError, match="current_limits must be finite"):
        _controller(coilset=_coilset_with_current_limits(np.array([1.0, np.inf], dtype=np.float64)))
    with pytest.raises(ValueError, match="explicit target values"):
        _controller(coilset=_coilset_without_targets())


def test_controller_accepts_missing_coil_current_limits() -> None:
    """Absent current limits are represented as unbounded finite-control state."""
    controller = _controller(coilset=_coilset_with_current_limits(None))

    assert controller.coil_current_limits.tolist() == [np.inf, np.inf]


def test_run_free_boundary_tracking_resolves_default_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public wrapper resolves the packaged ITER config when omitted."""
    default_path = Path("default_iter_config.json")
    monkeypatch.setattr(tracking_api, "default_iter_config_path", lambda: default_path)

    summary = tracking_api.run_free_boundary_tracking(
        kernel_factory=_factory(),
        shot_steps=1,
        verbose=False,
    )

    assert summary["config_path"] == str(default_path)


def test_runtime_latency_prediction_can_project_without_state_update() -> None:
    """Latency compensation can project a delayed vector without mutating estimator state."""
    controller = _controller(
        {
            "free_boundary_tracking": {
                "measurement_latency_steps": 2,
                "latency_compensation_gain": 1.0,
            }
        }
    )
    delayed = np.ones_like(controller.target_vector, dtype=np.float64)
    controller._last_delayed_measurement = np.zeros_like(delayed, dtype=np.float64)

    predicted = controller._predict_current_objectives(
        delayed,
        allow_compensation=True,
        update_state=False,
    )

    assert np.allclose(predicted, 2.0 * delayed)
    assert controller._last_delayed_measurement is not None
    assert np.allclose(controller._last_delayed_measurement, np.zeros_like(delayed))


def test_runtime_observation_uses_optional_ekf_refinement() -> None:
    """Observation snapshots can refine measured X-point coordinates through EKF state."""
    estimator = ExtendedKalmanFilter(
        x0=np.array([3.5, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        P0=np.eye(6, dtype=np.float64),
        Q=np.eye(6, dtype=np.float64) * 1.0e-4,
        R_cov=np.eye(4, dtype=np.float64) * 1.0e-3,
    )
    controller = _controller(state_estimator=estimator)

    snapshot = controller._observe_snapshot()

    assert np.allclose(snapshot.measured[2:4], estimator.estimate()[:2])
    assert snapshot.effective.shape == controller.target_vector.shape


def test_runtime_rejects_invalid_command_and_missing_solver() -> None:
    """Runtime actuator and solver guard rails fail closed on malformed inputs."""
    controller = _controller()
    with pytest.raises(ValueError, match="gain"):
        controller._command_currents(np.ones(controller.n_coils, dtype=np.float64), gain=0.0)
    with pytest.raises(ValueError, match="commanded currents"):
        controller._apply_commanded_currents(np.array([0.0], dtype=np.float64))

    solverless = FreeBoundaryTrackingController(
        "config.json",
        kernel_factory=_KernelWithoutSolver,
        verbose=False,
    )
    with pytest.raises(AttributeError, match="solve"):
        solverless._solve_free_boundary_state()


@pytest.mark.parametrize(
    ("block", "coil_attr", "match"),
    (
        (_ObjectiveBlock("shape_flux", 0, 2), "target_flux_points", "shape-flux"),
        (_ObjectiveBlock("x_point_flux", 0, 1), "x_point_target", "x_point_flux"),
        (
            _ObjectiveBlock("divertor_flux", 0, 1),
            "divertor_strike_points",
            "divertor-flux",
        ),
    ),
)
def test_runtime_observation_rejects_missing_objective_geometry(
    block: _ObjectiveBlock,
    coil_attr: str,
    match: str,
) -> None:
    """Observation fails closed when objective geometry is missing at runtime."""
    controller = _controller()
    setattr(controller.coils, coil_attr, None)
    controller.objective_blocks = (block,)

    with pytest.raises(ValueError, match=match):
        controller._observe_true_objectives()


def test_runtime_observation_rejects_unknown_objective_block() -> None:
    """Runtime observation rejects corrupted objective-block names."""
    controller = _controller()
    controller.objective_blocks = (_ObjectiveBlock("unknown_objective", 0, 1),)

    with pytest.raises(ValueError, match="Unknown objective block"):
        controller._observe_true_objectives()


def test_control_identification_handles_zero_perturbation_headroom() -> None:
    """Response identification marks a coil column zero when limits block perturbation."""
    controller = _controller()
    controller.coil_current_limits[:] = 0.0

    response = controller.identify_response_matrix(perturbation=0.25)

    assert np.allclose(response, np.zeros_like(response))
    assert controller.response_degenerate is True


def test_control_response_diagnostics_handle_empty_matrix() -> None:
    """Response diagnostics classify empty response spectra as degenerate."""
    controller = _controller()
    controller.response_matrix = np.empty((controller.target_vector.size, 0), dtype=np.float64)

    controller._update_response_diagnostics()

    assert controller.response_rank == 0
    assert controller.response_condition_number == np.inf
    assert controller.response_max_singular_value == 0.0
    assert controller.response_degenerate is True


def test_control_activation_mask_rejects_unknown_objective_block() -> None:
    """Control-row masking rejects corrupted objective block names."""
    controller = _controller()
    controller.objective_blocks = (_ObjectiveBlock("unknown_objective", 0, 1),)

    with pytest.raises(ValueError, match="Unknown objective block"):
        controller._build_control_activation_mask({"objective_checks": {}})


def test_control_penalties_handle_unbounded_coil_headroom() -> None:
    """Coil penalty weighting leaves unbounded current limits unpenalized."""
    controller = _controller(coilset=_coilset_with_current_limits(None))

    penalties = controller._build_coil_penalties(np.array([1.0, -1.0], dtype=np.float64))

    assert penalties.tolist() == [1.0, 1.0]


def test_control_correction_and_fallback_guards() -> None:
    """Correction and fallback helpers fail closed on malformed state."""
    controller = _controller()
    with pytest.raises(ValueError, match="observation"):
        controller.compute_correction(np.array([0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="fallback currents"):
        controller._apply_fallback_currents()

    controller.n_coils = 0
    controller._coil_actuators = []
    controller.fallback_currents = np.array([], dtype=np.float64)

    assert controller._apply_fallback_currents() == 0.0


def test_control_objective_metrics_cover_divertor_max_abs_check() -> None:
    """Divertor max-abs tolerance contributes to objective convergence checks."""
    controller = _controller({"free_boundary": {"objective_tolerances": {"divertor_max_abs": 0.1}}})

    metrics = controller.evaluate_objectives(controller.target_vector.copy())

    assert metrics["objective_checks"]["divertor_max_abs"] is True


def test_control_tolerance_regression_skips_missing_metric_values() -> None:
    """Tolerance-regression detection ignores metrics absent from either side."""
    regressions = FreeBoundaryTrackingController._detect_tolerance_regressions(
        {
            "objective_tolerances": {"shape_rms": 0.1},
            "shape_rms": None,
        },
        {"shape_rms": 0.2},
    )

    assert regressions == {}


def test_shot_holds_after_degenerate_response_and_stops_when_converged() -> None:
    """Shot execution arms reject hold and honors convergence early-stop."""
    controller = _controller(hold_steps_after_reject=2)

    summary = controller.run_tracking_shot(shot_steps=3, stop_on_convergence=True)

    assert summary["steps"] == 1
    assert summary["response_degenerate_count"] == 1
    assert controller.history["supervisor_hold_steps_remaining"] == [2]


def test_controller_rejects_x_point_flux_without_target() -> None:
    """An X-point flux target requires the corresponding X-point location."""
    with pytest.raises(ValueError, match="x_point_flux_target requires x_point_target"):
        _controller(coilset=_coilset(x_point_target=False, x_point_flux_target=0.2))


def test_actuator_restore_rejects_wrong_snapshot_count() -> None:
    """Actuator snapshots must match the controller coil count."""
    controller = _controller()

    with pytest.raises(ValueError, match="snapshot count"):
        controller._restore_actuator_states(())


def test_control_weight_builder_rejects_unknown_objective_block() -> None:
    """Defensive objective-block validation rejects corrupted controller state."""
    controller = _controller()
    controller.objective_blocks = (_ObjectiveBlock("unknown_objective", 0, 1),)

    with pytest.raises(ValueError, match="Unknown objective block"):
        controller._build_control_objective_weights()
