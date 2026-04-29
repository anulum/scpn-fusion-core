# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Control
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from scpn_fusion.control._free_boundary_tracking_types import (
    FloatArray,
    _ActuatorSnapshot,
    _ObjectiveBlock,
)
from scpn_fusion.control.tokamak_flight_sim import FirstOrderActuator


logger = logging.getLogger(__name__)


class _FreeBoundaryTrackingConfigMixin:
    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    @staticmethod
    def _resolve_objective_tolerances(
        cfg_tolerances: Any,
        override_tolerances: dict[str, float] | None,
    ) -> dict[str, float]:
        allowed = {
            "shape_rms",
            "shape_max_abs",
            "x_point_position",
            "x_point_flux",
            "divertor_rms",
            "divertor_max_abs",
        }
        merged: dict[str, float] = {}
        for raw, name in (
            (cfg_tolerances, "free_boundary.objective_tolerances"),
            (override_tolerances, "objective_tolerances"),
        ):
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{name} must be a mapping of tolerance names to non-negative floats."
                )
            for key, value in raw.items():
                if key not in allowed:
                    allowed_keys = ", ".join(sorted(allowed))
                    raise ValueError(f"Unknown {name} key {key!r}. Allowed keys: {allowed_keys}.")
                tol_value = float(value)
                if not np.isfinite(tol_value) or tol_value < 0.0:
                    raise ValueError(f"{name}.{key} must be finite and >= 0.")
                merged[key] = tol_value
        return merged

    @staticmethod
    def _resolve_positive_float(
        cfg_value: Any,
        override_value: float | None,
        *,
        default: float,
        name: str,
    ) -> float:
        raw_value = (
            default
            if override_value is None and cfg_value is None
            else (cfg_value if override_value is None else override_value)
        )
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0.")
        return value

    @staticmethod
    def _resolve_nonnegative_int(
        cfg_value: Any,
        override_value: int | None,
        *,
        default: int,
        name: str,
    ) -> int:
        raw_value = (
            default
            if override_value is None and cfg_value is None
            else (cfg_value if override_value is None else override_value)
        )
        value = int(raw_value)
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")
        return value

    @staticmethod
    def _resolve_nonnegative_float(
        cfg_value: Any,
        *,
        default: float,
        name: str,
    ) -> float:
        raw_value = default if cfg_value is None else cfg_value
        value = float(raw_value)
        if not np.isfinite(value) and not np.isinf(value):
            raise ValueError(f"{name} must be finite or infinity.")
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.")
        return value

    @staticmethod
    def _resolve_fraction(
        cfg_value: Any,
        *,
        default: float,
        name: str,
    ) -> float:
        raw_value = default if cfg_value is None else cfg_value
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
        return value

    def _resolve_coil_slew_limits(
        self,
        cfg_limits: Any,
        override_limits: float | list[float] | None,
    ) -> FloatArray:
        raw = cfg_limits if override_limits is None else override_limits
        if raw is None:
            return np.full(self.n_coils, np.inf, dtype=np.float64)
        if np.isscalar(raw):
            limits = np.full(self.n_coils, float(cast(Any, raw)), dtype=np.float64)
        else:
            limits = np.asarray(raw, dtype=np.float64).reshape(-1)
        if limits.shape != (self.n_coils,):
            raise ValueError("coil_slew_limits must be a scalar or match the number of coils.")
        if np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("coil_slew_limits must contain finite values > 0.")
        return cast(FloatArray, np.asarray(limits, dtype=np.float64))

    @staticmethod
    def _resolve_supervisor_limits(
        cfg_limits: Any,
        override_limits: dict[str, float] | None,
    ) -> dict[str, float]:
        allowed = {
            "tracking_error_norm",
            "shape_rms",
            "shape_max_abs",
            "x_point_position",
            "x_point_flux",
            "divertor_rms",
            "divertor_max_abs",
            "max_abs_coil_current",
            "max_abs_actuator_lag",
        }
        merged: dict[str, float] = {}
        for raw, source_name in (
            (cfg_limits, "free_boundary_tracking.supervisor_limits"),
            (override_limits, "supervisor_limits"),
        ):
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{source_name} must be a mapping of limit names to non-negative floats."
                )
            for key, value in raw.items():
                if key not in allowed:
                    allowed_keys = ", ".join(sorted(allowed))
                    raise ValueError(
                        f"Unknown {source_name} key {key!r}. Allowed keys: {allowed_keys}."
                    )
                limit_value = float(value)
                if not np.isfinite(limit_value) or limit_value < 0.0:
                    raise ValueError(f"{source_name}.{key} must be finite and >= 0.")
                merged[key] = limit_value
        return merged

    def _resolve_fallback_currents(self, cfg_value: Any) -> FloatArray | None:
        if cfg_value is None:
            return None
        values = np.asarray(cfg_value, dtype=np.float64).reshape(-1)
        if values.shape != (self.n_coils,):
            raise ValueError(
                "free_boundary_tracking.fallback_currents must match the number of coils."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("free_boundary_tracking.fallback_currents must be finite.")
        if np.any(np.abs(values) - self.coil_current_limits > 1e-12):
            raise ValueError(
                "free_boundary_tracking.fallback_currents must respect CoilSet.current_limits."
            )
        return cast(FloatArray, values.copy())

    def _build_coil_actuators(self) -> list[FirstOrderActuator]:
        actuators: list[FirstOrderActuator] = []
        for idx in range(self.n_coils):
            limit = float(self.coil_current_limits[idx])
            actuator = FirstOrderActuator(
                tau_s=self.coil_actuator_tau_s,
                dt_s=self.control_dt_s,
                u_min=-limit,
                u_max=limit,
                rate_limit=float(self.coil_slew_limits[idx]),
            )
            actuator.state = float(self.coils.currents[idx])
            actuator._delay_buffer = [float(self.coils.currents[idx])] * max(
                actuator.delay_steps, 1
            )
            actuators.append(actuator)
        return actuators

    def _snapshot_actuator_states(self) -> tuple[_ActuatorSnapshot, ...]:
        return tuple(
            _ActuatorSnapshot(
                state=float(actuator.state),
                delay_buffer=tuple(float(v) for v in actuator._delay_buffer),
            )
            for actuator in self._coil_actuators
        )

    def _restore_actuator_states(self, snapshots: tuple[_ActuatorSnapshot, ...]) -> None:
        if len(snapshots) != len(self._coil_actuators):
            raise ValueError("actuator snapshot count must match the number of coils.")
        for actuator, snapshot in zip(self._coil_actuators, snapshots):
            actuator.state = float(snapshot.state)
            actuator._delay_buffer = [float(v) for v in snapshot.delay_buffer]

    def _sync_actuators_from_coils(self) -> None:
        for idx, actuator in enumerate(self._coil_actuators):
            current = float(self.coils.currents[idx])
            actuator.state = current
            actuator._delay_buffer = [current] * max(actuator.delay_steps, 1)

    def _build_target_vector(self) -> tuple[FloatArray, tuple[_ObjectiveBlock, ...]]:
        values: list[float] = []
        blocks: list[_ObjectiveBlock] = []
        start = 0

        if self.coils.target_flux_points is not None and self.coils.target_flux_values is not None:
            target_flux = np.asarray(self.coils.target_flux_values, dtype=np.float64).reshape(-1)
            values.extend(float(v) for v in target_flux)
            stop = start + target_flux.size
            blocks.append(_ObjectiveBlock("shape_flux", start, stop))
            start = stop

        if self.coils.x_point_target is not None:
            x_target = np.asarray(self.coils.x_point_target, dtype=np.float64).reshape(2)
            values.extend((float(x_target[0]), float(x_target[1])))
            stop = start + 2
            blocks.append(_ObjectiveBlock("x_point_position", start, stop))
            start = stop
            if self.coils.x_point_flux_target is not None:
                values.append(float(self.coils.x_point_flux_target))
                stop = start + 1
                blocks.append(_ObjectiveBlock("x_point_flux", start, stop))
                start = stop
        elif self.coils.x_point_flux_target is not None:
            raise ValueError(
                "x_point_flux_target requires x_point_target for free-boundary tracking."
            )

        if (
            self.coils.divertor_strike_points is not None
            and self.coils.divertor_flux_values is not None
        ):
            divertor_flux = np.asarray(self.coils.divertor_flux_values, dtype=np.float64).reshape(
                -1
            )
            values.extend(float(v) for v in divertor_flux)
            stop = start + divertor_flux.size
            blocks.append(_ObjectiveBlock("divertor_flux", start, stop))
            start = stop

        return np.asarray(values, dtype=np.float64), tuple(blocks)

    def _resolve_measurement_vector(self, raw_value: Any, *, name: str) -> FloatArray:
        vector = np.zeros_like(self.target_vector, dtype=np.float64)
        if raw_value is None:
            return cast(FloatArray, vector)
        if not isinstance(raw_value, dict):
            raise ValueError(
                f"{name} must be a mapping of objective block names to finite scalars or vectors."
            )

        block_map = {block.name: block for block in self.objective_blocks}
        allowed_keys = ", ".join(sorted(block_map))
        for key, raw_block_value in raw_value.items():
            block = block_map.get(key)
            if block is None:
                raise ValueError(f"Unknown {name} key {key!r}. Allowed keys: {allowed_keys}.")
            width = block.stop - block.start
            if np.isscalar(raw_block_value):
                block_values = np.full(width, float(cast(Any, raw_block_value)), dtype=np.float64)
            else:
                block_values = np.asarray(raw_block_value, dtype=np.float64).reshape(-1)
                if block_values.size == 1:
                    block_values = np.full(width, float(block_values[0]), dtype=np.float64)
            if block_values.shape != (width,):
                raise ValueError(
                    f"{name}.{key} must be a scalar or contain exactly {width} entries."
                )
            if np.any(~np.isfinite(block_values)):
                raise ValueError(f"{name}.{key} must contain only finite values.")
            vector[block.start : block.stop] = block_values
        return cast(FloatArray, np.asarray(vector, dtype=np.float64))

    def _weight_from_tolerances(self, *keys: str) -> float:
        weight = 1.0
        for key in keys:
            tol = self.objective_tolerances.get(key)
            if tol is None:
                continue
            weight = max(weight, 1.0 / max(float(tol), 1.0e-12))
        return float(weight)

    def _build_control_objective_weights(self) -> FloatArray:
        weights = np.ones(self.target_vector.shape, dtype=np.float64)
        for block in self.objective_blocks:
            if block.name == "shape_flux":
                block_weight = self._weight_from_tolerances("shape_rms", "shape_max_abs")
            elif block.name == "x_point_position":
                block_weight = self._weight_from_tolerances("x_point_position")
            elif block.name == "x_point_flux":
                block_weight = self._weight_from_tolerances("x_point_flux")
            elif block.name == "divertor_flux":
                block_weight = self._weight_from_tolerances("divertor_rms", "divertor_max_abs")
            else:
                raise ValueError(f"Unknown objective block {block.name!r}.")
            weights[block.start : block.stop] = block_weight
        return cast(FloatArray, np.asarray(weights, dtype=np.float64))
