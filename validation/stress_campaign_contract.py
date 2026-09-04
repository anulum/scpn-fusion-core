# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stress-campaign scientific contract
"""Typed scientific contract for comparable controller stress campaigns."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCENARIO_SCHEMA = "scpn-fusion-core.stress-scenario.v3"
RESULT_SCHEMA = "scpn-fusion-core.stress-campaign-results.v3"
LaneStatus = Literal["pending", "running", "complete", "partial_failure", "failed", "unavailable"]


def canonical_json(data: Any) -> str:
    """Serialize *data* deterministically for hashes and identity checks."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(data: Any) -> str:
    """Return the SHA-256 digest of canonical JSON data."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StressScenario:
    """Physical disturbances shared by every comparable controller lane.

    Parameters
    ----------
    measurement_noise_std_m : float
        Standard deviation of independent, zero-mean Gaussian errors added to
        radial and vertical magnetic-axis measurements, in metres.
    actuator_delay_s : float
        Pure command transport delay before actuator lag, in seconds.
    master_seed : int
        Unsigned 64-bit seed from which controller/episode seeds are derived.
    """

    measurement_noise_std_m: float = 0.2
    actuator_delay_s: float = 0.05
    master_seed: int = 0

    def __post_init__(self) -> None:
        """Reject ambiguous, non-finite, or non-replayable scenario values."""
        if not math.isfinite(self.measurement_noise_std_m) or self.measurement_noise_std_m < 0.0:
            raise ValueError("measurement_noise_std_m must be finite and >= 0.")
        if not math.isfinite(self.actuator_delay_s) or self.actuator_delay_s < 0.0:
            raise ValueError("actuator_delay_s must be finite and >= 0.")
        if (
            isinstance(self.master_seed, bool)
            or not isinstance(self.master_seed, int)
            or not 0 <= self.master_seed < 2**64
        ):
            raise ValueError("master_seed must be an unsigned 64-bit integer.")

    def to_dict(self) -> dict[str, Any]:
        """Return the unit-explicit scenario representation stored in reports."""
        return {"schema": SCENARIO_SCHEMA, **asdict(self)}

    @property
    def digest(self) -> str:
        """Return the immutable scenario identity used for comparability."""
        return digest_json(self.to_dict())

    def episode_seed(self, episode_index: int) -> int:
        """Derive a stable common-random-number seed for one episode index."""
        if episode_index < 0:
            raise ValueError("episode_index must be >= 0.")
        material = canonical_json(
            {
                "master_seed": int(self.master_seed),
                "episode_index": int(episode_index),
                "scenario_digest": self.digest,
            }
        )
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class EpisodeResult:
    """Scientific and timing outputs from one successfully completed episode."""

    episode_index: int
    seed: int
    scenario_digest: str
    evaluation_contract_digest: str
    disturbance_trace_digest: str
    realized_measurement_noise_rms_m: float
    mean_abs_r_error_m: float
    mean_abs_z_error_m: float
    tracking_reward_m: float
    control_policy_latency_us: list[float]
    simulation_wall_time_us: float
    disrupted: bool
    t_disruption_s: float
    magnetic_actuator_absolute_current_offset_integral_ma_s: float
    mean_abs_coil_current_offset_tracking_error_ma: float

    def __post_init__(self) -> None:
        """Reject malformed or non-finite outputs before they reach a report."""
        if self.episode_index < 0:
            raise ValueError("episode_index must be >= 0.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**64
        ):
            raise ValueError("seed must be an unsigned 64-bit integer.")
        if not self.scenario_digest or not self.evaluation_contract_digest:
            raise ValueError("scenario and evaluation contract digests must not be empty.")
        if not self.disturbance_trace_digest:
            raise ValueError("disturbance_trace_digest must not be empty.")
        nonnegative = {
            "realized_measurement_noise_rms_m": self.realized_measurement_noise_rms_m,
            "mean_abs_r_error_m": self.mean_abs_r_error_m,
            "mean_abs_z_error_m": self.mean_abs_z_error_m,
            "simulation_wall_time_us": self.simulation_wall_time_us,
            "t_disruption_s": self.t_disruption_s,
            "magnetic_actuator_absolute_current_offset_integral_ma_s": (
                self.magnetic_actuator_absolute_current_offset_integral_ma_s
            ),
            "mean_abs_coil_current_offset_tracking_error_ma": (
                self.mean_abs_coil_current_offset_tracking_error_ma
            ),
        }
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative.values()):
            raise ValueError("episode nonnegative metrics must be finite and >= 0.")
        if not math.isfinite(self.tracking_reward_m):
            raise ValueError("tracking_reward_m must be finite.")
        if not self.control_policy_latency_us or any(
            not math.isfinite(value) or value < 0.0 for value in self.control_policy_latency_us
        ):
            raise ValueError("control_policy_latency_us must contain finite nonnegative samples.")
        if not isinstance(self.disrupted, bool):
            raise ValueError("disrupted must be a bool.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe episode record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeResult:
        """Restore an episode record from a validated checkpoint mapping."""
        return cls(**data)


@dataclass(frozen=True)
class EpisodeFailure:
    """Honest record of an episode that did not produce scientific metrics."""

    episode_index: int
    seed: int
    exception_type: str
    message: str
    backend: str
    stage: str
    traceback_text: str

    def __post_init__(self) -> None:
        """Reject failure records that cannot be bound to one episode."""
        if self.episode_index < 0:
            raise ValueError("episode_index must be >= 0.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**64
        ):
            raise ValueError("seed must be an unsigned 64-bit integer.")
        if not self.exception_type:
            raise ValueError("exception_type must not be empty.")
        if not self.backend or not self.stage:
            raise ValueError("failure backend and stage must not be empty.")
        if not self.traceback_text:
            raise ValueError("failure traceback_text must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe failure record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeFailure:
        """Restore a failure record from a checkpoint mapping."""
        return cls(**data)


@dataclass
class ControllerMetrics:
    """One lane's completion state and aggregate metrics.

    Aggregate values stay ``None`` until at least one episode succeeds. This
    prevents a failed or unavailable lane from being serialized as an apparent
    zero-error, zero-latency measurement.
    """

    name: str
    requested_episodes: int
    scenario_digest: str
    evaluation_contract_digest: str
    policy_implementation: str
    status: LaneStatus = "pending"
    reason: str | None = None
    n_episodes: int = 0
    failed_episodes: int = 0
    mean_tracking_reward_m: float | None = None
    std_tracking_reward_m: float | None = None
    mean_abs_r_error_m: float | None = None
    mean_abs_z_error_m: float | None = None
    p50_control_policy_latency_us: float | None = None
    p95_control_policy_latency_us: float | None = None
    p99_control_policy_latency_us: float | None = None
    disruption_rate: float | None = None
    mean_def: float | None = None
    mean_magnetic_actuator_absolute_current_offset_integral_ma_s: float | None = None
    mean_abs_coil_current_offset_tracking_error_ma: float | None = None
    episodes: list[EpisodeResult] = field(default_factory=list)
    failures: list[EpisodeFailure] = field(default_factory=list)

    def to_dict(self, *, include_episodes: bool = True) -> dict[str, Any]:
        """Return a report/checkpoint representation of the lane."""
        data = asdict(self)
        if not include_episodes:
            data.pop("episodes")
            data.pop("failures")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControllerMetrics:
        """Restore lane state, including individual success and failure records."""
        restored = dict(data)
        restored["episodes"] = [EpisodeResult.from_dict(item) for item in data.get("episodes", [])]
        restored["failures"] = [EpisodeFailure.from_dict(item) for item in data.get("failures", [])]
        return cls(**restored)

    def finalize(self, shot_duration_s: float) -> None:
        """Compute aggregates and an honest terminal status from episode records."""
        import numpy as np

        if not math.isfinite(shot_duration_s) or shot_duration_s <= 0.0:
            raise ValueError("shot_duration_s must be finite and > 0.")
        self.n_episodes = len(self.episodes)
        self.failed_episodes = len(self.failures)
        self.validate_records()
        if not self.episodes:
            self.status = "failed" if self.failures else "unavailable"
            if self.reason is None:
                self.reason = "no_successful_episodes"
            return

        rewards = np.asarray([item.tracking_reward_m for item in self.episodes], dtype=np.float64)
        latencies = np.asarray(
            [sample for item in self.episodes for sample in item.control_policy_latency_us],
            dtype=np.float64,
        )
        self.mean_tracking_reward_m = float(np.mean(rewards))
        self.std_tracking_reward_m = float(np.std(rewards))
        self.mean_abs_r_error_m = float(
            np.mean([item.mean_abs_r_error_m for item in self.episodes])
        )
        self.mean_abs_z_error_m = float(
            np.mean([item.mean_abs_z_error_m for item in self.episodes])
        )
        self.p50_control_policy_latency_us = float(np.percentile(latencies, 50))
        self.p95_control_policy_latency_us = float(np.percentile(latencies, 95))
        self.p99_control_policy_latency_us = float(np.percentile(latencies, 99))
        self.disruption_rate = float(np.mean([item.disrupted for item in self.episodes]))
        self.mean_def = float(
            np.mean([item.t_disruption_s / shot_duration_s for item in self.episodes])
        )
        self.mean_magnetic_actuator_absolute_current_offset_integral_ma_s = float(
            np.mean(
                [
                    item.magnetic_actuator_absolute_current_offset_integral_ma_s
                    for item in self.episodes
                ]
            )
        )
        self.mean_abs_coil_current_offset_tracking_error_ma = float(
            np.mean([item.mean_abs_coil_current_offset_tracking_error_ma for item in self.episodes])
        )
        if self.n_episodes == self.requested_episodes and not self.failures:
            self.status = "complete"
            self.reason = None
        else:
            self.status = "partial_failure"
            self.reason = "one_or_more_episodes_failed"

    def validate_records(self) -> None:
        """Reject duplicate, out-of-range, or cross-scenario episode records."""
        allowed_statuses = {
            "pending",
            "running",
            "complete",
            "partial_failure",
            "failed",
            "unavailable",
        }
        if self.status not in allowed_statuses:
            raise ValueError(f"lane {self.name!r} has an unsupported status.")
        if self.requested_episodes < 1:
            raise ValueError(f"lane {self.name!r} requested_episodes must be >= 1.")
        if not self.policy_implementation:
            raise ValueError(f"lane {self.name!r} policy_implementation must not be empty.")
        if self.n_episodes != len(self.episodes):
            raise ValueError(f"lane {self.name!r} success count does not match records.")
        if self.failed_episodes != len(self.failures):
            raise ValueError(f"lane {self.name!r} failure count does not match records.")
        successful = [item.episode_index for item in self.episodes]
        failed = [item.episode_index for item in self.failures]
        all_indices = [*successful, *failed]
        if len(all_indices) != len(set(all_indices)):
            raise ValueError(f"lane {self.name!r} contains duplicate episode indices.")
        if any(index < 0 or index >= self.requested_episodes for index in all_indices):
            raise ValueError(f"lane {self.name!r} contains an out-of-range episode index.")
        if any(item.scenario_digest != self.scenario_digest for item in self.episodes):
            raise ValueError(f"lane {self.name!r} contains a cross-scenario episode.")
        if any(
            item.evaluation_contract_digest != self.evaluation_contract_digest
            for item in self.episodes
        ):
            raise ValueError(f"lane {self.name!r} contains a cross-contract episode.")

    @property
    def comparable(self) -> bool:
        """Return whether this lane completed the entire declared scenario."""
        return (
            self.status == "complete"
            and self.n_episodes == self.requested_episodes
            and self.failed_episodes == 0
            and all(item.scenario_digest == self.scenario_digest for item in self.episodes)
            and all(
                item.evaluation_contract_digest == self.evaluation_contract_digest
                for item in self.episodes
            )
        )


class CampaignResults(dict[str, ControllerMetrics]):
    """Controller mapping carrying the exact scenario and recovery identity."""

    def __init__(
        self,
        *args: Any,
        scenario: StressScenario,
        campaign_identity: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Initialize the mapping and bind its common scientific identity."""
        super().__init__(*args, **kwargs)
        self.scenario = scenario
        self.campaign_identity = campaign_identity
