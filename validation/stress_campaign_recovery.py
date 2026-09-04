# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stress-campaign recovery
"""Atomic checkpoint, exact-identity resume, and operator-progress support."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from validation.stress_campaign_contract import ControllerMetrics, StressScenario, digest_json
from scpn_fusion.io.safe_loaders import checked_json_load

CHECKPOINT_SCHEMA = "scpn-fusion-core.stress-campaign-checkpoint.v2"
PROGRESS_SCHEMA = "scpn-fusion-core.stress-campaign-progress.v2"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Durably replace a JSON file without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_campaign_identity(
    *,
    repo_root: Path,
    config_path: Path,
    scenario: StressScenario,
    controllers: list[str],
    requested_episodes: int,
    shot_duration_s: int,
    surrogate: bool,
    software_versions: dict[str, str],
    execution_options: dict[str, bool],
    evaluation_contract: dict[str, Any],
    controller_implementations: dict[str, str],
) -> dict[str, Any]:
    """Bind recovery to exact inputs, source files, and execution environment."""
    source_paths = {
        path.relative_to(repo_root) for path in (repo_root / "src/scpn_fusion").rglob("*.py")
    }
    source_paths.update(
        path.relative_to(repo_root)
        for path in (repo_root / "validation").glob("stress_campaign*.py")
    )
    source_paths.add(Path("validation/stress_test_campaign.py"))
    for crate in ("fusion-control", "fusion-python"):
        source_paths.update(
            path.relative_to(repo_root)
            for path in (repo_root / "scpn-fusion-rs/crates" / crate).rglob("*.rs")
        )
    runtime_artifacts: dict[str, str] = {}
    if surrogate:
        weights_path = repo_root / "weights" / "neural_equilibrium_sparc.npz"
        runtime_artifacts[str(weights_path.relative_to(repo_root))] = (
            sha256_file(weights_path) if weights_path.is_file() else "absent"
        )
    payload: dict[str, Any] = {
        "scenario": scenario.to_dict(),
        "controllers": list(controllers),
        "requested_episodes_per_controller": int(requested_episodes),
        "shot_duration_s": int(shot_duration_s),
        "surrogate": bool(surrogate),
        "execution_options": dict(sorted(execution_options.items())),
        "evaluation_contract": evaluation_contract,
        "controller_implementations": dict(sorted(controller_implementations.items())),
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "source_sha256": {
            str(relative): sha256_file(repo_root / relative)
            for relative in sorted(source_paths, key=str)
        },
        "runtime_artifact_sha256": runtime_artifacts,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "software": dict(sorted(software_versions.items())),
        },
    }
    return {"digest": digest_json(payload), "payload": payload}


class RecoveryStore:
    """Manage one campaign's atomic lane checkpoints and progress document."""

    def __init__(
        self,
        directory: Path,
        campaign_identity: dict[str, Any],
        *,
        require_existing: bool = False,
    ) -> None:
        """Create a store rooted at *directory* for an exact campaign identity."""
        self.directory = directory
        self.campaign_identity = campaign_identity
        self.writer_run_id = str(uuid.uuid4())
        self.directory.mkdir(parents=True, exist_ok=True)
        identity_path = self.directory / "campaign.identity.json"
        if identity_path.is_file():
            existing = checked_json_load(identity_path)
            if existing != campaign_identity:
                raise ValueError(f"checkpoint campaign identity mismatch: {identity_path}")
        elif require_existing:
            raise FileNotFoundError(f"resume identity manifest does not exist: {identity_path}")
        else:
            atomic_write_json(identity_path, campaign_identity)

    @staticmethod
    def _lane_slug(controller_name: str) -> str:
        """Return a stable filesystem-safe lane name."""
        slug = re.sub(r"[^a-z0-9]+", "-", controller_name.lower()).strip("-")
        if not slug:
            raise ValueError("controller name does not contain a usable lane identifier.")
        return slug

    def checkpoint_path(self, controller_name: str) -> Path:
        """Return the per-lane checkpoint path."""
        return self.directory / f"{self._lane_slug(controller_name)}.checkpoint.json"

    def write_lane(self, metrics: ControllerMetrics) -> None:
        """Atomically persist all completed and failed episode records for a lane."""
        payload = self.campaign_identity["payload"]
        if metrics.name not in payload["controllers"]:
            raise ValueError("cannot checkpoint a controller outside campaign identity")
        if metrics.requested_episodes != payload["requested_episodes_per_controller"]:
            raise ValueError("cannot checkpoint a mismatched requested episode count")
        if metrics.scenario_digest != digest_json(payload["scenario"]):
            raise ValueError("cannot checkpoint a mismatched scenario digest")
        if metrics.evaluation_contract_digest != payload["evaluation_contract"]["digest"]:
            raise ValueError("cannot checkpoint a mismatched evaluation contract digest")
        if metrics.policy_implementation != payload["controller_implementations"][metrics.name]:
            raise ValueError("cannot checkpoint a mismatched policy implementation")
        metrics.validate_records()
        body = {
            "schema": CHECKPOINT_SCHEMA,
            "campaign_identity": self.campaign_identity,
            "controller": metrics.name,
            "metrics": metrics.to_dict(include_episodes=True),
        }
        body["content_digest"] = digest_json(body)
        atomic_write_json(self.checkpoint_path(metrics.name), body)

    def load_lane(self, controller_name: str) -> ControllerMetrics | None:
        """Restore a lane or reject corruption and any identity mismatch."""
        path = self.checkpoint_path(controller_name)
        if not path.exists():
            return None
        data = checked_json_load(path)
        content_digest = data.pop("content_digest", None)
        if content_digest != digest_json(data):
            raise ValueError(f"checkpoint integrity mismatch: {path}")
        if data.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema: {path}")
        if data.get("campaign_identity") != self.campaign_identity:
            raise ValueError(f"checkpoint campaign identity mismatch: {path}")
        if data.get("controller") != controller_name:
            raise ValueError(f"checkpoint controller mismatch: {path}")
        metrics = ControllerMetrics.from_dict(data["metrics"])
        identity_payload = self.campaign_identity["payload"]
        if metrics.name != controller_name:
            raise ValueError(f"checkpoint embedded controller mismatch: {path}")
        if metrics.requested_episodes != identity_payload["requested_episodes_per_controller"]:
            raise ValueError(f"checkpoint requested episode count mismatch: {path}")
        expected_scenario_digest = digest_json(identity_payload["scenario"])
        if metrics.scenario_digest != expected_scenario_digest:
            raise ValueError(f"checkpoint scenario digest mismatch: {path}")
        expected_contract_digest = identity_payload["evaluation_contract"]["digest"]
        if metrics.evaluation_contract_digest != expected_contract_digest:
            raise ValueError(f"checkpoint evaluation contract digest mismatch: {path}")
        expected_implementation = identity_payload["controller_implementations"][controller_name]
        if metrics.policy_implementation != expected_implementation:
            raise ValueError(f"checkpoint policy implementation mismatch: {path}")
        metrics.validate_records()
        scenario_data = identity_payload["scenario"]
        scenario = StressScenario(
            measurement_noise_std_m=scenario_data["measurement_noise_std_m"],
            actuator_delay_s=scenario_data["actuator_delay_s"],
            master_seed=scenario_data["master_seed"],
        )
        episode_seeds = [(record.episode_index, record.seed) for record in metrics.episodes] + [
            (record.episode_index, record.seed) for record in metrics.failures
        ]
        for episode_index, recorded_seed in episode_seeds:
            expected_seed = scenario.episode_seed(episode_index)
            if recorded_seed != expected_seed:
                raise ValueError(f"checkpoint episode seed mismatch: {path}")
        return metrics

    def write_progress(
        self,
        *,
        lanes: dict[str, ControllerMetrics],
        active_controller: str | None,
        active_episode_index: int | None,
        active_episode_seed: int | None,
        started_monotonic: float,
        attempted_at_start: int,
        snapshot_kind: str = "boundary",
    ) -> None:
        """Publish machine-readable completion counts and a wall-clock ETA."""
        if (active_episode_index is None) != (active_episode_seed is None):
            raise ValueError("active episode index and seed must be reported together")
        if active_controller is None and active_episode_index is not None:
            raise ValueError("an active episode requires an active controller")
        requested = sum(lane.requested_episodes for lane in lanes.values())
        attempted = sum(lane.n_episodes + lane.failed_episodes for lane in lanes.values())
        elapsed_s = max(0.0, time.monotonic() - started_monotonic)
        unavailable = sum(
            lane.requested_episodes for lane in lanes.values() if lane.status == "unavailable"
        )
        remaining = max(0, requested - attempted - unavailable)
        attempted_this_process = attempted - attempted_at_start
        eta_s = (
            elapsed_s / attempted_this_process * remaining
            if attempted_this_process > 0 and active_episode_index is None
            else None
        )
        atomic_write_json(
            self.directory / "progress.json",
            {
                "schema": PROGRESS_SCHEMA,
                "campaign_identity_digest": self.campaign_identity["digest"],
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "writer_run_id": self.writer_run_id,
                "writer_pid": os.getpid(),
                "snapshot_kind": snapshot_kind,
                "active_controller": active_controller,
                "active_episode_index": active_episode_index,
                "active_episode_seed": active_episode_seed,
                "requested_episodes": requested,
                "attempted_episodes": attempted,
                "completed_episodes": sum(lane.n_episodes for lane in lanes.values()),
                "failed_episodes": sum(lane.failed_episodes for lane in lanes.values()),
                "unavailable_episodes": unavailable,
                "remaining_episodes": remaining,
                "elapsed_s": elapsed_s,
                "eta_s": eta_s,
                "eta_method": (
                    "current_process_completed_episode_mean" if eta_s is not None else None
                ),
                "lanes": {
                    name: {
                        "status": lane.status,
                        "requested_episodes": lane.requested_episodes,
                        "completed_episodes": lane.n_episodes,
                        "failed_episodes": lane.failed_episodes,
                    }
                    for name, lane in lanes.items()
                },
            },
        )

    @contextmanager
    def progress_heartbeat(
        self,
        *,
        lanes: dict[str, ControllerMetrics],
        active_controller: str,
        active_episode_index: int,
        active_episode_seed: int,
        started_monotonic: float,
        attempted_at_start: int,
        interval_s: float = 10.0,
    ) -> Iterator[None]:
        """Refresh an active episode snapshot until its runner returns."""
        if not 0.0 < interval_s < float("inf"):
            raise ValueError("heartbeat interval_s must be finite and > 0")
        stop = threading.Event()
        errors: list[BaseException] = []

        def publish_until_stopped() -> None:
            while not stop.wait(interval_s):
                try:
                    self.write_progress(
                        lanes=lanes,
                        active_controller=active_controller,
                        active_episode_index=active_episode_index,
                        active_episode_seed=active_episode_seed,
                        started_monotonic=started_monotonic,
                        attempted_at_start=attempted_at_start,
                        snapshot_kind="heartbeat",
                    )
                except BaseException as exc:  # pragma: no cover - rare storage failure
                    errors.append(exc)
                    return

        worker = threading.Thread(
            target=publish_until_stopped,
            name=f"stress-heartbeat-{self._lane_slug(active_controller)}",
            daemon=True,
        )
        worker.start()
        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            stop.set()
            worker.join()
            if errors and not body_failed:
                raise RuntimeError("stress-campaign progress heartbeat failed") from errors[0]
