# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Recoverable Official TGLF Development Corpus
"""Deterministic sampling, recovery and verification for official TGLF corpora."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
from typing import Any, Final, cast

from scpn_fusion.core._tglf_interface_runtime import (
    _parse_gacode_tglf_output,
    run_tglf_binary,
)
from scpn_fusion.io.safe_loaders import checked_json_load
from scpn_fusion.io.tglf_dataset_contract import REQUIRED_TGLF_RAW_FILES, sha256_file
from scpn_fusion.io.tglf_development_plan import (
    TGLF_DEVELOPMENT_DESIGN_METHOD,
    TGLF_DEVELOPMENT_GACODE_REVISION,
    TGLF_DEVELOPMENT_PLAN_VERSION,
    TGLF_DEVELOPMENT_SEED,
    build_tglf_development_plan,
    canonical_tglf_development_digest,
    validate_tglf_development_plan,
)
from scpn_fusion.io.tglf_species_dataset_contract import (
    TGLF_DEVELOPMENT_METADATA_VERSION,
    build_tglf_species_dataset_manifest,
    tglf_species_deck_from_payload,
    tglf_species_output_payload,
    verify_tglf_species_dataset,
    write_tglf_species_dataset_manifest,
)

TGLF_DEVELOPMENT_RECOVERY_VERSION: Final = "scpn-fusion.tglf-development-recovery.v1"
TGLF_DEVELOPMENT_LOCAL_MAX_BYTES: Final = 256 * 1024 * 1024
TGLF_DEVELOPMENT_LOCAL_MAX_RUNS: Final = 1_000
_MAX_PLAN_BYTES: Final = 16 * 1024 * 1024


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _run_inventory(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"unsafe or missing run directory: {directory}")
    inventory: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"unsafe run entry: {path}")
        inventory.append(
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    names = {item["name"] for item in inventory}
    if not REQUIRED_TGLF_RAW_FILES.issubset(names):
        raise ValueError(f"run is missing required GACODE files: {directory.name}")
    return inventory


def _run_digest(directory: Path) -> str:
    return canonical_tglf_development_digest(_run_inventory(directory))


def _record_from_run(run: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    deck = tglf_species_deck_from_payload(cast(Mapping[str, Any], run["input"]))
    output = _parse_gacode_tglf_output(directory, deck)
    return {
        "sample_index": run["sample_index"],
        "group_id": run["group_id"],
        "sampling_stratum": run["sampling_stratum"],
        "composition": run["composition"],
        "paired_gradient_species": run["paired_gradient_species"],
        "regime": "unclassified",
        "input": run["input"],
        "output": tglf_species_output_payload(output),
    }


def _new_recovery(
    output: Path,
    plan: Mapping[str, Any],
    *,
    command: str,
    timeout_s: float,
    max_retries: int,
) -> dict[str, Any]:
    return {
        "schema_version": TGLF_DEVELOPMENT_RECOVERY_VERSION,
        "output_name": output.name,
        "plan_sha256": plan["plan_sha256"],
        "command_policy": {
            "command": command,
            "timeout_s": timeout_s,
            "max_retries": max_retries,
        },
        "next_sample_index": 0,
        "records_sha256": canonical_tglf_development_digest([]),
        "committed_runs": [],
    }


def _load_recovery(
    path: Path,
    *,
    output: Path,
    plan: Mapping[str, Any],
    command: str,
    timeout_s: float,
    max_retries: int,
) -> dict[str, Any]:
    raw = checked_json_load(path, max_bytes=_MAX_PLAN_BYTES)
    if not isinstance(raw, dict):
        raise ValueError("recovery checkpoint must be an object")
    recovery = cast(dict[str, Any], raw)
    expected = _new_recovery(
        output, plan, command=command, timeout_s=timeout_s, max_retries=max_retries
    )
    for key in ("schema_version", "output_name", "plan_sha256", "command_policy"):
        if recovery.get(key) != expected[key]:
            raise ValueError(f"recovery {key} differs from this invocation")
    next_index = recovery.get("next_sample_index")
    committed = recovery.get("committed_runs")
    if isinstance(next_index, bool) or not isinstance(next_index, int) or next_index < 0:
        raise ValueError("recovery next_sample_index is invalid")
    if not isinstance(committed, list) or len(committed) != next_index:
        raise ValueError("recovery committed-run prefix is invalid")
    return recovery


def _authenticate_prefix(
    partial: Path,
    plan: Mapping[str, Any],
    recovery: dict[str, Any],
) -> list[dict[str, Any]]:
    records_path = partial / "dataset.json"
    raw_records = checked_json_load(records_path, max_bytes=_MAX_PLAN_BYTES)
    if not isinstance(raw_records, list):
        raise ValueError("partial dataset records must be an array")
    records = cast(list[dict[str, Any]], raw_records)
    next_index = cast(int, recovery["next_sample_index"])
    if len(records) not in {next_index, next_index + 1}:
        raise ValueError("partial records are not a recoverable committed prefix")
    for index, committed in enumerate(cast(list[dict[str, Any]], recovery["committed_runs"])):
        directory = partial / "runs" / f"sample_{index:06d}"
        if committed != {"sample_index": index, "run_sha256": _run_digest(directory)}:
            raise ValueError("committed run SHA-256 differs from recovery checkpoint")
        expected = cast(list[dict[str, Any]], plan["runs"])[index]
        if records[index] != _record_from_run(expected, directory):
            raise ValueError("committed record differs from retained official output")
    if recovery.get("records_sha256") != canonical_tglf_development_digest(records[:next_index]):
        raise ValueError("recovery records SHA-256 mismatch")
    return records


def _checkpoint(recovery_path: Path, recovery: dict[str, Any], records: Sequence[Any]) -> None:
    recovery["next_sample_index"] = len(records)
    recovery["records_sha256"] = canonical_tglf_development_digest(list(records))
    _atomic_json(recovery_path, recovery)


def _adopt_completed_run(
    partial: Path,
    recovery_path: Path,
    recovery: dict[str, Any],
    plan: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> None:
    index = cast(int, recovery["next_sample_index"])
    if index >= cast(int, plan["accepted_runs"]):
        return
    directory = partial / "runs" / f"sample_{index:06d}"
    if not directory.exists():
        return
    expected = cast(list[dict[str, Any]], plan["runs"])[index]
    _run_inventory(directory)
    record = _record_from_run(expected, directory)
    if len(records) == index:
        records.append(record)
        _atomic_json(partial / "dataset.json", records)
    elif records[index] != record:
        raise ValueError("recoverable completed record differs from retained output")
    cast(list[dict[str, Any]], recovery["committed_runs"]).append(
        {"sample_index": index, "run_sha256": _run_digest(directory)}
    )
    _checkpoint(recovery_path, recovery, records)


def _development_metadata(
    plan: Mapping[str, Any], command_policy: Mapping[str, Any]
) -> dict[str, Any]:
    runs = cast(list[dict[str, Any]], plan["runs"])
    return {
        "schema_version": TGLF_DEVELOPMENT_METADATA_VERSION,
        "design_method": plan["design_method"],
        "plan_sha256": plan["plan_sha256"],
        "accepted_runs": len(runs),
        "base_groups": plan["base_groups"],
        "samples_per_group": 3,
        "sampling_strata_counts": dict(
            sorted(Counter(run["sampling_stratum"] for run in runs).items())
        ),
        "composition_counts": dict(sorted(Counter(run["composition"] for run in runs).items())),
        "command_policy": dict(command_policy),
        "storage_contract": {
            "working_location": "ignored-local-workstation",
            "local_max_bytes": TGLF_DEVELOPMENT_LOCAL_MAX_BYTES,
            "local_max_runs": TGLF_DEVELOPMENT_LOCAL_MAX_RUNS,
            "large_artifact_policy": "owner-controlled-storage-manifest-only-in-git",
        },
    }


def _tree_digest(root: Path) -> tuple[str, int]:
    inventory: list[dict[str, Any]] = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"dataset contains an unsafe entry: {path}")
        size = path.stat().st_size
        total += size
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )
    return canonical_tglf_development_digest(inventory), total


def verify_tglf_development_corpus(dataset_root: str | Path) -> dict[str, Any]:
    """Verify deterministic plan replay and the complete official raw-file tree.

    Parameters
    ----------
    dataset_root : str or Path
        Final corpus directory containing plan, rejection, records and v2 manifest files.

    Returns
    -------
    dict[str, Any]
        Pass/fail result with deterministic plan and full-tree digests.
    """
    root = Path(dataset_root)
    base = verify_tglf_species_dataset(root)
    if base["status"] != "passed":
        return {**base, "plan_replay": False}
    failures: list[str] = []
    try:
        manifest_raw = checked_json_load(root / "manifest.json", max_bytes=_MAX_PLAN_BYTES)
        plan_raw = checked_json_load(root / "plan.json", max_bytes=_MAX_PLAN_BYTES)
        rejections = checked_json_load(root / "rejections.json", max_bytes=_MAX_PLAN_BYTES)
        records = checked_json_load(root / "dataset.json", max_bytes=_MAX_PLAN_BYTES)
        if not all(
            isinstance(value, (dict, list))
            for value in (manifest_raw, plan_raw, rejections, records)
        ):
            raise ValueError("development corpus JSON roots are invalid")
        manifest = cast(dict[str, Any], manifest_raw)
        plan = validate_tglf_development_plan(cast(dict[str, Any], plan_raw))
        if rejections != plan["negative_controls"]:
            raise ValueError("rejections differ from the frozen negative controls")
        runs = cast(list[dict[str, Any]], plan["runs"])
        record_list = cast(list[dict[str, Any]], records)
        if len(record_list) != len(runs):
            raise ValueError("record count differs from deterministic plan")
        for index, (run, record) in enumerate(zip(runs, record_list, strict=True)):
            expected = {key: run[key] for key in run}
            observed = {key: record[key] for key in expected}
            if observed != expected:
                raise ValueError(f"record {index} input/group metadata differs from plan")
        allowed_root = {"dataset.json", "manifest.json", "plan.json", "rejections.json", "runs"}
        if {path.name for path in root.iterdir()} != allowed_root:
            raise ValueError("development corpus root inventory is not exact")
        tree_sha256, total_bytes = _tree_digest(root)
        if total_bytes > cast(int, manifest["development"]["storage_contract"]["local_max_bytes"]):
            raise ValueError("development corpus exceeds its frozen local storage boundary")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
        tree_sha256, total_bytes = "", 0
    if failures:
        return {**base, "status": "failed", "failures": failures, "plan_replay": False}
    return {
        **base,
        "plan_replay": True,
        "negative_controls_verified": len(cast(list[Any], rejections)),
        "tree_sha256": tree_sha256,
        "total_bytes": total_bytes,
    }


def generate_tglf_development_corpus(
    output_dir: str | Path,
    *,
    seed: int = TGLF_DEVELOPMENT_SEED,
    profile: str = "development",
    command: str = "tglf",
    timeout_s: float = 120.0,
    max_retries: int = 2,
    resume: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    """Generate or resume one official-GACODE development corpus atomically.

    Parameters
    ----------
    output_dir : str or Path
        Final output directory. Work is isolated in a sibling ``.partial`` tree.
    seed : int, optional
        Frozen deterministic plan seed.
    profile : {"development", "expanded", "fixture"}, optional
        Full 72-run development corpus, 216-run expanded selection corpus, or
        nine-run authentic contract fixture.
    command : str, optional
        PATH-resolved executable name; only ``tglf`` is admitted by the manifest.
    timeout_s : float, optional
        Positive per-attempt timeout in seconds.
    max_retries : int, optional
        Bounded production retry count.
    resume : bool, optional
        Authenticate and continue an existing partial tree.
    max_runs : int, optional
        Testing/operator checkpoint boundary for this invocation only.

    Returns
    -------
    dict[str, Any]
        Partial progress or the final independent verification result.
    """
    output = Path(output_dir)
    partial = output.with_name(f"{output.name}.partial")
    recovery_path = output.with_name(f"{output.name}.recovery.json")
    if command != "tglf":
        raise ValueError("development corpus requires the PATH command name tglf")
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if max_runs is not None and (
        isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 0
    ):
        raise ValueError("max_runs must be a non-negative integer")
    if output.exists():
        raise FileExistsError(f"final corpus already exists: {output}")
    plan = build_tglf_development_plan(seed=seed, profile=profile)
    command_policy = {"command": command, "timeout_s": timeout, "max_retries": max_retries}
    if resume:
        if not partial.is_dir() or partial.is_symlink() or not recovery_path.is_file():
            raise FileNotFoundError("resume requires the partial corpus and recovery checkpoint")
        stored_plan = checked_json_load(partial / "plan.json", max_bytes=_MAX_PLAN_BYTES)
        if not isinstance(stored_plan, dict) or validate_tglf_development_plan(stored_plan) != plan:
            raise ValueError("partial plan differs from deterministic regeneration")
        recovery = _load_recovery(
            recovery_path,
            output=output,
            plan=plan,
            command=command,
            timeout_s=timeout,
            max_retries=max_retries,
        )
    else:
        if partial.exists() or recovery_path.exists():
            raise FileExistsError("partial corpus or recovery checkpoint already exists")
        partial.mkdir(parents=False)
        (partial / "runs").mkdir()
        _atomic_json(partial / "plan.json", plan)
        _atomic_json(partial / "rejections.json", plan["negative_controls"])
        _atomic_json(partial / "dataset.json", [])
        recovery = _new_recovery(
            output, plan, command=command, timeout_s=timeout, max_retries=max_retries
        )
        _atomic_json(recovery_path, recovery)
    records = _authenticate_prefix(partial, plan, recovery)
    _adopt_completed_run(partial, recovery_path, recovery, plan, records)
    started = len(records)
    run_limit = len(cast(list[Any], plan["runs"]))
    if max_runs is not None:
        run_limit = min(run_limit, started + max_runs)
    for index in range(started, run_limit):
        run = cast(list[dict[str, Any]], plan["runs"])[index]
        deck = tglf_species_deck_from_payload(cast(Mapping[str, Any], run["input"]))
        temporary = partial / "runs" / f".sample_{index:06d}.tmp"
        final = partial / "runs" / f"sample_{index:06d}"
        if final.exists():
            raise ValueError(f"unexpected completed run outside recovery prefix: {final.name}")
        temporary.mkdir(exist_ok=True)
        run_tglf_binary(
            deck,
            tglf_command=command,
            timeout_s=timeout,
            work_dir=temporary,
            max_retries=max_retries,
        )
        _run_inventory(temporary)
        os.replace(temporary, final)
        _sync_directory(final.parent)
        record = _record_from_run(run, final)
        records.append(record)
        _atomic_json(partial / "dataset.json", records)
        cast(list[dict[str, Any]], recovery["committed_runs"]).append(
            {"sample_index": index, "run_sha256": _run_digest(final)}
        )
        _checkpoint(recovery_path, recovery, records)
    if len(records) < cast(int, plan["accepted_runs"]):
        return {
            "status": "partial",
            "output_dir": str(output),
            "partial_dir": str(partial),
            "accepted_runs": len(records),
            "required_runs": plan["accepted_runs"],
            "plan_sha256": plan["plan_sha256"],
        }
    manifest = build_tglf_species_dataset_manifest(
        partial,
        records,
        dataset_id=(f"gacode-b4933975-{profile}-v2-n{plan['accepted_runs']}-seed{seed}"),
        gacode_revision=TGLF_DEVELOPMENT_GACODE_REVISION,
        seed=seed,
        development=_development_metadata(plan, command_policy),
        plan_file="plan.json",
        rejections_file="rejections.json",
    )
    write_tglf_species_dataset_manifest(partial, manifest)
    verified = verify_tglf_development_corpus(partial)
    if verified["status"] != "passed":
        raise RuntimeError(f"completed development corpus failed verification: {verified}")
    os.replace(partial, output)
    _sync_directory(output.parent)
    recovery_path.unlink()
    _sync_directory(recovery_path.parent)
    return verify_tglf_development_corpus(output)


__all__ = [
    "TGLF_DEVELOPMENT_DESIGN_METHOD",
    "TGLF_DEVELOPMENT_GACODE_REVISION",
    "TGLF_DEVELOPMENT_PLAN_VERSION",
    "TGLF_DEVELOPMENT_RECOVERY_VERSION",
    "TGLF_DEVELOPMENT_SEED",
    "build_tglf_development_plan",
    "generate_tglf_development_corpus",
    "verify_tglf_development_corpus",
]
