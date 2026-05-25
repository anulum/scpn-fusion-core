#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Generate and validate full reference-data provenance + license manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "validation" / "reference_data"
DEFAULT_POLICY = DEFAULT_ROOT / "provenance_policy.json"
DEFAULT_MANIFEST = DEFAULT_ROOT / "provenance_manifest.json"
_BINARY_SUFFIXES = {".npz"}
REFERENCE_CURATION_FIELDS = (
    "reference_class",
    "reference_role",
    "reference_expected_contract",
    "reference_expected_convention",
)
REFERENCE_EQUILIBRIUM_SOURCE_TYPE = "reference_equilibrium_bundle"
REFERENCE_CLASSES = {"public_efit_reference", "synthetic_proxy_reference"}
REFERENCE_EXPECTED_CONTRACTS = {
    "public_efit_geqdsk_operator_and_profile_source_contract",
    "synthetic_solovev_geqdsk_diagnostic_contract",
}
REFERENCE_EXPECTED_CONVENTIONS = {
    "raw_canonical_strict_unless_named_adapter_passes",
    "synthetic_proxy_profile_source",
}
PUBLIC_EFIT_REFERENCE_PREFIXES = ("sparc/",)


def _content_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return raw
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _sha256(content: bytes) -> str:
    digest = hashlib.sha256()
    for idx in range(0, len(content), 1024 * 1024):
        digest.update(content[idx : idx + 1024 * 1024])
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return payload


def _normalize_license_registry(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_registry = policy.get("license_registry")
    if not isinstance(raw_registry, list) or not raw_registry:
        raise ValueError("provenance policy must contain non-empty 'license_registry' list.")

    out: dict[str, dict[str, Any]] = {}
    for idx, raw in enumerate(raw_registry):
        if not isinstance(raw, dict):
            raise ValueError(f"license_registry entry #{idx} must be a JSON object.")
        license_id = str(raw.get("id", "")).strip()
        if not license_id:
            raise ValueError(f"license_registry entry #{idx} missing non-empty id.")
        if license_id in out:
            raise ValueError(f"duplicate license_registry id: {license_id}")
        redistributable = raw.get("redistributable")
        attribution_required = raw.get("attribution_required")
        citation_required = raw.get("citation_required")
        requires_license_notice = raw.get("requires_license_notice", False)
        if not isinstance(redistributable, bool):
            raise ValueError(f"license_registry entry {license_id} missing bool redistributable.")
        if not isinstance(attribution_required, bool):
            raise ValueError(
                f"license_registry entry {license_id} missing bool attribution_required."
            )
        if not isinstance(citation_required, bool):
            raise ValueError(f"license_registry entry {license_id} missing bool citation_required.")
        if not isinstance(requires_license_notice, bool):
            raise ValueError(
                f"license_registry entry {license_id} missing bool requires_license_notice."
            )
        notes = str(raw.get("notes", "")).strip()
        out[license_id] = {
            "id": license_id,
            "redistributable": redistributable,
            "attribution_required": attribution_required,
            "citation_required": citation_required,
            "requires_license_notice": requires_license_notice,
            "notes": notes,
        }
    return out


def _normalize_rules(
    policy: dict[str, Any], *, license_registry: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_rules = policy.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("provenance policy must contain non-empty 'rules' list.")

    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            raise ValueError(f"rule #{idx} must be a JSON object.")
        dataset_id = str(raw.get("id", "")).strip()
        glob_pat = str(raw.get("glob", "")).strip()
        source = str(raw.get("source", "")).strip()
        license_name = str(raw.get("license", "")).strip()
        source_type = str(raw.get("source_type", "")).strip()
        if not dataset_id:
            raise ValueError(f"rule #{idx} missing non-empty id.")
        if not glob_pat:
            raise ValueError(f"rule {dataset_id} missing non-empty glob.")
        if not source:
            raise ValueError(f"rule {dataset_id} missing non-empty source.")
        if not license_name:
            raise ValueError(f"rule {dataset_id} missing non-empty license.")
        if not source_type:
            raise ValueError(f"rule {dataset_id} missing non-empty source_type.")
        if license_name not in license_registry:
            raise ValueError(
                f"rule {dataset_id} references unknown license '{license_name}'. "
                "Declare it in license_registry."
            )
        citation = str(raw.get("citation", "")).strip()
        license_notice = str(raw.get("license_notice", "")).strip()
        license_meta = license_registry[license_name]
        if bool(license_meta["citation_required"]) and not citation:
            raise ValueError(
                f"rule {dataset_id} uses license '{license_name}' which requires citation."
            )
        if bool(license_meta["requires_license_notice"]) and not license_notice:
            raise ValueError(
                f"rule {dataset_id} uses license '{license_name}' which requires license_notice."
            )
        curation = {field: str(raw.get(field, "")).strip() for field in REFERENCE_CURATION_FIELDS}
        missing_curation = [
            field
            for field, value in curation.items()
            if source_type == REFERENCE_EQUILIBRIUM_SOURCE_TYPE and not value
        ]
        if missing_curation:
            missing = ", ".join(missing_curation)
            raise ValueError(
                f"reference equilibrium rule {dataset_id} missing curation field(s): {missing}"
            )
        if curation["reference_role"] and curation["reference_role"] not in {
            "gate",
            "diagnostic",
        }:
            raise ValueError(f"rule {dataset_id} reference_role must be 'gate' or 'diagnostic'.")
        if source_type == REFERENCE_EQUILIBRIUM_SOURCE_TYPE:
            reference_class = curation["reference_class"]
            reference_role = curation["reference_role"]
            expected_contract = curation["reference_expected_contract"]
            expected_convention = curation["reference_expected_convention"]
            if reference_class not in REFERENCE_CLASSES:
                raise ValueError(f"rule {dataset_id} unknown reference_class: {reference_class}")
            if expected_contract not in REFERENCE_EXPECTED_CONTRACTS:
                raise ValueError(
                    f"rule {dataset_id} unknown reference_expected_contract: {expected_contract}"
                )
            if expected_convention not in REFERENCE_EXPECTED_CONVENTIONS:
                raise ValueError(
                    f"rule {dataset_id} unknown reference_expected_convention: {expected_convention}"
                )
            if (
                reference_class == "public_efit_reference"
                and expected_contract != "public_efit_geqdsk_operator_and_profile_source_contract"
            ):
                raise ValueError(
                    f"rule {dataset_id} public EFIT references require public EFIT contract."
                )
            if reference_class == "public_efit_reference" and reference_role != "gate":
                raise ValueError(f"rule {dataset_id} public EFIT references must be gate rows.")
            if reference_class == "public_efit_reference" and license_name == "synthetic-v1":
                raise ValueError(
                    f"rule {dataset_id} public EFIT references cannot use synthetic license."
                )
            if (
                reference_class == "public_efit_reference"
                and expected_convention != "raw_canonical_strict_unless_named_adapter_passes"
            ):
                raise ValueError(
                    f"rule {dataset_id} public EFIT references require raw canonical convention."
                )
            if reference_class == "synthetic_proxy_reference" and reference_role != "diagnostic":
                raise ValueError(
                    f"rule {dataset_id} synthetic proxy references must be diagnostic."
                )
            if reference_class == "synthetic_proxy_reference" and license_name != "synthetic-v1":
                raise ValueError(
                    f"rule {dataset_id} synthetic proxy references require synthetic license."
                )
            if (
                reference_class == "synthetic_proxy_reference"
                and expected_contract != "synthetic_solovev_geqdsk_diagnostic_contract"
            ):
                raise ValueError(
                    f"rule {dataset_id} synthetic proxy references require synthetic contract."
                )
            if (
                reference_class == "synthetic_proxy_reference"
                and expected_convention != "synthetic_proxy_profile_source"
            ):
                raise ValueError(
                    f"rule {dataset_id} synthetic proxy references require synthetic convention."
                )
            if reference_role == "gate" and license_name == "synthetic-v1":
                raise ValueError(
                    f"rule {dataset_id} gate reference equilibria cannot use synthetic license."
                )
        rule = {
            "id": dataset_id,
            "glob": glob_pat,
            "source": source,
            "license": license_name,
            "source_type": source_type,
            "citation": citation,
            "license_notice": license_notice,
        }
        rule.update(curation)
        out.append(rule)
    return out


def _match_rule(rel_path: str, rules: list[dict[str, str]]) -> dict[str, str]:
    rel = PurePosixPath(rel_path)
    wildcard_chars = {"*", "?", "["}
    matches: list[dict[str, str]] = []
    for rule in rules:
        pattern = rule["glob"]
        if any(ch in pattern for ch in wildcard_chars):
            is_match = rel.match(pattern)
        else:
            is_match = rel_path == pattern
        if is_match:
            matches.append(rule)
    if not matches:
        raise ValueError(f"No provenance policy rule matched file: {rel_path}")
    if len(matches) > 1:
        ids = ", ".join(sorted(rule["id"] for rule in matches))
        raise ValueError(f"Ambiguous provenance policy for {rel_path}: {ids}")
    return matches[0]


def _git_tracked_paths(root: Path) -> set[str] | None:
    """Return repo-tracked file paths relative to ``root`` when available.

    Falls back to ``None`` outside git-tracked contexts so tests/temp roots
    still work without requiring a repository.
    """
    try:
        root_rel = root.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None

    cmd = [
        "git",
        "-C",
        str(REPO_ROOT),
        "ls-files",
        "--",
        root_rel,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None

    tracked: set[str] = set()
    prefix = root_rel.rstrip("/") + "/"
    for line in result.stdout.splitlines():
        entry = line.strip().replace("\\", "/")
        if not entry.startswith(prefix):
            continue
        tracked.add(entry[len(prefix) :])
    return tracked


def build_manifest(
    *,
    root: Path,
    policy_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    policy_payload = _load_json(policy_path, label="provenance policy")
    license_registry = _normalize_license_registry(policy_payload)
    rules = _normalize_rules(policy_payload, license_registry=license_registry)

    root = Path(root)
    manifest_rel = manifest_path.relative_to(root).as_posix()
    policy_rel = policy_path.relative_to(root).as_posix()
    tracked_paths = _git_tracked_paths(root)
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )

    dataset_rows: dict[str, dict[str, Any]] = {}
    file_rows: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel == manifest_rel:
            continue
        if tracked_paths is not None and rel not in tracked_paths:
            continue

        rule = _match_rule(rel, rules)
        license_meta = license_registry[rule["license"]]
        if rule["source_type"] == REFERENCE_EQUILIBRIUM_SOURCE_TYPE:
            is_public_namespace = rel.startswith(PUBLIC_EFIT_REFERENCE_PREFIXES)
            reference_class = rule["reference_class"]
            if reference_class == "public_efit_reference" and not is_public_namespace:
                prefixes = ", ".join(PUBLIC_EFIT_REFERENCE_PREFIXES)
                raise ValueError(
                    f"public EFIT references must live under public reference namespace(s): "
                    f"{prefixes}"
                )
            if reference_class == "synthetic_proxy_reference" and is_public_namespace:
                raise ValueError(
                    "synthetic proxy references cannot live under public reference namespace(s)."
                )
        content = _content_bytes(path)
        size_bytes = int(len(content))
        sha256 = _sha256(content)

        row = {
            "path": rel,
            "dataset_id": rule["id"],
            "source_type": rule["source_type"],
            "source": rule["source"],
            "license": rule["license"],
            "license_redistributable": bool(license_meta["redistributable"]),
            "license_attribution_required": bool(license_meta["attribution_required"]),
            "license_citation_required": bool(license_meta["citation_required"]),
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
        if rule["citation"]:
            row["citation"] = rule["citation"]
        if rule["license_notice"]:
            row["license_notice"] = rule["license_notice"]
        for field in REFERENCE_CURATION_FIELDS:
            if rule[field]:
                row[field] = rule[field]
        file_rows.append(row)

        acc = dataset_rows.get(rule["id"])
        if acc is None:
            acc = {
                "id": rule["id"],
                "source_type": rule["source_type"],
                "source": rule["source"],
                "license": rule["license"],
                "license_redistributable": bool(license_meta["redistributable"]),
                "license_attribution_required": bool(license_meta["attribution_required"]),
                "license_citation_required": bool(license_meta["citation_required"]),
                "file_count": 0,
                "total_bytes": 0,
            }
            if rule["citation"]:
                acc["citation"] = rule["citation"]
            if rule["license_notice"]:
                acc["license_notice"] = rule["license_notice"]
            for field in REFERENCE_CURATION_FIELDS:
                if rule[field]:
                    acc[field] = rule[field]
            dataset_rows[rule["id"]] = acc
        else:
            if str(acc.get("license", "")) != rule["license"]:
                raise ValueError(
                    f"dataset id {rule['id']} maps to multiple licenses: "
                    f"{acc.get('license')} vs {rule['license']}"
                )
            acc_is_equilibrium = (
                str(acc.get("source_type", "")) == REFERENCE_EQUILIBRIUM_SOURCE_TYPE
            )
            rule_is_equilibrium = rule["source_type"] == REFERENCE_EQUILIBRIUM_SOURCE_TYPE
            if acc_is_equilibrium != rule_is_equilibrium:
                raise ValueError(
                    f"dataset id {rule['id']} mixes reference-equilibrium and "
                    "non-equilibrium provenance rows; use separate dataset ids."
                )
            for field in REFERENCE_CURATION_FIELDS:
                if rule[field] and not str(acc.get(field, "")):
                    acc[field] = rule[field]
                elif rule[field] and str(acc.get(field, "")) != rule[field]:
                    raise ValueError(
                        f"dataset id {rule['id']} maps to multiple {field} values: "
                        f"{acc.get(field)} vs {rule[field]}"
                    )
        acc["file_count"] = int(acc["file_count"]) + 1
        acc["total_bytes"] = int(acc["total_bytes"]) + size_bytes

    registry_rows = [
        {
            "id": str(item["id"]),
            "redistributable": bool(item["redistributable"]),
            "attribution_required": bool(item["attribution_required"]),
            "citation_required": bool(item["citation_required"]),
            "requires_license_notice": bool(item["requires_license_notice"]),
            "notes": str(item.get("notes", "")),
        }
        for item in license_registry.values()
    ]

    return {
        "manifest_version": "reference-data-provenance-v2",
        "dataset_root": "validation/reference_data",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_file": policy_rel,
        "policy_version": str(policy_payload.get("policy_version", "")).strip(),
        "license_registry": sorted(registry_rows, key=lambda item: item["id"]),
        "file_count": len(file_rows),
        "datasets": sorted(dataset_rows.values(), key=lambda item: str(item["id"])),
        "files": file_rows,
    }


def render_manifest_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _normalize_for_check(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "generated_at_utc" in out:
        out["generated_at_utc"] = "<normalized>"
    license_registry = out.get("license_registry")
    if isinstance(license_registry, list):
        out["license_registry"] = sorted(
            license_registry,
            key=lambda item: str(item.get("id", "")) if isinstance(item, dict) else "",
        )
    datasets = out.get("datasets")
    if isinstance(datasets, list):
        normalized_datasets: list[dict[str, Any] | Any] = []
        for item in datasets:
            if not isinstance(item, dict):
                normalized_datasets.append(item)
                continue
            row = dict(item)
            if "total_bytes" in row:
                row["total_bytes"] = "<normalized>"
            normalized_datasets.append(row)
        out["datasets"] = sorted(
            normalized_datasets,
            key=lambda item: str(item.get("id", "")) if isinstance(item, dict) else "",
        )
    files = out.get("files")
    if isinstance(files, list):
        normalized_files: list[dict[str, Any] | Any] = []
        for item in files:
            if not isinstance(item, dict):
                normalized_files.append(item)
                continue
            row = dict(item)
            if "size_bytes" in row:
                row["size_bytes"] = "<normalized>"
            if "sha256" in row:
                row["sha256"] = "<normalized>"
            normalized_files.append(row)
        out["files"] = sorted(
            normalized_files,
            key=lambda item: str(item.get("path", "")) if isinstance(item, dict) else "",
        )
    return out


def _emit_stale_diff_summary(
    existing_payload: dict[str, Any],
    generated_payload: dict[str, Any],
) -> None:
    existing = _normalize_for_check(existing_payload)
    generated = _normalize_for_check(generated_payload)

    print("Reference-data provenance mismatch summary:")
    for key in ("dataset_root", "policy_file", "file_count"):
        if existing.get(key) != generated.get(key):
            print(f"- {key}: existing={existing.get(key)!r} generated={generated.get(key)!r}")

    def _index(rows: Any, *, key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get(key)
            if isinstance(value, str) and value:
                out[value] = row
        return out

    existing_datasets = _index(existing.get("datasets"), key="id")
    generated_datasets = _index(generated.get("datasets"), key="id")
    missing_datasets = sorted(set(existing_datasets) - set(generated_datasets))
    added_datasets = sorted(set(generated_datasets) - set(existing_datasets))
    if missing_datasets:
        print(f"- missing datasets ({len(missing_datasets)}): {missing_datasets[:5]}")
    if added_datasets:
        print(f"- added datasets ({len(added_datasets)}): {added_datasets[:5]}")

    existing_files = _index(existing.get("files"), key="path")
    generated_files = _index(generated.get("files"), key="path")
    missing_paths = sorted(set(existing_files) - set(generated_files))
    added_paths = sorted(set(generated_files) - set(existing_files))
    if missing_paths:
        print(f"- missing files ({len(missing_paths)}): {missing_paths[:8]}")
    if added_paths:
        print(f"- added files ({len(added_paths)}): {added_paths[:8]}")

    for path in sorted(set(existing_files) & set(generated_files)):
        before = existing_files[path]
        after = generated_files[path]
        if before == after:
            continue
        print(f"- first differing file row: {path}")
        for field in sorted(set(before) | set(after)):
            if before.get(field) == after.get(field):
                continue
            print(f"  - {field}: existing={before.get(field)!r} generated={after.get(field)!r}")
        break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Reference-data root directory.")
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY),
        help="Reference-data provenance policy JSON path.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Output/input provenance manifest JSON path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when generated content differs from existing manifest.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    policy_path = Path(args.policy)
    manifest_path = Path(args.manifest)
    if not root.is_absolute():
        root = REPO_ROOT / root
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    if not root.exists():
        raise FileNotFoundError(f"Reference-data root not found: {root}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Provenance policy not found: {policy_path}")
    if policy_path.parent != root:
        raise ValueError("Policy file must live under the reference-data root.")
    if manifest_path.parent != root:
        raise ValueError("Manifest file must live under the reference-data root.")

    manifest_payload = build_manifest(
        root=root,
        policy_path=policy_path,
        manifest_path=manifest_path,
    )
    rendered = render_manifest_json(manifest_payload)

    if args.check:
        if not manifest_path.exists():
            print(f"Reference-data provenance manifest missing: {manifest_path}")
            return 1
        existing_payload = _load_json(manifest_path, label="provenance manifest")
        if _normalize_for_check(existing_payload) != _normalize_for_check(manifest_payload):
            print(
                "Reference-data provenance manifest is stale. "
                "Run tools/generate_reference_data_provenance_manifest.py to refresh."
            )
            _emit_stale_diff_summary(existing_payload, manifest_payload)
            return 1
        print(f"Reference-data provenance manifest is up to date: {manifest_path}")
        return 0

    manifest_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote reference-data provenance manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
