#!/usr/bin/env python
"""Generate and validate full reference-data provenance + license manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "validation" / "reference_data"
DEFAULT_POLICY = DEFAULT_ROOT / "provenance_policy.json"
DEFAULT_MANIFEST = DEFAULT_ROOT / "provenance_manifest.json"
_BINARY_SUFFIXES = {".npz"}


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


def _normalize_rules(policy: dict[str, Any]) -> list[dict[str, str]]:
    raw_rules = policy.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("provenance policy must contain non-empty 'rules' list.")

    out: list[dict[str, str]] = []
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
        out.append(
            {
                "id": dataset_id,
                "glob": glob_pat,
                "source": source,
                "license": license_name,
                "source_type": source_type,
            }
        )
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


def build_manifest(
    *,
    root: Path,
    policy_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    policy_payload = _load_json(policy_path, label="provenance policy")
    rules = _normalize_rules(policy_payload)

    root = Path(root)
    manifest_rel = manifest_path.relative_to(root).as_posix()
    policy_rel = policy_path.relative_to(root).as_posix()
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

        rule = _match_rule(rel, rules)
        content = _content_bytes(path)
        size_bytes = int(len(content))
        sha256 = _sha256(content)

        row = {
            "path": rel,
            "dataset_id": rule["id"],
            "source_type": rule["source_type"],
            "source": rule["source"],
            "license": rule["license"],
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
        file_rows.append(row)

        acc = dataset_rows.get(rule["id"])
        if acc is None:
            acc = {
                "id": rule["id"],
                "source_type": rule["source_type"],
                "source": rule["source"],
                "license": rule["license"],
                "file_count": 0,
                "total_bytes": 0,
            }
            dataset_rows[rule["id"]] = acc
        acc["file_count"] = int(acc["file_count"]) + 1
        acc["total_bytes"] = int(acc["total_bytes"]) + size_bytes

    return {
        "manifest_version": "reference-data-provenance-v1",
        "dataset_root": "validation/reference_data",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_file": policy_rel,
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
            print(
                f"  - {field}: existing={before.get(field)!r} generated={after.get(field)!r}"
            )
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
