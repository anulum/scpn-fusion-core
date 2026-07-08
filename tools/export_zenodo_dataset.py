#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Export local FAIR validation packs and their readiness report.

The T-3 FAIR data lane is intentionally local-first. This tool builds
redistribution-ready pack directories from tracked validation surfaces, writes
per-pack checksum manifests, and refreshes a repository report that states the
publication boundary. It does not mint a DOI or upload to Zenodo; external
publication remains an owner-gated action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "artifacts" / "fair_validation_packs"
DEFAULT_JSON_REPORT: Final[Path] = (
    REPO_ROOT / "validation" / "reports" / "fair_validation_packs.json"
)
DEFAULT_MD_REPORT: Final[Path] = REPO_ROOT / "validation" / "reports" / "fair_validation_packs.md"
READINESS_SCHEMA: Final[str] = "scpn-fusion-core.fair-validation-packs.v1"
PACK_SCHEMA: Final[str] = "scpn-fusion-core.fair-validation-pack.v1"
MINIMUM_PACK_COUNT: Final[int] = 3


@dataclass(frozen=True)
class PackDefinition:
    """A deterministic local FAIR validation pack definition."""

    pack_id: str
    title: str
    description: str
    license_id: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class FileRecord:
    """Checksum and size metadata for a repository-relative file."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PackManifest:
    """Machine-readable manifest for one exported validation pack."""

    definition: PackDefinition
    files: tuple[FileRecord, ...]

    @property
    def file_count(self) -> int:
        """Return the number of files included in the pack."""
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        """Return the total size of all files included in the pack."""
        return sum(record.size_bytes for record in self.files)


PACKS: Final[tuple[PackDefinition, ...]] = (
    PackDefinition(
        pack_id="safety_traceability",
        title="Safety Traceability Matrix Pack",
        description=(
            "Requirement, hazard, implementation, proof, test, and generated matrix "
            "evidence for the control-safety traceability gate."
        ),
        license_id="AGPL-3.0-or-later",
        files=(
            "validation/safety_traceability.json",
            "docs/SAFETY_TRACEABILITY_MATRIX.md",
            "tools/generate_safety_traceability.py",
            "tests/test_safety_traceability.py",
        ),
    ),
    PackDefinition(
        pack_id="surrogate_uq_cards",
        title="Surrogate UQ and OOD Cards Pack",
        description=(
            "Per-surrogate uncertainty, out-of-distribution, fallback, and "
            "retraining-boundary evidence for promoted and scoped surrogate lanes."
        ),
        license_id="AGPL-3.0-or-later",
        files=(
            "validation/surrogate_uq_cards.json",
            "docs/SURROGATE_UQ_CARDS.md",
            "tools/generate_surrogate_uq_cards.py",
            "tests/test_surrogate_uq_cards.py",
        ),
    ),
    PackDefinition(
        pack_id="inverse_equilibrium_attribution",
        title="Inverse-Equilibrium Attribution Pack",
        description=(
            "Solov'ev exact-suite evidence and psi reconstruction attribution "
            "documents for the inverse-equilibrium credibility boundary."
        ),
        license_id="AGPL-3.0-or-later",
        files=(
            "validation/validate_grad_shafranov_solovev.py",
            "validation/reports/grad_shafranov_solovev.json",
            "validation/reports/grad_shafranov_solovev.md",
            "docs/PSI_GATE_ATTRIBUTION.md",
            "tools/generate_psi_gate_attribution.py",
            "tests/test_grad_shafranov_solovev.py",
            "tests/test_psi_gate_attribution.py",
        ),
    ),
)


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {path}") from exc


def _resolve_pack_file(root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"pack path must stay inside the repository: {rel_path}")
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"pack input missing: {rel_path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pack_manifest(definition: PackDefinition, root: Path = REPO_ROOT) -> PackManifest:
    """Build checksum metadata for one pack definition.

    Args:
        definition: Pack definition to evaluate.
        root: Repository root used for resolving file anchors.

    Returns:
        The deterministic pack manifest.
    """
    records = tuple(
        FileRecord(
            path=rel_path,
            sha256=_sha256(_resolve_pack_file(root, rel_path)),
            size_bytes=_resolve_pack_file(root, rel_path).stat().st_size,
        )
        for rel_path in definition.files
    )
    return PackManifest(definition=definition, files=records)


def pack_manifest_payload(manifest: PackManifest) -> dict[str, object]:
    """Return a JSON-serializable manifest payload for one pack."""
    return {
        "schema": PACK_SCHEMA,
        "pack_id": manifest.definition.pack_id,
        "title": manifest.definition.title,
        "description": manifest.definition.description,
        "license": manifest.definition.license_id,
        "file_count": manifest.file_count,
        "total_bytes": manifest.total_bytes,
        "files": [
            {
                "path": record.path,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
            }
            for record in manifest.files
        ],
    }


def render_pack_readme(manifest: PackManifest) -> str:
    """Render a README for one exported validation pack."""
    lines = [
        f"# {manifest.definition.title}",
        "",
        manifest.definition.description,
        "",
        f"- Pack id: `{manifest.definition.pack_id}`",
        f"- License: `{manifest.definition.license_id}`",
        f"- Files: `{manifest.file_count}`",
        f"- Total bytes: `{manifest.total_bytes}`",
        "- Publication boundary: local validation pack only; DOI minting is not performed by this export.",
        "",
        "## Files",
        "",
        "| Path | Size bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    for record in manifest.files:
        lines.append(f"| `{record.path}` | {record.size_bytes} | `{record.sha256}` |")
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "Run `python tools/export_zenodo_dataset.py` from the repository root.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_pack(manifest: PackManifest, output_dir: Path, root: Path = REPO_ROOT) -> Path:
    """Copy pack files and manifests into one deterministic output directory.

    Args:
        manifest: Pack manifest to export.
        output_dir: Parent output directory for all packs.
        root: Repository root used for source-file resolution.

    Returns:
        Path to the exported pack directory.
    """
    pack_dir = output_dir / manifest.definition.pack_id
    files_dir = pack_dir / "files"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    files_dir.mkdir(parents=True, exist_ok=True)
    for record in manifest.files:
        src = _resolve_pack_file(root, record.path)
        dst = files_dir / record.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (pack_dir / "pack_manifest.json").write_text(
        json.dumps(pack_manifest_payload(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (pack_dir / "README.md").write_text(render_pack_readme(manifest), encoding="utf-8")
    return pack_dir


def build_readiness_report(
    *,
    manifests: tuple[PackManifest, ...],
    output_dir: Path,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Build the top-level FAIR validation-pack readiness report.

    Args:
        manifests: Pack manifests included in the local export.
        output_dir: Pack output directory.
        root: Repository root used to render relative output paths.

    Returns:
        JSON-serializable readiness report.
    """
    accepted = len(manifests) >= MINIMUM_PACK_COUNT and all(
        manifest.file_count > 0 and manifest.total_bytes > 0 for manifest in manifests
    )
    blockers: list[str] = []
    if len(manifests) < MINIMUM_PACK_COUNT:
        blockers.append(f"at least {MINIMUM_PACK_COUNT} local packs required")
    if not accepted:
        blockers.append("one or more pack manifests are empty")
    publication_blockers = [
        "owner approval for DOI publication and final data-license posture",
        "external Zenodo upload not executed by this local export",
    ]
    return {
        "schema": READINESS_SCHEMA,
        "status": "accepted_local_fair_pack_readiness"
        if accepted
        else "blocked_local_pack_readiness",
        "accepted_local_pack_readiness": accepted,
        "doi_publication_ready": False,
        "minimum_pack_count": MINIMUM_PACK_COUNT,
        "pack_count": len(manifests),
        "total_files": sum(manifest.file_count for manifest in manifests),
        "total_bytes": sum(manifest.total_bytes for manifest in manifests),
        "output_dir": _repo_relative(output_dir, root)
        if output_dir.is_relative_to(root)
        else str(output_dir),
        "publication_blockers": publication_blockers,
        "local_blockers": blockers,
        "packs": [pack_manifest_payload(manifest) for manifest in manifests],
    }


def render_readiness_markdown(report: dict[str, object]) -> str:
    """Render the top-level readiness report as Markdown."""
    packs = report["packs"]
    if not isinstance(packs, list):
        raise ValueError("readiness report packs must be a list")
    lines = [
        "# FAIR Validation Pack Readiness",
        "",
        "This report is generated by `tools/export_zenodo_dataset.py`.",
        "It covers local pack readiness only; DOI publication remains owner-gated.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Accepted local pack readiness: `{report['accepted_local_pack_readiness']}`",
        f"- DOI publication ready: `{report['doi_publication_ready']}`",
        f"- Output directory: `{report['output_dir']}`",
        f"- Packs: `{report['pack_count']}`",
        f"- Files: `{report['total_files']}`",
        f"- Total bytes: `{report['total_bytes']}`",
        "",
        "## Publication blockers",
        "",
    ]
    publication_blockers = report["publication_blockers"]
    if not isinstance(publication_blockers, list):
        raise ValueError("publication_blockers must be a list")
    lines.extend(f"- {blocker}" for blocker in publication_blockers)
    local_blockers = report["local_blockers"]
    if not isinstance(local_blockers, list):
        raise ValueError("local_blockers must be a list")
    if local_blockers:
        lines.extend(["", "## Local blockers", ""])
        lines.extend(f"- {blocker}" for blocker in local_blockers)
    lines.extend(
        [
            "",
            "## Packs",
            "",
            "| Pack | Status | Files | Bytes | License |",
            "|---|---|---:|---:|---|",
        ]
    )
    for item in packs:
        if not isinstance(item, dict):
            raise ValueError("pack rows must be objects")
        lines.append(
            f"| `{item['pack_id']}` | `ready_local` | {item['file_count']} | "
            f"{item['total_bytes']} | `{item['license']}` |"
        )
    return "\n".join(lines) + "\n"


def write_readiness_reports(
    report: dict[str, object],
    *,
    json_report: Path = DEFAULT_JSON_REPORT,
    md_report: Path = DEFAULT_MD_REPORT,
) -> None:
    """Write JSON and Markdown readiness reports."""
    json_report.parent.mkdir(parents=True, exist_ok=True)
    md_report.parent.mkdir(parents=True, exist_ok=True)
    json_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_report.write_text(render_readiness_markdown(report), encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def check_readiness_reports(
    report: dict[str, object],
    *,
    json_report: Path = DEFAULT_JSON_REPORT,
    md_report: Path = DEFAULT_MD_REPORT,
) -> list[str]:
    """Return drift errors for the tracked readiness reports."""
    expected_json = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    expected_md = render_readiness_markdown(report)
    errors: list[str] = []
    if _read_text(json_report) != expected_json:
        errors.append(f"stale JSON report: {json_report}")
    if _read_text(md_report) != expected_md:
        errors.append(f"stale Markdown report: {md_report}")
    return errors


def build_all_manifests(root: Path = REPO_ROOT) -> tuple[PackManifest, ...]:
    """Return manifests for all configured FAIR validation packs."""
    return tuple(build_pack_manifest(definition, root=root) for definition in PACKS)


def run_export(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    json_report: Path = DEFAULT_JSON_REPORT,
    md_report: Path = DEFAULT_MD_REPORT,
    export_files: bool = True,
    check: bool = False,
    root: Path = REPO_ROOT,
) -> int:
    """Run the FAIR pack export or drift check.

    Args:
        output_dir: Parent directory for pack exports.
        json_report: Top-level JSON readiness report path.
        md_report: Top-level Markdown readiness report path.
        export_files: Whether to copy pack contents into ``output_dir``.
        check: Whether to drift-check reports instead of writing them.
        root: Repository root for tests and alternate checkouts.

    Returns:
        Process-style status code.
    """
    manifests = build_all_manifests(root=root)
    if export_files and not check:
        output_dir.mkdir(parents=True, exist_ok=True)
        for manifest in manifests:
            export_pack(manifest, output_dir=output_dir, root=root)
    report = build_readiness_report(manifests=manifests, output_dir=output_dir, root=root)
    if check:
        errors = check_readiness_reports(report, json_report=json_report, md_report=md_report)
        for error in errors:
            print(error)
        return 1 if errors else 0
    write_readiness_reports(report, json_report=json_report, md_report=md_report)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the FAIR pack exporter CLI.

    Args:
        argv: Optional CLI arguments. If omitted, process arguments are used.

    Returns:
        Process-style status code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where local validation pack folders are exported.",
    )
    parser.add_argument(
        "--json-report",
        default=str(DEFAULT_JSON_REPORT),
        help="Tracked JSON readiness report path.",
    )
    parser.add_argument(
        "--md-report",
        default=str(DEFAULT_MD_REPORT),
        help="Tracked Markdown readiness report path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if tracked readiness reports are stale.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Refresh or check reports without copying pack contents.",
    )
    args = parser.parse_args(argv)
    return run_export(
        output_dir=Path(args.output_dir),
        json_report=Path(args.json_report),
        md_report=Path(args.md_report),
        export_files=not args.no_export,
        check=bool(args.check),
    )


if __name__ == "__main__":
    raise SystemExit(main())
