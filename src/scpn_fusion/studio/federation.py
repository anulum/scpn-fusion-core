# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Studio federation document (schema_a core + architecture_map extension)
"""The FUSION studio's federation document for STUDIO/Hub ingestion.

The federation document is one JSON with two blocks, per the locked fleet convention:

* ``schema_a`` — the platform :class:`~scpn_studio_platform.manifest.CapabilityManifest`
  (verbs, evidence schemas, content digest). This is the federation contract the Hub
  ingests; its vocabulary is the locked SDK enums emitted verbatim.
* ``architecture_map`` — an additive superset block the Hub ignores for federation but
  the architecture docs consume: the per-stage IO pipeline, the capability inventory,
  the backend/dispatch matrix, the interface surface, the cross-repo wire formats, and
  the honest scope boundaries. The field set is the fleet ``architecture-map.v2`` schema
  (peer-aligned with SC-NEUROCORE and QUANTUM, 2026-06-24); it mirrors ``docs/ARCHITECTURE.md``.

This is emitted to a dedicated file so it never collides with the repository
inventory manifest (``docs/_generated/capability_manifest.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .manifest import build_manifest

#: Where the federation document is written, relative to the repository root.
STUDIO_MANIFEST_PATH = Path("docs/_generated/studio_manifest.json")

#: The fleet architecture-map extension schema version (peer-aligned with the fleet).
ARCHITECTURE_MAP_VERSION = "architecture-map.v2"


def _pipeline_stages() -> list[dict[str, Any]]:
    """Return the canonical data pipeline with per-stage IO contracts."""
    return [
        {
            "stage": "configuration",
            "inputs": ["machine config:json (geometry, coils, profiles, limits)"],
            "outputs": ["TokamakConfig (validated, frozen)"],
            "processing_model": "schema validation + physical-bound checks",
        },
        {
            "stage": "equilibrium",
            "inputs": ["TokamakConfig", "diagnostics (optional, for reconstruction)"],
            "outputs": ["Psi(R,Z)", "q-profile", "flux surfaces", "magnetic axis / X-point"],
            "processing_model": "free-boundary Grad-Shafranov (Picard/Newton/multigrid) "
            "or EFIT/kinetic-EFIT/neural reconstruction; FRC rigid-rotor branch",
        },
        {
            "stage": "transport",
            "inputs": ["equilibrium", "heating/fuelling sources", "boundary conditions"],
            "outputs": ["Ti/Te/ne profiles", "confinement time", "conservation diagnostics"],
            "processing_model": "Crank-Nicolson implicit diffusion; multi-ion + "
            "neoclassical/turbulent + gyrokinetic/neural transport closures",
        },
        {
            "stage": "stability",
            "inputs": ["equilibrium", "profiles"],
            "outputs": ["MHD growth rates", "gyrokinetic spectra", "GENE/CGYRO/GS2/TGLF parity"],
            "processing_model": "ballooning/tearing/sawtooth analysis; linear+nonlinear GK",
        },
        {
            "stage": "control",
            "inputs": ["state estimate", "targets", "actuator limits"],
            "outputs": ["actuator commands", "closed-loop trajectory", "control-replay pack"],
            "processing_model": "flight-simulator MPC / H-infinity / shape+vertical+burn "
            "control (simulated, fail-closed safety interlocks)",
        },
        {
            "stage": "disruption",
            "inputs": ["plasma features", "profiles"],
            "outputs": ["disruption probability", "mitigation trigger", "calibration"],
            "processing_model": "ML disruption forecasting + SPI/pellet mitigation surrogate",
        },
        {
            "stage": "evidence",
            "inputs": ["observables", "provenance"],
            "outputs": ["claim-classified, artefact-backed evidence packs"],
            "processing_model": "bounded-claim ledger with experimental-parity boundaries",
        },
    ]


def _capabilities() -> list[dict[str, str]]:
    """Return the capability inventory with honest per-capability status."""
    return [
        {"name": "equilibrium-solver", "domain": "Equilibrium", "tier": "core", "status": "wired"},
        {
            "name": "equilibrium-reconstruction",
            "domain": "Equilibrium",
            "tier": "core",
            "status": "wired",
        },
        {"name": "integrated-transport", "domain": "Transport", "tier": "core", "status": "wired"},
        {"name": "gyrokinetic-suite", "domain": "Transport", "tier": "core", "status": "wired"},
        {"name": "mhd-stability", "domain": "Stability", "tier": "core", "status": "wired"},
        {"name": "plasma-control", "domain": "Control", "tier": "core", "status": "wired"},
        {
            "name": "disruption-prediction",
            "domain": "Disruption",
            "tier": "core",
            "status": "wired",
        },
        {
            "name": "disruption-mitigation",
            "domain": "Disruption",
            "tier": "core",
            "status": "wired",
        },
        {"name": "evidence-ledger", "domain": "Provenance", "tier": "core", "status": "wired"},
        {
            "name": "neural-surrogates",
            "domain": "Transport",
            "tier": "extended",
            "status": "wired",
        },
        {
            "name": "heating-neutronics",
            "domain": "Engineering",
            "tier": "extended",
            "status": "wired",
        },
        {
            "name": "gpu-acceleration",
            "domain": "Performance",
            "tier": "extended",
            "status": "build-available",
        },
    ]


def _backends() -> list[dict[str, Any]]:
    """Return the backend/dispatch matrix with runtime-availability status.

    The dispatch order mirrors the runtime accelerator chain declared in
    :mod:`scpn_fusion.core._multi_compat` (Rust -> Mojo -> Julia -> Go -> JAX -> NumPy),
    fastest-measured first with NumPy as the guaranteed floor.
    """
    return [
        {
            "name": "rust",
            "language": "Rust",
            "role": "hot kernels (13-crate scpn-fusion-rs workspace, PyO3)",
            "dispatch_order": 1,
            "status": "runtime-active",
        },
        {
            "name": "mojo",
            "language": "Mojo",
            "role": "vectorised kernels (probed via scpn_fusion_mojo)",
            "dispatch_order": 2,
            "status": "declared",
        },
        {
            "name": "julia",
            "language": "Julia",
            "role": "Grad-Shafranov / linear-algebra path (juliacall)",
            "dispatch_order": 3,
            "status": "build-available",
        },
        {
            "name": "go",
            "language": "Go",
            "role": "concurrent solver path (cgo)",
            "dispatch_order": 4,
            "status": "build-available",
        },
        {
            "name": "jax",
            "language": "Python",
            "role": "GPU/traceable equilibrium, gyrokinetic and transport solvers",
            "dispatch_order": 5,
            "status": "build-available",
        },
        {
            "name": "python",
            "language": "Python",
            "role": "guaranteed numerical floor (NumPy/SciPy)",
            "dispatch_order": 6,
            "status": "runtime-active",
        },
    ]


def _interfaces() -> list[dict[str, str]]:
    """Return the interface surface (CLI entry points, library, studio feed)."""
    return [
        {"kind": "cli", "entry": "scpn-fusion = scpn_fusion.cli:main"},
        {"kind": "cli", "entry": "scpn-dashboard = scpn_fusion.ui.dashboard_launcher:main"},
        {
            "kind": "cli",
            "entry": "scpn-emit-studio-manifest = scpn_fusion.studio.federation:main",
        },
        {"kind": "library", "entry": "scpn_fusion"},
        {"kind": "studio_feed", "entry": STUDIO_MANIFEST_PATH.as_posix()},
    ]


def _wire_formats() -> list[dict[str, str]]:
    """Return the named cross-boundary wire formats with schema references."""
    return [
        {
            "name": "GEQDSK",
            "schema_ref": "scpn_fusion.core.eqdsk (free-boundary equilibrium G-EQDSK exchange)",
        },
        {
            "name": "IMAS-IDS",
            "schema_ref": "scpn_fusion.io.imas_connector (ITER IMAS interface data structures)",
        },
        {
            "name": "studio-evidence",
            "schema_ref": "studio.*.v1 (9 evidence schemas from the bounded-claim ledger)",
        },
        {
            "name": "KuramotoProblem",
            "schema_ref": "scpn_fusion.phase.plasma_knm (plasma-native K/omega -> quantum bridge)",
        },
    ]


def _cross_repo() -> list[dict[str, str]]:
    """Return the cross-repository sibling adapters and their wire formats."""
    return [
        {
            "sibling": "scpn-quantum-control",
            "adapter": "phase.plasma_knm",
            "wire_format": "FRC equilibrium / plasma K_nm -> KuramotoProblem (pulsed-shot surrogate)",
        },
        {
            "sibling": "scpn-mif-core",
            "adapter": "campaigns.fusion_coupled_merge_trigger",
            "wire_format": "pulsed-compression seam -> Faraday-recovery merge trigger",
        },
        {
            "sibling": "scpn-control",
            "adapter": "control.tokamak_flight_sim",
            "wire_format": "shared controller interfaces / replay benchmarks",
        },
        {
            "sibling": "scpn-phase-orchestrator",
            "adapter": "phase.ws_phase_stream",
            "wire_format": "plasma phase stream -> UPDE orchestrator advance/hold/rollback",
        },
        {
            "sibling": "director-ai",
            "adapter": "control.director_interface",
            "wire_format": "streaming-halt disruption guard over the control loop",
        },
    ]


def _boundaries() -> dict[str, list[str]]:
    """Return the honest scope boundaries (executed / bounded / feasibility-only / closed)."""
    return {
        "executed": [
            "free-boundary Grad-Shafranov equilibrium",
            "integrated multi-ion transport",
            "linear/nonlinear gyrokinetics with external-code parity",
            "MHD stability analysis",
            "closed-loop control in the flight simulator",
            "disruption forecasting + mitigation surrogates",
            "experimental validation against ITER/SPARC/DIII-D/JET/MAST references",
            "bounded-claim evidence ledger",
        ],
        "bounded": [
            "neural transport / equilibrium surrogates (trained-regime only)",
            "reduced-order structural and heat-load screens (not finite-element analysis)",
            "quasi-3D field-line and stellarator extensions",
        ],
        "feasibility_only": [
            "GPU real-time deployment",
            "hardware-in-the-loop on physical actuators",
            "full 3D MHD",
        ],
        "closed": [
            "live tokamak / machine control (all execution is simulated)",
            "reactor certification or component qualification",
        ],
    }


def build_architecture_map_extension() -> dict[str, Any]:
    """Return the architecture-map extension block (fleet ``architecture-map.v2`` schema).

    Additive superset over schema A: the pipeline, capability inventory, backend/dispatch
    matrix, interface surface, cross-repo wire formats, and honest scope boundaries. The
    field set is peer-aligned with the fleet; the Hub ignores it for federation, the
    architecture docs consume it.

    Returns
    -------
    dict[str, Any]
        The ``architecture-map.v2`` extension block.
    """
    return {
        "version": ARCHITECTURE_MAP_VERSION,
        "pipeline_stages": _pipeline_stages(),
        "capabilities": _capabilities(),
        "backends": _backends(),
        "interfaces": _interfaces(),
        "wire_formats": _wire_formats(),
        "cross_repo": _cross_repo(),
        "boundaries": _boundaries(),
    }


def build_federation_document() -> dict[str, Any]:
    """Return the full federation document: schema_a core + architecture_map extension.

    Returns
    -------
    dict[str, Any]
        Mapping with a ``schema_a`` capability-manifest block and an
        ``architecture_map`` extension block.
    """
    return {
        "schema_a": build_manifest().to_dict(),
        "architecture_map": build_architecture_map_extension(),
    }


def _serialised_document() -> str:
    """Return the canonical serialised federation document (the on-disk form)."""
    return json.dumps(build_federation_document(), indent=2, sort_keys=True) + "\n"


def write_federation_document(repo_root: Path | None = None) -> Path:
    """Write the federation document to :data:`STUDIO_MANIFEST_PATH` and return the path.

    Parameters
    ----------
    repo_root
        Repository root; defaults to the current working directory.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    root = repo_root or Path.cwd()
    out = root / STUDIO_MANIFEST_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_serialised_document(), encoding="utf-8")
    return out


def studio_manifest_drift(repo_root: Path | None = None) -> str | None:
    """Return a drift description if the committed studio manifest is stale, else ``None``.

    The committed ``docs/_generated/studio_manifest.json`` must byte-match the generator
    output, so an edit to the verbs/evidence/architecture-map that forgets to re-emit
    cannot ship a stale federation artefact for the keeper gate to ingest.

    Parameters
    ----------
    repo_root
        Repository root; defaults to the current working directory.

    Returns
    -------
    str or None
        A human-readable drift description, or ``None`` when the artefact is current.
    """
    root = repo_root or Path.cwd()
    out = root / STUDIO_MANIFEST_PATH
    if not out.exists():
        return f"missing generated studio manifest: {STUDIO_MANIFEST_PATH.as_posix()}"
    if out.read_text(encoding="utf-8") != _serialised_document():
        return f"stale generated studio manifest: {STUDIO_MANIFEST_PATH.as_posix()}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Emit the federation document, or ``--check`` it for drift.

    Parameters
    ----------
    argv
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` when ``--check`` finds a drifted or missing artefact.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed studio manifest is current without writing it",
    )
    args = parser.parse_args(argv)
    if args.check:
        drift = studio_manifest_drift()
        if drift is not None:
            print(f"studio manifest drift: {drift}", file=sys.stderr)
            return 1
        print(f"studio manifest is current ({STUDIO_MANIFEST_PATH.as_posix()})")
        return 0
    path = write_federation_document()
    digest = build_manifest().to_dict()["content_digest"]
    print(f"Wrote {path} (schema_a content_digest={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
