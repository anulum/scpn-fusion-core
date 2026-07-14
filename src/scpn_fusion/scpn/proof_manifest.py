# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Lean Proof-Contract Manifest
"""Deterministic checksum of the Lean Petri→SNN compiler-contract proofs.

The `scpn-fusion-lean` package machine-checks that the Petri→SNN compiler
preserves the safety-relevant semantics: reachability (static graph contract) and
interlock enforcement + replay invariance (dynamic marking contract). This module
computes a stable, source-based SHA-256 over exactly the proof files that back the
contract, so a controller artifact (``.scpnctl.json``) can carry the proof hash of
the theorems that certify its compilation. The checksum is over the *sources* (not
the compiled ``.olean``) so it is reproducible across machines and Lean toolchains;
the Lean CI lane is what proves the sources actually type-check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

#: Proof system identifier recorded in the artifact.
PROOF_SYSTEM = "lean4"

#: Lean source files that constitute the Petri→SNN compiler+runtime contract,
#: relative to the Lean package directory. Order is irrelevant (hash is sorted).
CONTRACT_FILES: tuple[str, ...] = (
    "SCPNFusionSolvers.lean",
    "SNNReachabilityPreservation.lean",
    "InterlockReplayInvariance.lean",
)

#: The contract theorems certified by the proof bundle (for self-description).
CONTRACT_THEOREMS: tuple[str, ...] = (
    # Static graph contract (SNNReachabilityPreservation).
    "compile_preserves_direct_edge",
    "compile_reflects_direct_edge",
    "compile_reachability_equivalent",
    "compile_preserves_well_formed_edges",
    "compile_has_no_spurious_reachable_path",
    # Dynamic interlock + replay contract (InterlockReplayInvariance).
    "guard_marked_disables",
    "interlock_raised_noop",
    "compile_preserves_enabled",
    "compile_step_commutes",
    "compile_preserves_interlock_block",
    "replay_append",
    "compile_replay_commutes",
    "replay_keeps_guard_clear",
)


def default_lean_dir() -> Path:
    """Return the repository's ``scpn-fusion-lean`` package directory."""
    # src/scpn_fusion/scpn/proof_manifest.py -> repo root is three parents up from src.
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "scpn-fusion-lean"


def _normalise_source(text: str) -> str:
    """Normalise a Lean source for hashing (line endings only; content is exact)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def compute_proof_checksum(lean_dir: str | Path | None = None) -> str:
    """Return the SHA-256 hex digest of the Lean contract proof sources.

    The digest is computed over the contract files in a fixed sorted order, each
    contribution being ``"<name>\\n<normalised-source>\\n"``, so it is independent
    of filesystem ordering and reproducible across machines.

    Parameters
    ----------
    lean_dir : str | Path | None
        The Lean package directory. Defaults to the repository's
        ``scpn-fusion-lean`` directory.

    Raises
    ------
    FileNotFoundError
        If any contract file is missing (a broken proof bundle must not silently
        produce a checksum).
    """
    base = Path(lean_dir) if lean_dir is not None else default_lean_dir()
    digest = hashlib.sha256()
    for name in sorted(CONTRACT_FILES):
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(f"Lean contract proof file missing: {path}")
        source = _normalise_source(path.read_text(encoding="utf-8"))
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
        digest.update(source.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def proof_contract_manifest(lean_dir: str | Path | None = None) -> dict[str, Any]:
    """Return the self-describing proof-contract manifest.

    Includes the proof system, the contract files, the certified theorem names,
    and the deterministic checksum.
    """
    return {
        "proof_system": PROOF_SYSTEM,
        "files": list(CONTRACT_FILES),
        "theorems": list(CONTRACT_THEOREMS),
        "checksum": compute_proof_checksum(lean_dir),
    }
