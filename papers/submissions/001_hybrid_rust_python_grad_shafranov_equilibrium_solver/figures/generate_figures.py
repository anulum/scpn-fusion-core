#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Paper 001 deterministic artifact runner.
"""Regenerate every figure and evidence manifest used by Paper 001."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Run paper-local generators in a stable dependency order."""
    here = Path(__file__).resolve().parent
    scripts = (
        "fig_inverse_reconstruction.py",
        "fig_sparc_equilibrium.py",
        "generate_evidence_manifest.py",
    )
    for script in scripts:
        subprocess.run([sys.executable, str(here / script)], check=True)


if __name__ == "__main__":
    main()
