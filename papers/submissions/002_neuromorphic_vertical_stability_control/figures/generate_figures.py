#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Regenerate every figure and generated table used by submission 002."""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Run paper-local generators in a stable dependency order."""
    here = Path(__file__).resolve().parent
    scripts = (
        "fig_petri_net.py",
        "fig_compilation_pipeline.py",
        "fig_lif_neuron.py",
        "fig_vertical_stability.py",
        "generate_latency_table.py",
        "fig_latency_comparison.py",
        "fig_radiation_tolerance.py",
        "generate_evidence_manifest.py",
    )
    for script in scripts:
        subprocess.run([sys.executable, str(here / script)], check=True)


if __name__ == "__main__":
    main()
