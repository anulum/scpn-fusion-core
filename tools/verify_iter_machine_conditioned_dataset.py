# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Machine-Conditioned Dataset Verifier
"""Verify a machine-conditioned synthetic equilibrium dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scpn_fusion.io.machine_conditioned_equilibrium_dataset import (
    verify_machine_conditioned_dataset,
)


def main() -> None:
    """Verify the requested directory and emit one JSON status object."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--full-field-scan", action="store_true")
    args = parser.parse_args()
    result = verify_machine_conditioned_dataset(
        args.dataset_dir,
        full_field_scan=args.full_field_scan,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
