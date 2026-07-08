#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Compatibility entrypoint for the canonical Python preflight runner."""

from __future__ import annotations

import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_python_preflight import main as _run_python_preflight_main


def main(argv: list[str] | None = None) -> int:
    """Delegate to ``tools/run_python_preflight.py`` without duplicating checks.

    Args:
        argv: Optional command arguments to pass through to the canonical runner.

    Returns:
        Process-style status code from the canonical runner.
    """
    return _run_python_preflight_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
