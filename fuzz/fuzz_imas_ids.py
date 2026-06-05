# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS IDS Fuzz Harness
"""Atheris-compatible fuzz target for bounded IMAS IDS JSON loading."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from scpn_fusion.io.safe_loaders import MAX_JSON_BYTES
from scpn_fusion.io.imas_connector_storage import read_ids

_MAX_INPUT_BYTES = min(MAX_JSON_BYTES, 256 * 1024)
_EXPECTED_REJECTIONS = (OSError, UnicodeDecodeError, ValueError)


def TestOneInput(data: bytes) -> None:
    """Exercise IMAS IDS JSON loading against arbitrary malformed input."""
    if len(data) > _MAX_INPUT_BYTES:
        data = data[:_MAX_INPUT_BYTES]
    with tempfile.TemporaryDirectory(prefix="scpn-imas-ids-fuzz-") as tmp:
        path = Path(tmp) / "case.json"
        path.write_bytes(data)
        try:
            read_ids(path)
        except _EXPECTED_REJECTIONS:
            return


def main() -> None:
    """Run the Atheris fuzz loop."""
    try:
        import atheris
    except ImportError as exc:
        raise SystemExit("Install atheris to run this fuzz target") from exc
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
