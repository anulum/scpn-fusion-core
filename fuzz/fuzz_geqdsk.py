# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GEQDSK Fuzz Harness
"""Atheris-compatible fuzz target for the GEQDSK parser."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from scpn_fusion.core.eqdsk import MAX_GEQDSK_BYTES, read_geqdsk

_MAX_INPUT_BYTES = min(MAX_GEQDSK_BYTES, 256 * 1024)
_EXPECTED_REJECTIONS = (OSError, UnicodeDecodeError, ValueError)


def TestOneInput(data: bytes) -> None:
    if len(data) > _MAX_INPUT_BYTES:
        data = data[:_MAX_INPUT_BYTES]
    with tempfile.TemporaryDirectory(prefix="scpn-geqdsk-fuzz-") as tmp:
        path = Path(tmp) / "case.geqdsk"
        path.write_bytes(data)
        try:
            read_geqdsk(path)
        except _EXPECTED_REJECTIONS:
            return


def main() -> None:
    try:
        import atheris
    except ImportError as exc:
        raise SystemExit("Install atheris to run this fuzz target") from exc
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
