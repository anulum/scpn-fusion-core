# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption NPZ Fuzz Harness
"""Atheris-compatible fuzz target for disruption-shot NPZ loading."""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

from scpn_fusion.io.tokamak_disruption_archive import load_disruption_shot

_MAX_INPUT_BYTES = 10 * 1024 * 1024
_EXPECTED_REJECTIONS = (OSError, ValueError, KeyError, FileNotFoundError, zipfile.BadZipFile)


def TestOneInput(data: bytes) -> None:
    if len(data) > _MAX_INPUT_BYTES:
        data = data[:_MAX_INPUT_BYTES]
    with tempfile.TemporaryDirectory(prefix="scpn-npz-fuzz-") as tmp:
        root = Path(tmp)
        path = root / "case.npz"
        path.write_bytes(data)
        try:
            load_disruption_shot(path, disruption_dir=root)
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
