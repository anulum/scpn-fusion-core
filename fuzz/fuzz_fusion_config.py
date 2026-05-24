# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fusion Config Fuzz Harness
"""Atheris-compatible fuzz target for FusionKernel JSON configuration loading."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from scpn_fusion.core.fusion_kernel import MAX_CONFIG_BYTES, FusionKernel

_MAX_INPUT_BYTES = min(MAX_CONFIG_BYTES, 256 * 1024)
_EXPECTED_REJECTIONS = (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError, ValidationError)


def TestOneInput(data: bytes) -> None:
    if len(data) > _MAX_INPUT_BYTES:
        data = data[:_MAX_INPUT_BYTES]
    with tempfile.TemporaryDirectory(prefix="scpn-config-fuzz-") as tmp:
        path = Path(tmp) / "config.json"
        path.write_bytes(data)
        kernel = FusionKernel.__new__(FusionKernel)
        try:
            kernel.load_config(path)
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
