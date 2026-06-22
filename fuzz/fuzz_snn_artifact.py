# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — SNN Controller Artifact Fuzz Harness
"""Atheris-compatible fuzz target for compiled SNN controller artifact loading.

Exercises ``load_artifact``, the deserialisation boundary for a ``.scpnctl.json``
spiking-controller artifact: UTF-8 + JSON parsing, raw-schema validation, weight
parsing, and the zlib/base64 compact packed-weight payload decode (a compression
boundary already guarded by explicit size limits). Every malformed input must be
rejected with a ``ValueError`` (which covers ``json.JSONDecodeError``,
``UnicodeDecodeError``, and ``ArtifactValidationError``) or an ``OSError``; any
other escaping exception is a validation gap for the fuzzer to surface.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from scpn_fusion.scpn.artifact import load_artifact

_MAX_INPUT_BYTES = 4 * 1024 * 1024
_EXPECTED_REJECTIONS = (OSError, ValueError)


def TestOneInput(data: bytes) -> None:
    if len(data) > _MAX_INPUT_BYTES:
        data = data[:_MAX_INPUT_BYTES]
    with tempfile.TemporaryDirectory(prefix="scpn-snn-artifact-fuzz-") as tmp:
        path = Path(tmp) / "case.scpnctl.json"
        path.write_bytes(data)
        try:
            load_artifact(path)
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
