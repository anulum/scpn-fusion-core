# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Runtime CLI
"""Execute one versioned TORAX request and atomically publish its outcome."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .contracts import ToraxRunRequest
from .serialization import load_json_object, write_json_atomic
from .worker import execute_request, invalid_request_outcome


def main(argv: Sequence[str] | None = None) -> int:
    """Run one real TORAX request through the process-isolated public surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        raw = load_json_object(arguments.request)
    except Exception as error:
        raw = {}
        outcome = invalid_request_outcome(
            raw, f"request loading failed: {type(error).__name__}: {error}"
        )
        write_json_atomic(arguments.result, outcome.to_dict())
        return 2
    try:
        request = ToraxRunRequest.from_dict(raw)
    except Exception as error:
        outcome = invalid_request_outcome(
            raw, f"request validation failed: {type(error).__name__}: {error}"
        )
        write_json_atomic(arguments.result, outcome.to_dict())
        return 2
    manifest_path = arguments.output_sidecar.with_suffix(
        arguments.output_sidecar.suffix + ".manifest.json"
    )
    outcome = execute_request(
        request,
        sidecar_path=arguments.output_sidecar,
        manifest_path=manifest_path,
    )
    write_json_atomic(arguments.result, outcome.to_dict())
    return 0 if outcome.success else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
