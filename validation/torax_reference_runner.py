#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX Reference Runner
"""Run an open-source TORAX example config and export reference profiles.

Executes inside the dedicated TORAX virtual environment (``.venv-torax``),
NOT the main project venv: TORAX requires numpy>=2 and jax>=0.10, which are
incompatible with the project pins. The exported JSON carries full
provenance (TORAX version, config identity and SHA-256, git-independent
runtime metadata) so the acquired reference is auditable and the comparison
lane in the main venv stays deterministic against the committed artifact.

TORAX is Apache-2.0 (DeepMind); running it and redistributing derived
outputs with attribution is licence-compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    """Run the TORAX reference case and write the profile artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="basic_config",
        help="TORAX examples config module name (default: basic_config).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import torax  # type: ignore[import-not-found]  # only importable in .venv-torax

    config_path = Path(torax.__file__).parent / "examples" / f"{args.config_name}.py"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    torax_config = torax.build_torax_config_from_file(str(config_path))
    data_tree, state_history = torax.run_simulation(torax_config, progress_bar=False)
    sim_error = getattr(state_history, "sim_error", None)

    profiles = data_tree.profiles
    final = profiles.isel(time=-1)
    rho_norm = final["T_e"].coords["rho_norm"].values.tolist()

    payload: dict[str, Any] = {
        "schema": "scpn-fusion-core.torax-reference-profiles.v1",
        "provenance": {
            "code": "TORAX",
            "code_url": "https://github.com/google-deepmind/torax",
            "licence": "Apache-2.0",
            "torax_version": torax.__version__,
            "config_name": args.config_name,
            "config_sha256": config_sha256,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sim_error": str(sim_error),
        },
        "final_time_s": float(data_tree.time.values[-1]),
        "profiles": {
            "rho_norm": rho_norm,
            "T_e_keV": final["T_e"].values.tolist(),
            "T_i_keV": final["T_i"].values.tolist(),
            "n_e_m3": final["n_e"].values.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
