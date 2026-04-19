# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Streamlit Dashboard Launcher
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_path = Path(__file__).resolve().with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
