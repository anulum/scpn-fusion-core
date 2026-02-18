# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Streamlit Dashboard Launcher
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
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

