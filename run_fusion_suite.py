# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Legacy Launcher Wrapper
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from scpn_fusion.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
