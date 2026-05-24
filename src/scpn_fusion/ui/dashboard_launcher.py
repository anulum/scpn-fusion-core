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
import tempfile
from pathlib import Path
import os


_SECURITY_CONFIG = """\
[browser]
gatherUsageStats = false

[global]
developmentMode = false

[server]
headless = true
enableCORS = true
enableXsrfProtection = true
"""


def _write_security_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(_SECURITY_CONFIG, encoding="utf-8")


def main() -> int:
    app_path = Path(__file__).resolve().with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    with tempfile.TemporaryDirectory(prefix="scpn-streamlit-") as tmp:
        config_dir = Path(tmp)
        _write_security_config(config_dir)
        env = os.environ.copy()
        env["STREAMLIT_CONFIG_DIR"] = str(config_dir)
        env["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        return int(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    raise SystemExit(main())
