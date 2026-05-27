# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Launcher utility for running the Streamlit control-room dashboard.

The launcher writes a temporary Streamlit config to enforce safe defaults before
execing the dashboard process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

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
    """Write the temporary Streamlit security config file.

    Args:
        config_dir: Directory where `config.toml` should be written.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(_SECURITY_CONFIG, encoding="utf-8")


def _build_streamlit_command() -> list[str]:
    """Build the Streamlit CLI command for this app."""
    app_path = Path(__file__).resolve().with_name("app.py")
    return [sys.executable, "-m", "streamlit", "run", str(app_path)]


def main() -> int:
    """Execute Streamlit in a process with hardened temporary defaults.

    Returns:
        The subprocess exit code from Streamlit.
    """
    cmd = _build_streamlit_command()
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
