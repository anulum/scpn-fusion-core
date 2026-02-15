# -----------------------------------------------------------------------
# SCPN Fusion Core -- Sphinx Configuration
# Copyright 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU Affero General Public License v3.0
#          Commercial licensing available upon request.
# -----------------------------------------------------------------------
"""Sphinx configuration for SCPN-Fusion-Core documentation."""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- ensure ``src/`` is importable so autodoc can resolve modules
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "SCPN-Fusion-Core"
copyright = "1998-2026 Miroslav Sotek (ANULUM CH & LI). Licensed under GNU AGPL v3"
author = "Miroslav Sotek"

# Extract version from the package __init__.py without importing it (avoids
# pulling in heavy dependencies like NumPy at doc-build time when they may
# not be present in the docs environment).
try:
    from scpn_fusion import __version__ as _pkg_version
except Exception:
    _init_path = _project_root / "src" / "scpn_fusion" / "__init__.py"
    _text = _init_path.read_text(encoding="utf-8")
    _match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', _text)
    _pkg_version = _match.group(1) if _match else "0.0.0"

version = _pkg_version          # short X.Y
release = _pkg_version          # full X.Y.Z

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings from source
    "sphinx.ext.napoleon",      # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",      # [source] links on API pages
    "sphinx.ext.intersphinx",   # cross-reference NumPy/SciPy/Python docs
    "sphinx.ext.mathjax",       # render LaTeX math in HTML
    "sphinx.ext.todo",          # .. todo:: directive support
    "sphinx.ext.ifconfig",      # conditional content
    "sphinx.ext.githubpages",   # .nojekyll for GitHub Pages deployment
]

# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
pygments_style = "friendly"

# ---------------------------------------------------------------------------
# HTML output -- Furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1d4ed8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/anulum/scpn-fusion-core",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
                "2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49"
                "-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15"
                "-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
                "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 "
                '2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 '
                "2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 "
                "2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 "
                '1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 '
                '8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"SCPN-Fusion-Core v{release}"
html_short_title = "SCPN-Fusion-Core"
html_favicon = None  # add favicon.ico to _static/ when available

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    # Heavy or optional dependencies that may not be installed in the docs
    # build environment.  Mocking them lets autodoc parse type annotations
    # without actually importing the C-extension modules.
    "sc_neurocore",
    "scpn_fusion_rs",
    "torch",
    "torchvision",
    "jax",
    "jaxlib",
    "cupy",
    "brainflow",
    "streamlit",
    "plotly",
    "maturin",
    "pyo3",
]

# ---------------------------------------------------------------------------
# Napoleon -- Google & NumPy docstring support
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_param = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------------
# Intersphinx -- cross-link to upstream project docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# ---------------------------------------------------------------------------
# MathJax -- enable equation numbering
# ---------------------------------------------------------------------------
mathjax3_config = {
    "tex": {
        "tags": "ams",
    },
}

# ---------------------------------------------------------------------------
# Todo extension
# ---------------------------------------------------------------------------
todo_include_todos = True
