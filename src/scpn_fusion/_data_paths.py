# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Reference-Data Path Resolution
"""Layout-independent resolution of the bundled reference-data root.

Modules historically located the ``validation/`` and ``weights/`` data trees with
``Path(__file__).resolve().parents[3]``, which is correct only inside the source
checkout. In an installed wheel ``scpn_fusion`` lives in ``site-packages`` while the
``validation`` package is a sibling there, so the source-tree offset points at the
wrong directory. :func:`data_root` resolves the directory that contains
``validation/`` (and, in a checkout, ``weights/``) for the source tree, an editable
install, and a built wheel alike.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def data_root() -> Path:
    """Return the directory that contains the bundled ``validation`` data tree.

    Resolution order:

    1. The source-checkout root two levels above this package (``src/scpn_fusion`` →
       repository root), used when running from a checkout or ``PYTHONPATH=src``.
    2. The location of the installed ``validation`` package, used when ``scpn_fusion``
       is installed from a wheel and ``validation`` is a sibling top-level package.

    Returns
    -------
    Path
        Directory ``D`` such that ``D / "validation" / "reference_data"`` exists for
        the active install layout. Falls back to the source-tree candidate when the
        ``validation`` package cannot be located.
    """
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "validation").is_dir():
        return source_candidate

    spec = importlib.util.find_spec("validation")
    if spec is not None and spec.origin is not None:
        # validation/__init__.py → validation/ → the directory holding validation/.
        return Path(spec.origin).resolve().parent.parent

    return source_candidate


def artifact_root() -> Path:
    """Return the writable default root for generated SCPN Fusion artifacts.

    Resolution order:

    1. ``SCPN_ARTIFACT_DIR`` when set, for explicit operator control in CI,
       notebooks, and production deployments.
    2. ``XDG_CACHE_HOME/scpn-fusion/artifacts`` on Linux-style systems.
    3. ``~/.cache/scpn-fusion/artifacts`` when ``XDG_CACHE_HOME`` is unset.

    Returns
    -------
    Path
        Absolute directory path intended for generated checkpoints, reports,
        plots, and training outputs. The directory is not created until a caller
        writes a concrete artifact below it.
    """
    override = os.environ.get("SCPN_ARTIFACT_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return (cache_root / "scpn-fusion" / "artifacts").resolve()


def default_artifact_path(*parts: str) -> Path:
    """Return a generated-artifact path below :func:`artifact_root`.

    Parameters
    ----------
    *parts
        Relative path components below the artifact root. Empty components are
        ignored so callers can compose optional subdirectories without producing
        malformed paths.

    Returns
    -------
    Path
        Absolute writable default path for a generated artifact.
    """
    path = artifact_root()
    for part in parts:
        if part:
            path /= part
    return path


def default_iter_config_path() -> Path:
    """Return the bundled ITER reference configuration path.

    Returns
    -------
    Path
        Absolute path to the default ITER-format JSON configuration bundled
        under the ``validation`` package data tree.
    """
    return data_root() / "validation" / "iter_config.json"
