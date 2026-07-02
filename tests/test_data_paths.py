# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Reference-Data Path Resolution Tests
"""Tests for layout-independent reference-data root resolution."""

from __future__ import annotations

import importlib
import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

import scpn_fusion._data_paths as data_paths
from scpn_fusion._data_paths import (
    artifact_root,
    data_root,
    default_artifact_path,
    default_iter_config_path,
)


def test_data_root_contains_validation_reference_data() -> None:
    """The resolved root must contain the bundled validation reference-data tree."""
    root = data_root()
    assert (root / "validation" / "reference_data").is_dir()


def test_data_root_resolves_ipb98_coefficients() -> None:
    """The canonical IPB98 coefficient file (loaded at import) must be reachable."""
    coeff = data_root() / "validation" / "reference_data" / "itpa" / "ipb98y2_coefficients.json"
    assert coeff.is_file()


def test_default_iter_config_path_resolves_packaged_validation_config() -> None:
    """The default ITER config must be bundled below the validation package."""
    config_path = default_iter_config_path()
    assert config_path == data_root() / "validation" / "iter_config.json"
    assert config_path.is_file()


def test_default_iter_config_no_longer_depends_on_repo_root_file() -> None:
    """Runtime defaults must not depend on a repository-root config file."""
    assert not (data_root() / "iter_config.json").exists()


def test_data_root_returns_absolute_path() -> None:
    """Resolution always yields an absolute path so callers can open files directly."""
    assert data_root().is_absolute()


def test_data_root_matches_installed_validation_when_importable() -> None:
    """When the validation package is importable, the root must hold it as a child."""
    spec = importlib.util.find_spec("validation")
    if spec is None or spec.origin is None:
        pytest.skip("validation package not importable in this layout")
    validation_dir = Path(spec.origin).resolve().parent
    assert validation_dir.parent == data_root() or (data_root() / "validation").is_dir()


def test_data_root_uses_installed_validation_package_when_checkout_root_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wheel-style installs resolve through the sibling ``validation`` package."""
    fake_module = tmp_path / "wheel" / "site-packages" / "scpn_fusion" / "_data_paths.py"
    validation_init = tmp_path / "wheel" / "site-packages" / "validation" / "__init__.py"
    validation_init.parent.mkdir(parents=True)
    validation_init.write_text('"""Synthetic validation package."""\n', encoding="utf-8")
    spec = ModuleSpec("validation", loader=None, origin=str(validation_init))

    monkeypatch.setattr(data_paths, "__file__", str(fake_module))
    monkeypatch.setattr(
        "scpn_fusion._data_paths.importlib.util.find_spec",
        lambda _name: spec,
    )

    assert data_root() == validation_init.parent.parent


def test_data_root_falls_back_to_source_candidate_without_validation_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The resolver still returns a stable absolute root if package discovery fails."""
    fake_module = tmp_path / "wheel" / "site-packages" / "scpn_fusion" / "_data_paths.py"
    expected_root = fake_module.resolve().parents[2]

    monkeypatch.setattr(data_paths, "__file__", str(fake_module))
    monkeypatch.setattr(
        "scpn_fusion._data_paths.importlib.util.find_spec",
        lambda _name: None,
    )

    assert data_root() == expected_root


def test_reference_data_root_resolves_under_data_root() -> None:
    """The tokamak-archive reference root tracks the installed data root.

    The resolver must stay reachable from an installed wheel.
    """
    from scpn_fusion.io.tokamak_archive_profiles import default_reference_data_root

    assert default_reference_data_root() == data_root() / "validation" / "reference_data"


def test_diagnostics_default_config_resolves_under_data_root() -> None:
    """The diagnostics demo config must resolve via :func:`data_root` (wheel-safe)."""
    from scpn_fusion.diagnostics.run_diagnostics import DEFAULT_CONFIG_PATH

    assert data_root() / "validation" / "iter_validated_config.json" == DEFAULT_CONFIG_PATH


def test_runtime_weight_paths_resolve_under_data_root() -> None:
    """Default surrogate-weight constants track :func:`data_root`.

    Bundled QLKNN, neural-equilibrium, and pretrained heads must load from a
    wheel without writing generated files into package data.
    """
    from scpn_fusion.core.neural_equilibrium import (
        DEFAULT_WEIGHTS_PATH,
        ITER_SURROGATE_WEIGHTS_PATH,
    )
    from scpn_fusion.core.pretrained_surrogates import DEFAULT_WEIGHTS_DIR

    weights_root = data_root() / "weights"
    assert weights_root == DEFAULT_WEIGHTS_DIR
    assert weights_root / "neural_equilibrium_sparc.npz" == DEFAULT_WEIGHTS_PATH
    assert weights_root / "neural_equilibrium_iter_v1.npz" == ITER_SURROGATE_WEIGHTS_PATH
    # The shipped runtime surrogates must exist in the active layout.
    assert DEFAULT_WEIGHTS_PATH.is_file()


def test_artifact_root_uses_xdg_cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Generated artifacts default to a user-writable cache tree."""
    monkeypatch.delenv("SCPN_ARTIFACT_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    root = artifact_root()

    assert root == tmp_path / "scpn-fusion" / "artifacts"
    assert root.is_absolute()
    assert data_root() not in root.parents


def test_artifact_root_respects_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Operators can route generated artifacts to an explicit writable tree."""
    override = tmp_path / "fusion-artifacts"
    monkeypatch.setenv("SCPN_ARTIFACT_DIR", str(override))

    assert artifact_root() == override
    assert default_artifact_path("weights", "demo.npz") == override / "weights" / "demo.npz"


def test_training_output_defaults_resolve_under_artifact_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Training/demo defaults must not write into bundled wheel data."""
    monkeypatch.setenv("SCPN_ARTIFACT_DIR", str(tmp_path))

    import scpn_fusion.control.disruption_checkpoint_policy as checkpoint_policy
    import scpn_fusion.core.fno_training as fno_training
    import scpn_fusion.core.gs_transport_surrogate_training as gs_training
    import scpn_fusion.core.pretrained_surrogates as pretrained_surrogates

    checkpoint_policy = importlib.reload(checkpoint_policy)
    gs_training = importlib.reload(gs_training)
    fno_training = importlib.reload(fno_training)
    pretrained_surrogates = importlib.reload(pretrained_surrogates)

    weights_root = artifact_root() / "weights"
    assert weights_root / "fno_turbulence.npz" == fno_training.DEFAULT_WEIGHTS_PATH
    assert weights_root / "fno_turbulence_sparc.npz" == fno_training.DEFAULT_SPARC_WEIGHTS_PATH
    assert (
        weights_root / "gs_transport_surrogate.npz"
        == fno_training.DEFAULT_GS_TRANSPORT_WEIGHTS_PATH
    )
    assert weights_root / "gs_transport_surrogate.npz" == (
        gs_training.DEFAULT_GS_TRANSPORT_WEIGHTS_PATH
    )
    assert weights_root == pretrained_surrogates.DEFAULT_BUNDLE_WEIGHTS_DIR
    assert artifact_root() / "disruptor.pth" == checkpoint_policy.default_model_path(
        "disruptor.pth"
    )


def test_reference_defaults_resolve_under_bundled_data_root() -> None:
    """Reference readers must keep using bundled validation data in a wheel."""
    from scpn_fusion.core._tglf_interface_types import TGLF_REF_DIR
    from scpn_fusion.core.geometry_3d import _default_config_path
    from scpn_fusion.core.public_frc_reference import REFERENCE_ROOT

    assert data_root() / "validation" / "tglf_reference" == TGLF_REF_DIR
    assert data_root() / "validation" / "reference_data" == REFERENCE_ROOT
    assert data_root() / "validation" / "iter_validated_config.json" == _default_config_path()
