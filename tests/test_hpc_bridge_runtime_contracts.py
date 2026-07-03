# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HPC Bridge Runtime Contract Tests
"""Focused runtime-contract coverage for the optional native HPC bridge."""

from __future__ import annotations

import builtins as builtins_mod
import hashlib as hashlib_mod
import json
import platform as platform_mod
import shutil as shutil_mod
import subprocess as subprocess_mod
import sys
import types as types_mod
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.hpc.hpc_bridge import HPCBridge
from scpn_fusion.hpc import hpc_bridge as hpc_mod


class _NativeSymbol:
    """Callable native symbol stand-in that accepts ctypes annotations."""

    argtypes: object
    restype: object
    calls: list[tuple[object, ...]]

    def __init__(self, return_value: object = None) -> None:
        self.return_value = return_value
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        return self.return_value


class _BaseSignatureLib:
    """Minimal native library shape with required solver symbols."""

    create_solver: _NativeSymbol
    run_step: _NativeSymbol

    def __init__(self) -> None:
        self.create_solver = _NativeSymbol(456)
        self.run_step = _NativeSymbol()


class _DeleteOnlySignatureLib(_BaseSignatureLib):
    """Native library shape exposing the legacy delete-solver alias."""

    delete_solver: _NativeSymbol

    def __init__(self) -> None:
        super().__init__()
        self.delete_solver = _NativeSymbol()


class _FullInitLib(_BaseSignatureLib):
    """Native library shape with the boundary setter used during init."""

    set_boundary_dirichlet: _NativeSymbol

    def __init__(self) -> None:
        super().__init__()
        self.set_boundary_dirichlet = _NativeSymbol()


class _RaisingDestroyLib:
    """Native library shape whose destroy path fails deterministically."""

    def destroy_solver(self, solver_ptr: int) -> None:
        raise RuntimeError(f"destroy failed for {solver_ptr}")


def _new_bridge() -> HPCBridge:
    """Return an uninitialised bridge object for branch-level tests."""
    bridge = HPCBridge.__new__(HPCBridge)
    bridge.lib = None
    bridge.loaded = False
    bridge.solver_ptr = None
    bridge.close_error = None
    bridge.close_error_count = 0
    bridge._destroy_symbol = None
    bridge._has_boundary_api = False
    bridge._has_converged_api = False
    return bridge


def _bridge_with_native_state(
    *,
    lib: object | None,
    loaded: bool = True,
    solver_ptr: int | None = 99,
) -> HPCBridge:
    """Build a bridge instance with explicit native state for tests."""
    bridge = _new_bridge()
    bridge.lib = lib
    bridge.loaded = loaded
    bridge.solver_ptr = solver_ptr
    bridge.close_error = None
    bridge.close_error_count = 0
    bridge._destroy_symbol = None
    bridge._has_boundary_api = False
    bridge._has_converged_api = False
    bridge.nr = 2
    bridge.nz = 3
    return bridge


def _digest(payload: bytes = b"trusted-native") -> str:
    """Return a stable SHA-256 digest for native-library trust tests."""
    return hashlib_mod.sha256(payload).hexdigest()


def _write_trusted_library(tmp_path: Path, payload: bytes = b"trusted-native") -> Path:
    """Write a tiny native-library placeholder payload to ``tmp_path``."""
    lib_path = tmp_path / "libscpn_solver.so"
    lib_path.write_bytes(payload)
    return lib_path


def _prepare_native_build_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a trusted source tree and compiler stand-in for build tests."""
    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module\n", encoding="utf-8")
    solver_src = tmp_path / "solver.cpp"
    solver_src.write_text("int scpn_solver_test = 0;\n", encoding="utf-8")
    solver_src.chmod(0o644)
    compiler = tmp_path / "g++"
    compiler.write_text("#!/bin/sh\n", encoding="utf-8")
    compiler.chmod(0o755)
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    monkeypatch.setattr(platform_mod, "system", lambda: "Linux")
    monkeypatch.setattr(shutil_mod, "which", lambda _name, path=None: str(compiler))
    return compiler


def _install_neural_kernel_module(
    monkeypatch: pytest.MonkeyPatch,
    kernel_type: type[object],
) -> None:
    """Install a fake neural-equilibrium module into ``sys.modules``."""
    module = types_mod.ModuleType("scpn_fusion.core.neural_equilibrium_kernel")
    module.__dict__["NeuralEquilibriumKernel"] = kernel_type
    monkeypatch.setitem(sys.modules, "scpn_fusion.core.neural_equilibrium_kernel", module)


def test_normalise_sha256_rejects_malformed_digest() -> None:
    """Reject malformed explicit trust digests before any native load."""
    with pytest.raises(ValueError, match="SHA-256"):
        hpc_mod._normalise_sha256("not-a-digest")


def test_resolve_cpp_compiler_rejects_missing_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed when no allowlisted compiler resolves on ``os.defpath``."""
    monkeypatch.setattr(shutil_mod, "which", lambda _name, path=None: None)
    assert hpc_mod._resolve_cpp_compiler() is None


def test_resolve_cpp_compiler_rejects_non_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a compiler path that resolves to a directory."""
    compiler_dir = tmp_path / "g++"
    compiler_dir.mkdir()
    monkeypatch.setattr(shutil_mod, "which", lambda _name, path=None: str(compiler_dir))
    assert hpc_mod._resolve_cpp_compiler() is None


def test_resolve_cpp_compiler_rejects_writable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject group- or world-writable compiler executables."""
    compiler = tmp_path / "g++"
    compiler.write_text("#!/bin/sh\n", encoding="utf-8")
    compiler.chmod(0o775)
    monkeypatch.setattr(shutil_mod, "which", lambda _name, path=None: str(compiler))
    assert hpc_mod._resolve_cpp_compiler() is None


def test_validate_cpp_source_rejects_external_or_missing_source(tmp_path: Path) -> None:
    """Native builds may only use the bundled regular ``solver.cpp`` file."""
    outside = tmp_path / "outside.cpp"
    outside.write_text("int untrusted = 0;\n", encoding="utf-8")
    assert not hpc_mod._validate_cpp_source(outside, tmp_path)
    assert not hpc_mod._validate_cpp_source(tmp_path / "solver.cpp", tmp_path)


def test_cpp_build_env_preserves_required_windows_process_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows builds retain only required process variables and fixed flags."""
    monkeypatch.setattr(platform_mod, "system", lambda: "Windows")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("TEMP", r"C:\Temp")
    monkeypatch.setenv("TMP", r"C:\Tmp")
    monkeypatch.setenv("CXXFLAGS", "-include attacker.hpp")

    env = hpc_mod._cpp_build_env()

    assert env["SystemRoot"] == r"C:\Windows"
    assert env["TEMP"] == r"C:\Temp"
    assert env["TMP"] == r"C:\Tmp"
    assert "CXXFLAGS" not in env


def test_sidecar_digest_reads_sha256_metadata(tmp_path: Path) -> None:
    """Read trust metadata from a package-adjacent ``.sha256`` sidecar."""
    lib_path = _write_trusted_library(tmp_path)
    expected = _digest()
    lib_path.with_suffix(lib_path.suffix + ".sha256").write_text(
        f"{expected}  {lib_path.name}\n",
        encoding="utf-8",
    )
    assert hpc_mod._sidecar_digest(lib_path) == expected


def test_manifest_digest_accepts_nested_library_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve trusted native-library digests from JSON manifest maps."""
    lib_path = _write_trusted_library(tmp_path)
    manifest = tmp_path / "trust.json"
    expected = _digest()
    manifest.write_text(
        json.dumps({"libraries": {lib_path.name: expected}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCPN_SOLVER_TRUST_MANIFEST", str(manifest))

    assert hpc_mod._manifest_digest(lib_path) == expected
    assert hpc_mod._expected_library_digest(lib_path) == expected


def test_manifest_digest_returns_none_when_manifest_has_no_path_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return ``None`` for valid manifests that do not name the library."""
    lib_path = _write_trusted_library(tmp_path)
    manifest = tmp_path / "trust.json"
    manifest.write_text(
        json.dumps({"libraries": {"other_solver.so": _digest()}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCPN_SOLVER_TRUST_MANIFEST", str(manifest))

    assert hpc_mod._manifest_digest(lib_path) is None


def test_manifest_digest_rejects_non_object_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject malformed trust manifests instead of guessing trust state."""
    lib_path = _write_trusted_library(tmp_path)
    manifest = tmp_path / "trust.json"
    manifest.write_text(json.dumps([_digest()]), encoding="utf-8")
    monkeypatch.setenv("SCPN_SOLVER_TRUST_MANIFEST", str(manifest))

    with pytest.raises(ValueError, match="JSON object"):
        hpc_mod._manifest_digest(lib_path)


def test_manifest_digest_rejects_non_string_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject non-string trust manifest digest entries."""
    lib_path = _write_trusted_library(tmp_path)
    manifest = tmp_path / "trust.json"
    manifest.write_text(
        json.dumps({"libraries": {str(lib_path): 123}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCPN_SOLVER_TRUST_MANIFEST", str(manifest))

    with pytest.raises(ValueError, match="digest must be a string"):
        hpc_mod._manifest_digest(lib_path)


def test_verify_native_library_trust_rejects_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a native library when the trusted digest mismatches bytes."""
    lib_path = _write_trusted_library(tmp_path, payload=b"actual")
    monkeypatch.setenv("SCPN_SOLVER_LIB_SHA256", _digest(b"expected"))

    with pytest.raises(ValueError, match="does not match"):
        hpc_mod._verify_native_library_trust(lib_path)


def test_require_explicit_library_path_rejects_symlink_and_directory(tmp_path: Path) -> None:
    """Reject explicit native-library paths that are not direct files."""
    lib_path = _write_trusted_library(tmp_path)
    link_path = tmp_path / "linked_solver.so"
    link_path.symlink_to(lib_path)

    with pytest.raises(ValueError, match="symlink"):
        hpc_mod._require_explicit_library_path(link_path)
    with pytest.raises(ValueError, match="regular file"):
        hpc_mod._require_explicit_library_path(tmp_path)


def test_default_library_path_returns_existing_package_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefer an existing package-local native library over the fallback path."""
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module\n", encoding="utf-8")
    lib_path = tmp_path / "libscpn_solver.so"
    lib_path.write_bytes(b"native")
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))

    assert hpc_mod._default_library_path("libscpn_solver.so") == lib_path.resolve()


def test_bridge_availability_reflects_loaded_flag() -> None:
    """Expose native accelerator availability as the loaded flag."""
    bridge = _bridge_with_native_state(lib=None, loaded=True)
    assert bridge.is_available()


def test_close_records_destroy_exception() -> None:
    """Preserve close failure telemetry while dropping the stale pointer."""
    bridge = _bridge_with_native_state(lib=_RaisingDestroyLib(), loaded=True, solver_ptr=77)
    bridge._destroy_symbol = "destroy_solver"

    bridge.close()

    assert bridge.solver_ptr is None
    assert bridge.close_error is not None
    assert "destroy failed" in bridge.close_error
    assert bridge.close_error_count == 1


def test_setup_signatures_noops_without_library() -> None:
    """Leave signature setup as a no-op when no native library is present."""
    bridge = _bridge_with_native_state(lib=None)
    bridge._setup_signatures()
    assert bridge.lib is None


def test_setup_signatures_supports_delete_alias_without_optional_apis() -> None:
    """Detect the legacy delete alias while marking optional APIs absent."""
    lib = _DeleteOnlySignatureLib()
    bridge = _bridge_with_native_state(lib=lib)

    bridge._setup_signatures()

    assert bridge._destroy_symbol == "delete_solver"
    assert not bridge._has_converged_api
    assert not bridge._has_boundary_api
    assert lib.delete_solver.argtypes is not None


def test_setup_signatures_marks_destroy_symbol_absent() -> None:
    """Represent native libraries with no destroy/delete hook explicitly."""
    bridge = _bridge_with_native_state(lib=_BaseSignatureLib())

    bridge._setup_signatures()

    assert bridge._destroy_symbol is None


def test_initialize_returns_when_unloaded_or_library_missing() -> None:
    """Initialization degrades to no-op when the native side is unavailable."""
    unloaded = _bridge_with_native_state(lib=_FullInitLib(), loaded=False, solver_ptr=None)
    unloaded.initialize(2, 3, (1.0, 2.0), (-1.0, 1.0))
    assert unloaded.solver_ptr is None

    missing_lib = _bridge_with_native_state(lib=None, loaded=True, solver_ptr=None)
    missing_lib.initialize(2, 3, (1.0, 2.0), (-1.0, 1.0))
    assert missing_lib.solver_ptr is None


def test_initialize_rejects_invalid_grid_contract() -> None:
    """Validate grid dimensions and coordinate ranges before native calls."""
    bridge = _bridge_with_native_state(lib=_FullInitLib(), loaded=True, solver_ptr=None)

    with pytest.raises(ValueError, match=">= 2"):
        bridge.initialize(1, 3, (1.0, 2.0), (-1.0, 1.0))
    with pytest.raises(ValueError, match="min < max"):
        bridge.initialize(2, 3, (2.0, 1.0), (-1.0, 1.0))


def test_initialize_creates_solver_and_sets_boundary() -> None:
    """Create the solver handle and forward boundary initialization."""
    lib = _FullInitLib()
    bridge = _bridge_with_native_state(lib=lib, loaded=True, solver_ptr=None)
    bridge._has_boundary_api = True

    bridge.initialize(2, 3, (1.0, 2.0), (-1.0, 1.0), boundary_value=2.5)

    assert bridge.nr == 2
    assert bridge.nz == 3
    assert bridge.solver_ptr == 456
    assert lib.create_solver.calls == [(2, 3, 1.0, 2.0, -1.0, 1.0)]
    assert lib.set_boundary_dirichlet.calls == [(456, 2.5)]


def test_solve_returns_none_when_solve_into_degrades() -> None:
    """Propagate ``None`` when output solving degrades after input prep."""
    bridge = _bridge_with_native_state(lib=None, loaded=True, solver_ptr=99)
    result = bridge.solve(np.zeros((3, 2), dtype=np.float64))
    assert result is None


def test_solve_into_returns_none_without_prepared_inputs() -> None:
    """Return ``None`` when inputs cannot be prepared for native solving."""
    bridge = _bridge_with_native_state(lib=_BaseSignatureLib(), loaded=True, solver_ptr=None)
    psi_out = np.zeros((3, 2), dtype=np.float64)
    assert bridge.solve_into(np.zeros((3, 2), dtype=np.float64), psi_out) is None


def test_solve_until_converged_degrades_when_native_call_degrades() -> None:
    """Propagate convergence-path degradation from the in-place API."""
    bridge = _bridge_with_native_state(lib=None, loaded=True, solver_ptr=99)
    result = bridge.solve_until_converged(np.zeros((3, 2), dtype=np.float64))
    assert result is None


def test_solve_until_converged_returns_none_without_prepared_inputs() -> None:
    """Return ``None`` from the wrapper when convergence inputs are absent."""
    bridge = _bridge_with_native_state(lib=_BaseSignatureLib(), loaded=True, solver_ptr=None)
    assert bridge.solve_until_converged(np.zeros((3, 2), dtype=np.float64)) is None


def test_solve_until_converged_into_returns_none_without_prepared_inputs() -> None:
    """Return ``None`` when convergence inputs cannot be prepared."""
    bridge = _bridge_with_native_state(lib=_BaseSignatureLib(), loaded=True, solver_ptr=None)
    psi_out = np.zeros((3, 2), dtype=np.float64)
    assert bridge.solve_until_converged_into(np.zeros((3, 2), dtype=np.float64), psi_out) is None


def test_neural_solver_degrades_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return ``None`` when the optional neural kernel cannot be imported."""
    original_import: Any = builtins_mod.__import__

    def _raising_import(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "scpn_fusion.core.neural_equilibrium_kernel":
            raise ImportError("blocked for test")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins_mod, "__import__", _raising_import)
    assert _new_bridge().solve_neural() is None


def test_neural_solver_uses_default_config_and_returns_psi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the bundled ITER config default and return a float64 Psi array."""
    seen_paths: list[Path] = []
    config_path = tmp_path / "iter_config.json"

    class _Kernel:
        """Neural-equilibrium kernel stand-in returning Psi data."""

        def __init__(self, config: str | Path) -> None:
            seen_paths.append(Path(config))

        def solve_equilibrium(self) -> dict[str, object]:
            return {"Psi": [[1.0, 2.0], [3.0, 4.0]]}

    _install_neural_kernel_module(monkeypatch, _Kernel)
    monkeypatch.setattr(hpc_mod, "default_iter_config_path", lambda: config_path)

    psi = _new_bridge().solve_neural()

    assert seen_paths == [config_path]
    assert psi is not None
    assert psi.dtype == np.float64
    assert np.allclose(psi, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))


def test_neural_solver_returns_none_when_kernel_has_no_psi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return ``None`` when neural inference omits the Psi field."""

    class _Kernel:
        """Neural-equilibrium kernel stand-in without Psi output."""

        def __init__(self, config: str | Path) -> None:
            self.config = Path(config)

        def solve_equilibrium(self) -> dict[str, object]:
            return {}

    _install_neural_kernel_module(monkeypatch, _Kernel)
    assert _new_bridge().solve_neural("config.json") is None


def test_neural_solver_returns_none_when_inference_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return ``None`` when neural surrogate inference raises."""

    class _Kernel:
        """Neural-equilibrium kernel stand-in raising during solve."""

        def __init__(self, config: str | Path) -> None:
            self.config = Path(config)

        def solve_equilibrium(self) -> dict[str, object]:
            raise RuntimeError("inference failed")

    _install_neural_kernel_module(monkeypatch, _Kernel)
    assert _new_bridge().solve_neural("config.json") is None


def test_compile_cpp_handles_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return ``None`` when the trusted native build exceeds its timeout."""
    _prepare_native_build_tree(tmp_path, monkeypatch)

    def _timeout_run(
        cmd: list[str],
        check: bool,
        timeout: float,
        env: dict[str, str],
    ) -> None:
        raise subprocess_mod.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess_mod, "run", _timeout_run)
    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_handles_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return ``None`` when the trusted compiler process fails."""
    _prepare_native_build_tree(tmp_path, monkeypatch)

    def _failed_run(
        cmd: list[str],
        check: bool,
        timeout: float,
        env: dict[str, str],
    ) -> None:
        raise subprocess_mod.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess_mod, "run", _failed_run)
    assert hpc_mod.compile_cpp() is None
