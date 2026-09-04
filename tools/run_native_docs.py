# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Build every shipped language-native API documentation surface."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zlib
from collections.abc import Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("python", "rust", "go", "julia", "cpp", "lean")
LEAN_DOC_TARGETS = (
    "SCPNFusionSolvers:docs",
    "SafetyProof:docs",
    "PIDBoundedOutput:docs",
    "SNNReachabilityPreservation:docs",
    "PetriTokenBoundedness:docs",
    "InterlockReplayInvariance:docs",
)
EXECUTABLES = {
    "python": "sphinx-build",
    "rust": "cargo",
    "go": "go",
    "julia": "julia",
    "cpp": "doxygen",
    "lean": "lake",
}
INTERSPHINX_INVENTORIES = {
    "python": ("https://docs.python.org/3/objects.inv",),
    "numpy": ("https://numpy.org/doc/stable/objects.inv",),
    "scipy": (
        "https://scipy.github.io/devdocs/objects.inv",
        "https://docs.scipy.org/doc/scipy/objects.inv",
    ),
}
INTERSPHINX_PROJECTS = {
    "python": "Python",
    "numpy": "NumPy",
    "scipy": "SciPy",
}
INTERSPHINX_ATTEMPTS = 5
INTERSPHINX_TIMEOUT_SECONDS = 20.0
INTERSPHINX_MAX_BYTES = 16 * 1024 * 1024


def _unavailable_reason(language: str) -> str | None:
    """Return the missing generator prerequisite for a language, if any."""
    if language == "python":
        if importlib.util.find_spec("sphinx") is None:
            return "missing Python module sphinx"
        if importlib.util.find_spec("sphinx_autodoc_typehints") is None:
            return "missing Python module sphinx_autodoc_typehints"
        return None
    executable = EXECUTABLES[language]
    if shutil.which(executable) is None:
        return f"missing executable {executable}"
    if language == "cpp" and shutil.which("dot") is None:
        return "missing executable dot"
    return None


def _run(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one documentation command and preserve its return code."""
    print(f"[native-docs] {cwd}: {' '.join(command)}")
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def _validate_intersphinx_inventory(name: str, payload: bytes) -> None:
    """Reject malformed inventories and inventories for the wrong project."""
    header = payload.split(b"\n", 4)
    if len(header) != 5 or header[0] != b"# Sphinx inventory version 2":
        raise ValueError("response is not a Sphinx v2 inventory")
    project_prefix = b"# Project: "
    if not header[1].startswith(project_prefix):
        raise ValueError("inventory has no project identity")
    project = header[1][len(project_prefix) :].decode("utf-8")
    if project != INTERSPHINX_PROJECTS[name]:
        raise ValueError(
            f"inventory project is {project!r}, expected {INTERSPHINX_PROJECTS[name]!r}"
        )
    try:
        records = zlib.decompress(header[4])
    except zlib.error as error:
        raise ValueError("inventory payload is not valid zlib data") from error
    if not records.strip():
        raise ValueError("inventory contains no object records")


def _download_intersphinx_inventory(
    name: str,
    urls: Sequence[str],
    target: Path,
) -> bool:
    """Fetch one validated Sphinx inventory from bounded official sources."""
    for url in urls:
        request = Request(url, headers={"User-Agent": "SCPN-FUSION-CORE-docs/1"})
        for attempt in range(1, INTERSPHINX_ATTEMPTS + 1):
            try:
                with urlopen(  # nosec B310
                    request,
                    timeout=INTERSPHINX_TIMEOUT_SECONDS,
                ) as response:
                    payload = response.read(INTERSPHINX_MAX_BYTES + 1)
                if len(payload) > INTERSPHINX_MAX_BYTES:
                    raise ValueError("inventory exceeds the 16 MiB safety limit")
                _validate_intersphinx_inventory(name, payload)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(f"{target.suffix}.tmp")
                temporary.write_bytes(payload)
                temporary.replace(target)
                print(f"[native-docs] {name}: inventory ready from {url} at {target}")
                return True
            except (OSError, URLError, UnicodeDecodeError, ValueError) as error:
                print(
                    f"[native-docs] {name}: {url} attempt "
                    f"{attempt}/{INTERSPHINX_ATTEMPTS} failed: {error}",
                    file=sys.stderr,
                )
                if attempt < INTERSPHINX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), 8))
    return False


def _python_docs_environment() -> dict[str, str] | None:
    """Return an environment backed by locally validated inventories."""
    env = os.environ.copy()
    inventory_dir = REPO_ROOT / "docs" / "sphinx" / "_build" / "intersphinx"
    for name, urls in INTERSPHINX_INVENTORIES.items():
        target = inventory_dir / f"{name}.objects.inv"
        if not _download_intersphinx_inventory(name, urls, target):
            return None
        env[f"SCPN_INTERSPHINX_{name.upper()}_INVENTORY"] = str(target)
    return env


def _build_python_docs() -> int:
    env = _python_docs_environment()
    if env is None:
        return 1
    return _run(
        (
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "-b",
            "html",
            "docs/sphinx",
            "docs/sphinx/_build/html",
        ),
        env=env,
    ).returncode


def _build_rust_docs() -> int:
    env = os.environ.copy()
    flags = env.get("RUSTDOCFLAGS", "").strip()
    env["RUSTDOCFLAGS"] = " ".join(part for part in (flags, "-D warnings -D missing_docs") if part)
    return _run(
        (
            "cargo",
            "doc",
            "--workspace",
            "--no-deps",
            "--all-features",
            "--exclude",
            "scpn-fusion-rs",
        ),
        cwd=REPO_ROOT / "scpn-fusion-rs",
        env=env,
    ).returncode


def _build_go_docs() -> int:
    module = REPO_ROOT / "scpn-fusion-go"
    coverage = _run(("go", "run", "./cmd/doccheck", "./..."), cwd=module)
    if coverage.returncode != 0:
        return coverage.returncode
    listed = _run(("go", "list", "./..."), cwd=module, capture_output=True)
    if listed.returncode != 0:
        sys.stderr.write(listed.stderr)
        return listed.returncode
    packages = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not packages:
        print("[native-docs] go list returned no packages", file=sys.stderr)
        return 1
    for package in packages:
        rendered = _run(("go", "doc", "-all", package), cwd=module, capture_output=True)
        if rendered.returncode != 0:
            sys.stderr.write(rendered.stderr)
            return rendered.returncode
    print(f"[native-docs] Go rendered {len(packages)} package(s)")
    return 0


def _build_julia_docs() -> int:
    package = REPO_ROOT / "scpn-fusion-jl"
    return _run(
        (
            "julia",
            f"--project={package / 'docs'}",
            "-e",
            'using Pkg; Pkg.instantiate(); include("docs/make.jl")',
        ),
        cwd=package,
    ).returncode


def _build_cpp_docs() -> int:
    # Doxygen creates its final ``cpp`` directory, but not a missing parent.
    # Keep clean checkouts equivalent to developer trees that already contain
    # another documentation build output.
    (REPO_ROOT / "docs" / "_build").mkdir(parents=True, exist_ok=True)
    return _run(("doxygen", "docs/Doxyfile")).returncode


def _build_lean_docs() -> int:
    env = os.environ.copy()
    # doc-gen4 otherwise spends most of the build deriving equations for the
    # Lean standard library and may emit upstream heartbeat warnings unrelated
    # to this project's API surface.
    env["DISABLE_EQUATIONS"] = "1"
    docbuild = REPO_ROOT / "scpn-fusion-lean" / "docbuild"
    # doc-gen4 targets share back-reference JSON files and cannot safely write
    # them concurrently in one Lake invocation.
    for target in LEAN_DOC_TARGETS:
        result = _run(("lake", "build", target), cwd=docbuild, env=env)
        if result.returncode != 0:
            return result.returncode
    return 0


BUILDERS = {
    "python": _build_python_docs,
    "rust": _build_rust_docs,
    "go": _build_go_docs,
    "julia": _build_julia_docs,
    "cpp": _build_cpp_docs,
    "lean": _build_lean_docs,
}


def _requested_languages(values: Sequence[str]) -> tuple[str, ...]:
    """Expand `all` and deduplicate languages while preserving order."""
    expanded: list[str] = []
    for value in values:
        candidates = LANGUAGES if value == "all" else (value,)
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return tuple(expanded)


def main(argv: Sequence[str] | None = None) -> int:
    """Build requested native documentation and fail on the first error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        action="append",
        choices=("all", *LANGUAGES),
        default=None,
        help="Documentation surface to build; repeatable (default: all).",
    )
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="Skip a surface only when its generator executable is unavailable.",
    )
    args = parser.parse_args(argv)
    languages = _requested_languages(args.language or ("all",))

    for language in languages:
        unavailable = _unavailable_reason(language)
        if unavailable is not None:
            message = f"[native-docs] {language}: {unavailable}"
            if args.skip_unavailable:
                print(f"{message}; skipped")
                continue
            print(message, file=sys.stderr)
            return 127
        rc = BUILDERS[language]()
        if rc != 0:
            print(f"[native-docs] {language}: failed with exit code {rc}", file=sys.stderr)
            return rc
        print(f"[native-docs] {language}: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
