#!/usr/bin/env python
"""Check PyPI version parity against local project metadata."""

from __future__ import annotations

import argparse
import json
import re
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"


def read_local_version(pyproject_path: Path) -> str:
    """Read project.version from pyproject.toml."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"Unable to determine project.version from {pyproject_path.as_posix()}"
        )
    return version.strip()


def fetch_pypi_version(package: str, timeout: float) -> str:
    """Fetch the latest published version for a package from PyPI."""
    url = f"https://pypi.org/pypi/{package}/json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
    info = payload.get("info", {})
    version = info.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"PyPI payload missing info.version for package {package!r}")
    return version.strip()


def normalize_version(version: str, *, strip_v_prefix: bool) -> str:
    out = version.strip()
    if strip_v_prefix and out.lower().startswith("v"):
        return out[1:]
    return out


def _coerce_numeric_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in version.split("."):
        match = re.match(r"^(\d+)", raw_part)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def compare_versions(local: str, remote: str, *, mode: str) -> tuple[bool, str]:
    """Compare local and remote versions under the chosen mode."""
    if mode not in {"equal", "not-behind"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if local == remote:
        return True, f"Local version matches PyPI ({local})."

    local_tuple = _coerce_numeric_tuple(local)
    remote_tuple = _coerce_numeric_tuple(remote)
    if local_tuple and remote_tuple:
        width = max(len(local_tuple), len(remote_tuple))
        local_cmp = local_tuple + (0,) * (width - len(local_tuple))
        remote_cmp = remote_tuple + (0,) * (width - len(remote_tuple))
        ordering = (local_cmp > remote_cmp) - (local_cmp < remote_cmp)
    else:
        ordering = (local > remote) - (local < remote)

    if mode == "equal":
        return False, (
            f"Version mismatch: local={local}, pypi={remote}. "
            "Use this mode after publish verification."
        )

    if ordering < 0:
        return False, (
            f"Local version is behind PyPI: local={local}, pypi={remote}. "
            "Bump local metadata before release."
        )

    return True, (
        f"Local version is ahead of PyPI (local={local}, pypi={remote}); "
        "this is expected before publish."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="scpn-fusion")
    parser.add_argument(
        "--pyproject",
        default=str(DEFAULT_PYPROJECT),
        help="Path to pyproject.toml (used when --local-version is omitted).",
    )
    parser.add_argument(
        "--local-version",
        default=None,
        help="Override local version (e.g., v3.9.3 for tag checks).",
    )
    parser.add_argument(
        "--mode",
        choices=["equal", "not-behind"],
        default="not-behind",
        help="Comparison mode: equal (strict parity) or not-behind (release-prep).",
    )
    parser.add_argument(
        "--strip-v-prefix",
        action="store_true",
        help="Strip leading 'v' from local/remote versions before comparison.",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument(
        "--allow-network-failure",
        action="store_true",
        help="Return success on network failures (warning-only mode).",
    )
    args = parser.parse_args(argv)

    pyproject_path = Path(args.pyproject)
    if not pyproject_path.is_absolute():
        pyproject_path = REPO_ROOT / pyproject_path

    local_raw = args.local_version or read_local_version(pyproject_path)
    local = normalize_version(local_raw, strip_v_prefix=args.strip_v_prefix)

    attempts = max(1, int(args.retries))
    remote_raw: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            remote_raw = fetch_pypi_version(args.package, timeout=float(args.timeout))
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 - preserve user-facing diagnostics
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, float(args.retry_delay)))

    if remote_raw is None:
        message = (
            f"Unable to fetch PyPI version for {args.package!r}: {last_error!r}"
            if last_error
            else f"Unable to fetch PyPI version for {args.package!r}."
        )
        if args.allow_network_failure:
            print(f"WARNING: {message}")
            return 0
        print(f"ERROR: {message}")
        return 2

    remote = normalize_version(remote_raw, strip_v_prefix=args.strip_v_prefix)
    ok, detail = compare_versions(local, remote, mode=args.mode)
    if ok:
        print(f"OK: {detail}")
        return 0
    print(f"ERROR: {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
