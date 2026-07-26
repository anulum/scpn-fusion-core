# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — atomic PyPI download-series snapshot
"""Record SCPN Fusion Core's daily PyPI download series in a bounded CSV."""

from __future__ import annotations

import argparse
import csv
import http.client
import importlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import quote


class _TomlLoader(Protocol):
    """Describe the TOML loader shared by supported Python versions."""

    def loads(self, data: str, /) -> dict[str, Any]:
        """Parse one TOML document."""


tomllib = cast(
    _TomlLoader,
    importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli"),
)

PYPISTATS_HOST = "pypistats.org"
PYPISTATS_PATH = "/api/packages/{package}/overall"
PYPISTATS_TYPE = "overall_downloads"
CATEGORIES = ("without_mirrors", "with_mirrors")
CSV_HEADER = ("date", *CATEGORIES)
PAYLOAD_KEYS = frozenset(("data", "package", "type"))
ROW_KEYS = frozenset(("category", "date", "downloads"))
PROJECT_PACKAGE = "scpn-fusion"
PROJECT_CSV = Path("downloads/scpn-fusion.csv")
REQUEST_TIMEOUT_SECONDS = 30
MAX_RESPONSE_BYTES = 5_000_000
RETRYABLE_STATUSES = (429, 502, 503, 504)
RETRY_DELAYS = (15.0, 30.0, 60.0)
MAX_TRANSIENT_WINDOW_SECONDS = REQUEST_TIMEOUT_SECONDS * (len(RETRY_DELAYS) + 1) + int(
    sum(RETRY_DELAYS)
)

Fetch = Callable[[str], bytes]
Sleep = Callable[[float], None]
DownloadRows = dict[str, dict[str, int]]


class DownloadSnapshotError(RuntimeError):
    """Report a fail-closed remote or local snapshot failure."""


class RetryableSnapshotError(DownloadSnapshotError):
    """Report a transient remote failure eligible for bounded retry."""


class DuplicateJSONObjectKeyError(ValueError):
    """Report an ambiguous JSON object containing a repeated member name."""


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate names at every depth."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONObjectKeyError(f"duplicate JSON object name {key!r}")
        result[key] = value
    return result


def detect_package(pyproject_path: Path) -> str:
    """Return the distribution name from a PEP 621 project table."""
    document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"no [project] table in {pyproject_path}")
    raw_name = project.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"no [project] name in {pyproject_path}")
    return raw_name.strip()


def require_project_csv_target(package: str, csv_path: Path) -> None:
    """Accept only SCPN Fusion Core's exact package and relative CSV path."""
    if package != PROJECT_PACKAGE:
        raise ValueError(f"project metrics package must be {PROJECT_PACKAGE!r}")
    if csv_path.is_absolute() or csv_path.as_posix() != PROJECT_CSV.as_posix():
        raise ValueError(f"project metrics CSV must be {PROJECT_CSV.as_posix()!r}")


def package_endpoint_path(package: str) -> str:
    """Return the fixed-host pypistats path for one package."""
    if not package.strip():
        raise ValueError("package name must not be empty")
    return PYPISTATS_PATH.format(package=quote(package.strip(), safe=""))


def _http_get(package: str) -> bytes:
    """Fetch one size-bounded pypistats payload from the fixed HTTPS host."""
    connection = http.client.HTTPSConnection(PYPISTATS_HOST, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        connection.request(
            "GET",
            package_endpoint_path(package),
            headers={
                "Accept": "application/json",
                "User-Agent": "scpn-fusion-metrics/1",
            },
        )
        response = connection.getresponse()
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if response.status in RETRYABLE_STATUSES:
            raise RetryableSnapshotError(
                f"pypistats returned transient HTTP {response.status} {response.reason}"
            )
        if response.status != 200:
            raise DownloadSnapshotError(
                f"pypistats returned HTTP {response.status} {response.reason}"
            )
        content_type = response.getheader("Content-Type", "").lower()
        if not content_type.startswith("application/json"):
            raise DownloadSnapshotError(
                f"pypistats returned unexpected Content-Type {content_type!r}"
            )
        if len(body) > MAX_RESPONSE_BYTES:
            raise DownloadSnapshotError("pypistats response exceeded the size limit")
        return body
    except (OSError, http.client.HTTPException) as exc:
        raise RetryableSnapshotError(f"pypistats request failed: {exc}") from exc
    finally:
        connection.close()


def _valid_date(raw_date: object) -> str | None:
    """Return one canonical ISO date or ``None`` for malformed input."""
    if not isinstance(raw_date, str) or raw_date != raw_date.strip():
        return None
    try:
        parsed = date.fromisoformat(raw_date)
    except ValueError:
        return None
    return raw_date if parsed.isoformat() == raw_date else None


def _valid_count(raw_count: object) -> int | None:
    """Return one non-negative integer download count or ``None``."""
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        return None
    return raw_count if raw_count >= 0 else None


def validate_overall(payload: object, package: str) -> DownloadRows:
    """Validate exact identity, every row and sparse-series invariants."""
    if not isinstance(payload, dict):
        raise DownloadSnapshotError("pypistats response must be a JSON object")
    if set(payload) != PAYLOAD_KEYS:
        raise DownloadSnapshotError("pypistats response has unexpected top-level keys")
    if payload.get("package") != package:
        raise DownloadSnapshotError("pypistats response package identity mismatch")
    if payload.get("type") != PYPISTATS_TYPE:
        raise DownloadSnapshotError("pypistats response type mismatch")
    raw_rows = payload.get("data")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise DownloadSnapshotError("pypistats response data must be a non-empty list")

    rows: DownloadRows = {}
    seen: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict) or set(raw_row) != ROW_KEYS:
            raise DownloadSnapshotError(f"pypistats row {index} has an invalid object schema")
        category = raw_row.get("category")
        row_date = _valid_date(raw_row.get("date"))
        downloads = _valid_count(raw_row.get("downloads"))
        if category not in CATEGORIES:
            raise DownloadSnapshotError(f"pypistats row {index} has invalid category")
        if row_date is None:
            raise DownloadSnapshotError(f"pypistats row {index} has invalid date")
        if downloads is None:
            raise DownloadSnapshotError(f"pypistats row {index} has invalid downloads")
        key = (row_date, cast(str, category))
        if key in seen:
            raise DownloadSnapshotError(f"pypistats response duplicates {row_date}/{category}")
        seen.add(key)
        rows.setdefault(row_date, {})[cast(str, category)] = downloads

    for row_date, values in rows.items():
        if "with_mirrors" not in values:
            raise DownloadSnapshotError(
                f"pypistats response is missing with_mirrors for {row_date}"
            )
        without_mirrors = values.get("without_mirrors")
        if without_mirrors is not None and without_mirrors > values["with_mirrors"]:
            raise DownloadSnapshotError(
                f"pypistats response violates mirror-count ordering for {row_date}"
            )
    return rows


def fetch_overall(package: str, fetch: Fetch = _http_get) -> DownloadRows:
    """Fetch, decode and strictly validate one overall-series payload."""
    try:
        decoded: object = json.loads(
            fetch(package), object_pairs_hook=_reject_duplicate_object_keys
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateJSONObjectKeyError,
    ) as exc:
        raise DownloadSnapshotError(f"pypistats returned invalid JSON: {exc}") from exc
    return validate_overall(decoded, package)


def fetch_overall_with_retry(
    package: str,
    fetch: Fetch = _http_get,
    sleep: Sleep = time.sleep,
) -> DownloadRows | None:
    """Retry transient failures within a 225-second worst-case window."""
    for delay in (*RETRY_DELAYS, None):
        try:
            return fetch_overall(package, fetch)
        except RetryableSnapshotError:
            if delay is not None:
                sleep(delay)
    return None


def _validate_rows(rows: Mapping[str, Mapping[str, int]], label: str) -> None:
    """Validate an exact in-memory CSV row mapping before persistence."""
    for row_date, values in rows.items():
        if _valid_date(row_date) is None:
            raise ValueError(f"invalid date in {label}: {row_date!r}")
        if not values or not set(values).issubset(CATEGORIES):
            raise ValueError(f"invalid categories in {label}: {row_date}")
        if "with_mirrors" not in values:
            raise ValueError(f"missing with_mirrors in {label}: {row_date}")
        for category in values:
            if _valid_count(values.get(category)) is None:
                raise ValueError(f"invalid {category} count in {label}: {row_date}")
        without_mirrors = values.get("without_mirrors")
        if without_mirrors is not None and without_mirrors > values["with_mirrors"]:
            raise ValueError(f"invalid mirror-count ordering in {label}: {row_date}")


def read_csv(path: Path) -> DownloadRows:
    """Read and strictly validate an existing download-series CSV."""
    if not path.exists():
        return {}
    rows: DownloadRows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_HEADER:
            raise ValueError(f"unexpected CSV header in {path}")
        for line_number, record in enumerate(reader, start=2):
            if set(record) != set(CSV_HEADER):
                raise ValueError(f"unexpected CSV fields at {path}:{line_number}")
            row_date = _valid_date(record.get("date"))
            if row_date is None:
                raise ValueError(f"invalid date at {path}:{line_number}")
            if row_date in rows:
                raise ValueError(f"duplicate date {row_date} at {path}:{line_number}")
            values: dict[str, int] = {}
            for category in CATEGORIES:
                raw_count = record.get(category)
                if category == "without_mirrors" and raw_count == "":
                    continue
                try:
                    parsed_count: object = int(raw_count) if raw_count is not None else None
                except ValueError:
                    parsed_count = None
                count = _valid_count(parsed_count)
                if count is None or str(count) != raw_count:
                    raise ValueError(f"invalid {category} count at {path}:{line_number}")
                values[category] = count
            rows[row_date] = values
    _validate_rows(rows, str(path))
    return rows


def merge_rows(existing: DownloadRows, fresh: DownloadRows) -> DownloadRows:
    """Upsert fresh counts without mutating either validated input mapping."""
    _validate_rows(existing, "existing rows")
    _validate_rows(fresh, "fresh rows")
    merged = {row_date: dict(values) for row_date, values in existing.items()}
    for row_date, values in fresh.items():
        merged[row_date] = dict(values)
    return merged


def write_csv(path: Path, rows: DownloadRows) -> None:
    """Atomically write one date-sorted, schema-fixed download series."""
    _validate_rows(rows, "output rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            for row_date in sorted(rows):
                writer.writerow(
                    [
                        row_date,
                        *(rows[row_date].get(category, "") for category in CATEGORIES),
                    ]
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def summary(package: str, rows: DownloadRows) -> str:
    """Return a deterministic one-line latest-day summary."""
    if not rows:
        return f"{package}: no download data available yet"
    latest = max(rows)
    without_mirrors = rows[latest].get("without_mirrors")
    without_summary = "n/a" if without_mirrors is None else str(without_mirrors)
    return (
        f"{package}: {len(rows)} days recorded; latest {latest} "
        f"without_mirrors={without_summary} "
        f"with_mirrors={rows[latest]['with_mirrors']}"
    )


def main(
    argv: list[str] | None = None,
    fetch: Fetch = _http_get,
    sleep: Sleep = time.sleep,
) -> int:
    """Resolve, fetch, validate, merge and persist one bounded snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--package")
    parser.add_argument("--csv")
    parser.add_argument("--project-csv-only", action="store_true")
    parser.add_argument("--print-package", action="store_true")
    arguments = parser.parse_args(argv)

    try:
        package = arguments.package or detect_package(Path(arguments.pyproject))
        if arguments.print_package:
            print(package)
            return 0
        if not arguments.csv:
            parser.error("--csv is required unless --print-package is used")
        csv_path = Path(arguments.csv)
        if arguments.project_csv_only:
            require_project_csv_target(package, csv_path)
        fresh = fetch_overall_with_retry(package, fetch, sleep)
        if fresh is None:
            print(
                "snapshot skipped: pypistats remained unavailable after bounded retries; "
                "the next successful rolling-window fetch will backfill it",
                file=sys.stderr,
            )
            return 0
        rows = merge_rows(read_csv(csv_path), fresh)
        write_csv(csv_path, rows)
    except (DownloadSnapshotError, OSError, ValueError) as exc:
        print(f"snapshot failed: {exc}", file=sys.stderr)
        return 1
    print(summary(package, rows))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by workflow invocation
    raise SystemExit(main())
