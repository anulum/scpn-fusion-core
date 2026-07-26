# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — bounded PyPI metrics tests
"""Tests for SCPN Fusion Core's fail-closed PyPI download history."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "tools" / "pypi_downloads.py"
_SPEC = importlib.util.spec_from_file_location("scpn_fusion_pypi_downloads", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
downloads = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(downloads)

_SAMPLE: dict[str, Any] = {
    "data": [
        {"category": "without_mirrors", "date": "2026-07-17", "downloads": 11},
        {"category": "with_mirrors", "date": "2026-07-17", "downloads": 19},
        {"category": "without_mirrors", "date": "2026-07-18", "downloads": 13},
        {"category": "with_mirrors", "date": "2026-07-18", "downloads": 23},
    ],
    "package": "scpn-fusion",
    "type": "overall_downloads",
}


class _FakeResponse:
    """Minimal deterministic HTTP response for the fixed-host transport tests."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"{}",
        content_type: str = "application/json",
    ) -> None:
        self.status = status
        self.reason = "test-reason"
        self.body = body
        self.content_type = content_type

    def read(self, limit: int) -> bytes:
        """Return the fixture body while proving the size bound is supplied."""
        assert limit == downloads.MAX_RESPONSE_BYTES + 1
        return self.body

    def getheader(self, name: str, default: str = "") -> str:
        """Return only the content type used by the production transport."""
        return self.content_type if name == "Content-Type" else default


class _FakeConnection:
    """Minimal deterministic HTTPS connection with observable lifecycle."""

    def __init__(
        self,
        response: _FakeResponse,
        *,
        request_error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.request_error = request_error
        self.request_args: tuple[Any, ...] | None = None
        self.request_kwargs: dict[str, Any] | None = None
        self.closed = False

    def request(self, *args: Any, **kwargs: Any) -> None:
        """Capture request arguments or raise the configured transport error."""
        self.request_args = args
        self.request_kwargs = kwargs
        if self.request_error is not None:
            raise self.request_error

    def getresponse(self) -> _FakeResponse:
        """Return the configured response."""
        return self.response

    def close(self) -> None:
        """Record deterministic connection cleanup."""
        self.closed = True


def _payload_bytes(payload: object) -> bytes:
    """Encode one synthetic remote payload."""
    return json.dumps(payload).encode()


def test_detect_package_reads_project_contract(tmp_path: Path) -> None:
    """PEP 621 metadata must be the package-name authority."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "scpn-fusion"\n', encoding="utf-8")
    assert downloads.detect_package(pyproject) == "scpn-fusion"


def test_project_writer_rejects_every_alternate_target_before_fetch(
    tmp_path: Path,
) -> None:
    """The privileged writer must freeze package and lexical CSV path."""
    downloads.require_project_csv_target("scpn-fusion", Path("downloads/scpn-fusion.csv"))
    for package, csv_path in (
        ("other", Path("downloads/scpn-fusion.csv")),
        ("scpn-fusion", Path("downloads/other.csv")),
        ("scpn-fusion", Path("downloads/../downloads/scpn-fusion.csv")),
        ("scpn-fusion", tmp_path / "downloads/scpn-fusion.csv"),
    ):
        with pytest.raises(ValueError, match="project metrics"):
            downloads.require_project_csv_target(package, csv_path)

    fetched = False

    def unexpected_fetch(package: str) -> bytes:
        nonlocal fetched
        fetched = True
        return b"{}"

    assert (
        downloads.main(
            [
                "--package",
                "other",
                "--csv",
                "downloads/scpn-fusion.csv",
                "--project-csv-only",
            ],
            unexpected_fetch,
        )
        == 1
    )
    assert not fetched


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("[build-system]\nrequires = []\n", "no [project] table"),
        ('[project]\nversion = "1"\n', "no [project] name"),
        ('[project]\nname = ""\n', "no [project] name"),
    ],
)
def test_detect_package_rejects_missing_identity(
    tmp_path: Path,
    document: str,
    message: str,
) -> None:
    """Missing or empty package identities must fail closed."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        downloads.detect_package(pyproject)


def test_package_endpoint_encodes_path_characters() -> None:
    """Untrusted package text must remain inside the fixed API path."""
    assert downloads.package_endpoint_path("package/name") == (
        "/api/packages/package%2Fname/overall"
    )
    with pytest.raises(ValueError, match="must not be empty"):
        downloads.package_endpoint_path("  ")
    assert downloads._valid_date(None) is None
    assert downloads._valid_date(" 2026-07-17") is None
    assert downloads._valid_date("20260717") is None


def test_http_get_uses_fixed_host_bounded_read_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transport must bind its host, request and cleanup lifecycle."""
    response = _FakeResponse(body=b'{"ok": true}', content_type="application/json; charset=utf-8")
    connection = _FakeConnection(response)

    def make_connection(host: str, *, timeout: int) -> _FakeConnection:
        assert host == downloads.PYPISTATS_HOST
        assert timeout == downloads.REQUEST_TIMEOUT_SECONDS
        return connection

    monkeypatch.setattr(downloads.http.client, "HTTPSConnection", make_connection)
    assert downloads._http_get("scpn-fusion") == b'{"ok": true}'
    assert connection.request_args == (
        "GET",
        "/api/packages/scpn-fusion/overall",
    )
    assert connection.request_kwargs == {
        "headers": {
            "Accept": "application/json",
            "User-Agent": "scpn-fusion-metrics/1",
        }
    }
    assert connection.closed


@pytest.mark.parametrize(
    ("response", "error_type", "message"),
    [
        (_FakeResponse(status=429), downloads.RetryableSnapshotError, "transient HTTP 429"),
        (_FakeResponse(status=404), downloads.DownloadSnapshotError, "HTTP 404"),
        (
            _FakeResponse(content_type="text/html"),
            downloads.DownloadSnapshotError,
            "Content-Type",
        ),
        (
            _FakeResponse(body=b"x" * (downloads.MAX_RESPONSE_BYTES + 1)),
            downloads.DownloadSnapshotError,
            "size limit",
        ),
    ],
)
def test_http_get_fails_closed_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeResponse,
    error_type: type[Exception],
    message: str,
) -> None:
    """HTTP status, media type and body-size failures must be explicit."""
    connection = _FakeConnection(response)
    monkeypatch.setattr(
        downloads.http.client,
        "HTTPSConnection",
        lambda host, timeout: connection,
    )
    with pytest.raises(error_type, match=message):
        downloads._http_get("scpn-fusion")
    assert connection.closed


def test_http_get_wraps_transport_failure_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network failures must enter only the bounded transient retry class."""
    connection = _FakeConnection(_FakeResponse(), request_error=OSError("offline"))
    monkeypatch.setattr(
        downloads.http.client,
        "HTTPSConnection",
        lambda host, timeout: connection,
    )
    with pytest.raises(downloads.RetryableSnapshotError, match="request failed"):
        downloads._http_get("scpn-fusion")
    assert connection.closed


@pytest.mark.parametrize("payload", [b"not-json", b"[]", b"null"])
def test_fetch_overall_rejects_invalid_json_contract(payload: bytes) -> None:
    """Only the exact remote JSON object contract is accepted."""
    with pytest.raises(downloads.DownloadSnapshotError):
        downloads.fetch_overall("scpn-fusion", lambda package: payload)


@pytest.mark.parametrize(
    "payload",
    [
        (
            b'{"data":[{"category":"with_mirrors","date":"2026-07-17",'
            b'"downloads":19}],"package":"other","package":"scpn-fusion",'
            b'"type":"overall_downloads"}'
        ),
        (
            b'{"data":[{"category":"with_mirrors","date":"2026-07-17",'
            b'"downloads":1,"downloads":19}],"package":"scpn-fusion",'
            b'"type":"overall_downloads"}'
        ),
    ],
)
def test_duplicate_remote_object_names_fail_before_csv_write(
    payload: bytes,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ambiguous top-level and nested JSON names must never reach persistence."""
    csv_path = tmp_path / "downloads" / "scpn-fusion.csv"
    assert (
        downloads.main(
            ["--package", "scpn-fusion", "--csv", str(csv_path)],
            lambda package: payload,
        )
        == 1
    )
    assert not csv_path.exists()
    assert "duplicate JSON object name" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(extra=True), "top-level keys"),
        (lambda payload: payload.update(package="other"), "package identity"),
        (lambda payload: payload.update(type="recent_downloads"), "type mismatch"),
        (lambda payload: payload.update(data=[]), "non-empty list"),
        (lambda payload: payload["data"].append(None), "object schema"),
        (
            lambda payload: payload["data"][0].update(extra=True),
            "object schema",
        ),
        (
            lambda payload: payload["data"][0].update(category="unknown"),
            "invalid category",
        ),
        (lambda payload: payload["data"][0].update(date="bad"), "invalid date"),
        (
            lambda payload: payload["data"][0].update(downloads="11"),
            "invalid downloads",
        ),
        (lambda payload: payload["data"].append(dict(payload["data"][0])), "duplicates"),
        (lambda payload: payload["data"].pop(), "missing with_mirrors"),
        (
            lambda payload: payload["data"][0].update(downloads=20),
            "mirror-count ordering",
        ),
    ],
)
def test_remote_schema_fails_closed_on_identity_or_partial_data(
    mutate: Any,
    message: str,
) -> None:
    """Wrong identity, malformed rows and partial dates must never be written."""
    payload = json.loads(json.dumps(_SAMPLE))
    mutate(payload)
    with pytest.raises(downloads.DownloadSnapshotError, match=message):
        downloads.fetch_overall("scpn-fusion", lambda package: _payload_bytes(payload))


def test_fetch_overall_returns_complete_validated_rows() -> None:
    """The valid exact remote schema must reduce to complete date rows."""
    assert downloads.fetch_overall("scpn-fusion", lambda package: _payload_bytes(_SAMPLE)) == {
        "2026-07-17": {"without_mirrors": 11, "with_mirrors": 19},
        "2026-07-18": {"without_mirrors": 13, "with_mirrors": 23},
    }


def test_remote_schema_preserves_legitimate_sparse_without_mirrors() -> None:
    """An omitted zero-like series stays absent instead of becoming fabricated data."""
    payload = json.loads(json.dumps(_SAMPLE))
    payload["data"] = [
        row
        for row in payload["data"]
        if not (row["date"] == "2026-07-17" and row["category"] == "without_mirrors")
    ]
    assert downloads.fetch_overall("scpn-fusion", lambda package: _payload_bytes(payload))[
        "2026-07-17"
    ] == {"with_mirrors": 19}


def test_csv_roundtrip_is_sorted_and_schema_fixed(tmp_path: Path) -> None:
    """CSV output must be deterministic and losslessly readable."""
    csv_path = tmp_path / "downloads" / "scpn-fusion.csv"
    rows = {
        "2026-07-18": {"without_mirrors": 13, "with_mirrors": 23},
        "2026-07-17": {"without_mirrors": 11, "with_mirrors": 19},
    }
    downloads.write_csv(csv_path, rows)
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "date,without_mirrors,with_mirrors",
        "2026-07-17,11,19",
        "2026-07-18,13,23",
    ]
    assert downloads.read_csv(csv_path) == rows


def test_csv_roundtrip_preserves_sparse_category_as_blank(tmp_path: Path) -> None:
    """A missing upstream category must round-trip as missing, not numeric zero."""
    csv_path = tmp_path / "downloads" / "scpn-fusion.csv"
    rows = {"2026-07-17": {"with_mirrors": 19}}
    downloads.write_csv(csv_path, rows)
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "date,without_mirrors,with_mirrors",
        "2026-07-17,,19",
    ]
    assert downloads.read_csv(csv_path) == rows


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({"bad": {"with_mirrors": 1}}, "invalid date"),
        ({"2026-07-17": {}}, "invalid categories"),
        ({"2026-07-17": {"other": 1, "with_mirrors": 1}}, "invalid categories"),
        ({"2026-07-17": {"without_mirrors": 1}}, "missing with_mirrors"),
        ({"2026-07-17": {"with_mirrors": True}}, "invalid with_mirrors"),
        (
            {"2026-07-17": {"without_mirrors": 2, "with_mirrors": 1}},
            "mirror-count ordering",
        ),
    ],
)
def test_in_memory_rows_fail_closed(rows: Any, message: str) -> None:
    """Malformed in-memory state must fail before any atomic replacement."""
    with pytest.raises(ValueError, match=message):
        downloads.merge_rows(rows, {})


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("date,with_mirrors\n2026-07-18,2\n", "unexpected CSV header"),
        ("date,without_mirrors,with_mirrors\nbad,1,2\n", "invalid date"),
        (
            "date,without_mirrors,with_mirrors\n2026-07-18,1,2\n2026-07-18,3,4\n",
            "duplicate date",
        ),
        (
            "date,without_mirrors,with_mirrors\n2026-07-18,-1,2\n",
            "invalid without",
        ),
        (
            "date,without_mirrors,with_mirrors\n2026-07-18,01,2\n",
            "invalid without",
        ),
        (
            "date,without_mirrors,with_mirrors\n2026-07-18,nope,2\n",
            "invalid without",
        ),
        (
            "date,without_mirrors,with_mirrors\n2026-07-18,1,2,3\n",
            "unexpected CSV fields",
        ),
    ],
)
def test_read_csv_rejects_corrupt_history(
    tmp_path: Path,
    csv_text: str,
    message: str,
) -> None:
    """Existing history must be strictly validated before merging."""
    csv_path = tmp_path / "series.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        downloads.read_csv(csv_path)


def test_missing_csv_and_empty_summary_are_explicit(tmp_path: Path) -> None:
    """A new branch starts empty without fabricating a history row."""
    assert downloads.read_csv(tmp_path / "missing.csv") == {}
    assert downloads.summary("scpn-fusion", {}) == ("scpn-fusion: no download data available yet")
    assert "without_mirrors=n/a" in downloads.summary(
        "scpn-fusion", {"2026-07-17": {"with_mirrors": 19}}
    )


def test_atomic_write_removes_temporary_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed replacement must not leave a partial temporary series."""
    csv_path = tmp_path / "downloads" / "scpn-fusion.csv"

    def fail_replace(source: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        downloads.write_csv(
            csv_path,
            {"2026-07-17": {"without_mirrors": 11, "with_mirrors": 19}},
        )
    assert list(csv_path.parent.iterdir()) == []


def test_merge_and_main_preserve_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fresh bounded payload must upsert without dropping old dates."""
    csv_path = tmp_path / "downloads" / "scpn-fusion.csv"
    downloads.write_csv(
        csv_path,
        {"2026-07-16": {"without_mirrors": 7, "with_mirrors": 9}},
    )
    assert (
        downloads.main(
            ["--package", "scpn-fusion", "--csv", str(csv_path)],
            lambda package: _payload_bytes(_SAMPLE),
        )
        == 0
    )
    assert set(downloads.read_csv(csv_path)) == {
        "2026-07-16",
        "2026-07-17",
        "2026-07-18",
    }
    assert "latest 2026-07-18" in capsys.readouterr().out


def test_main_failure_keeps_existing_series(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Permanent remote failure must not rewrite existing history."""
    csv_path = tmp_path / "series.csv"
    original = "date,without_mirrors,with_mirrors\n2026-07-16,7,9\n"
    csv_path.write_text(original, encoding="utf-8")

    def fail_fetch(package: str) -> bytes:
        raise downloads.DownloadSnapshotError("offline")

    assert downloads.main(["--package", "scpn-fusion", "--csv", str(csv_path)], fail_fetch) == 1
    assert csv_path.read_text(encoding="utf-8") == original
    assert "snapshot failed: offline" in capsys.readouterr().err


def test_main_print_package_and_requires_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Identity inspection is write-free and snapshot mode requires a target."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "scpn-fusion"\n', encoding="utf-8")
    assert downloads.main(["--pyproject", str(pyproject), "--print-package"]) == 0
    assert capsys.readouterr().out.strip() == "scpn-fusion"
    with pytest.raises(SystemExit) as exc_info:
        downloads.main(["--package", "scpn-fusion"])
    assert exc_info.value.code == 2


def test_transient_failures_retry_then_soft_skip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Transient throttles retry finitely and never corrupt history."""
    calls = 0
    waits: list[float] = []

    def recover(package: str) -> bytes:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise downloads.RetryableSnapshotError("busy")
        return _payload_bytes(_SAMPLE)

    result = downloads.fetch_overall_with_retry("scpn-fusion", recover, waits.append)
    assert result is not None
    assert set(result) == {"2026-07-17", "2026-07-18"}
    assert waits == [15.0, 30.0]

    csv_path = tmp_path / "downloads/scpn-fusion.csv"
    downloads.write_csv(csv_path, {"2026-07-16": {"without_mirrors": 7, "with_mirrors": 9}})

    def unavailable(package: str) -> bytes:
        raise downloads.RetryableSnapshotError("busy")

    waits.clear()
    assert (
        downloads.main(
            ["--package", "scpn-fusion", "--csv", str(csv_path)],
            unavailable,
            waits.append,
        )
        == 0
    )
    assert waits == list(downloads.RETRY_DELAYS)
    assert downloads.MAX_TRANSIENT_WINDOW_SECONDS == 225
    assert downloads.MAX_TRANSIENT_WINDOW_SECONDS < 10 * 60
    assert "snapshot skipped" in capsys.readouterr().err
    assert downloads.read_csv(csv_path) == {"2026-07-16": {"without_mirrors": 7, "with_mirrors": 9}}


def test_workflow_has_one_push_only_bounded_writer_and_pinned_actions() -> None:
    """The scheduled writer must remain exact-path and credential-safe."""
    workflow = (_ROOT / ".github" / "workflows" / "pypi-downloads.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in workflow
    assert workflow.count("contents: write") == 1
    assert "cancel-in-progress: false" in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
    assert 'readonly package="scpn-fusion"' in workflow
    assert 'readonly csv_path="downloads/scpn-fusion.csv"' in workflow
    assert "--project-csv-only" in workflow
    assert 'test "${#tracked_paths[@]}" -eq 1' in workflow
    assert 'test "${tracked_paths[0]}" = "$csv_path"' in workflow
    assert 'test "${#staged_paths[@]}" -eq 1' in workflow
    assert 'test "${staged_paths[0]}" = "$csv_path"' in workflow
    assert 'git -C "$METRICS_DIR" push origin HEAD:refs/heads/metrics' in workflow
    assert workflow.count("GH_TOKEN:") == 1
    assert workflow.index("GH_TOKEN:") > workflow.index("Push only the reviewed metrics ref")
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow
    assert "scpn-fusion-metrics/downloads/scpn-fusion.csv" in workflow
    assert "retention-days: 30" in workflow
    assert "secrets." not in workflow
    action_refs = re.findall(r"uses: [^@\n]+@([^\s#]+)", workflow)
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in action_refs)
