# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — CI Workflow Reference Tests
"""Tests for Python file references in the GitHub Actions workflow."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from tools.check_ci_workflow_ownership import workflow_sources
from tools.ci_test_shards import SHARD_COUNT


ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
BENCHMARK_THRESHOLDS = ROOT / "tools" / "benchmark_regression_thresholds.json"
PYTHON_PATH_RE = re.compile(r"\b(?:tests|tools|validation)/[A-Za-z0-9_./-]+\.py\b")


def _workflow_text() -> str:
    """Read every declared CI workflow after validating its ownership graph."""
    return "\n".join(workflow_sources(ROOT).values())


def test_ci_workflow_python_file_references_exist() -> None:
    """All Python paths referenced by the CI workflow exist in the checkout."""
    references = sorted(set(PYTHON_PATH_RE.findall(_workflow_text())))

    assert references
    assert [path for path in references if not (ROOT / path).exists()] == []


def test_ci_workflow_uses_current_torax_gate() -> None:
    """The workflow uses the current real-TORAX parity gate and test module."""
    workflow = _workflow_text()

    assert "validation/benchmark_vs_torax.py" not in workflow
    assert "tests/test_benchmark_vs_torax.py" not in workflow
    assert "validation/benchmark_torax_real_parity.py --check" in workflow
    assert (
        "validation/benchmark_torax_real_parity.py --output artifacts/torax_benchmark.json"
        in workflow
    )
    assert "tests/test_torax_real_parity.py" in workflow


def test_ci_rejects_silent_tracked_evidence_drift_after_preflight() -> None:
    """The release preflight cannot silently rewrite tracked evidence."""
    workflow = _workflow_text()
    preflight = "python tools/run_python_preflight.py --gate release"
    drift_guard = "git diff --exit-code -- artifacts validation/reports"

    assert workflow.count(drift_guard) == 1
    assert workflow.index(preflight) < workflow.index(drift_guard)


def test_ci_uses_local_paths_for_routine_guarded_evidence() -> None:
    """Routine CI benchmarks cannot target protected evidence destinations."""
    workflow = _workflow_text()
    protected_names = (
        "artifacts/vertical_control_replay_benchmark",
        "artifacts/vertical_control_replay_profiles",
        "artifacts/disruption_transfer_generalization",
        "artifacts/scpn_end_to_end_latency_ci",
    )
    local_names = (
        "artifacts/_local_vertical_control_replay_benchmark",
        "artifacts/_local_vertical_control_replay_profiles",
        "artifacts/_local_disruption_transfer_generalization",
        "artifacts/_local_scpn_end_to_end_latency_ci",
    )

    assert all(name not in workflow for name in protected_names)
    assert all(name in workflow for name in local_names)

    thresholds = json.loads(BENCHMARK_THRESHOLDS.read_text(encoding="utf-8"))
    report_paths = {report["id"]: report["path"] for report in thresholds["reports"]}
    assert report_paths["vertical_control_replay"] == (
        "artifacts/_local_vertical_control_replay_benchmark.json"
    )
    assert report_paths["vertical_control_profile_suite"] == (
        "artifacts/_local_vertical_control_replay_profiles.json"
    )
    assert report_paths["disruption_transfer_generalization"] == (
        "artifacts/_local_disruption_transfer_generalization.json"
    )


def test_ci_does_not_rerun_hypothesis_file_after_full_suite() -> None:
    """The sharded full-suite lanes already include the property tests once."""
    workflow = _workflow_text()

    assert "python tools/ci_test_shards.py run --lane" in workflow
    assert "lane_args=(-v)" in workflow
    assert "lane_args=(-q --cov=scpn_fusion --cov-branch --cov-report=)" in workflow
    assert "--timeout-minutes 75" in workflow
    assert "tests/test_hypothesis_properties.py" not in workflow
    assert "--hypothesis-seed" not in workflow


def _python_test_jobs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the jobs of the sharded test workflow and of its union workflow."""
    import yaml

    workflows = ROOT / ".github" / "workflows"
    tests = yaml.load((workflows / "ci-python-tests.yml").read_text(), Loader=yaml.BaseLoader)
    union = yaml.load((workflows / "ci-python-test-union.yml").read_text(), Loader=yaml.BaseLoader)
    return tests["jobs"], union["jobs"]["python-test-union"]


def test_python_test_shards_match_the_scheduler_and_the_union() -> None:
    """Matrix, names, shard arguments and the union's expectations share one shard count."""
    jobs, union = _python_test_jobs()
    count = SHARD_COUNT
    shard_job = jobs["python-tests"]
    lanes = shard_job["strategy"]["matrix"]["python-version"]

    assert lanes == ["3.11", "3.12"]
    assert shard_job["strategy"]["matrix"]["shard"] == [str(index) for index in range(1, count + 1)]
    assert shard_job["name"].endswith(f"shard ${{{{ matrix.shard }}}}/{count}")
    assert jobs["python-test-inventory"]["strategy"]["matrix"]["python-version"] == lanes
    run = next(
        step["run"] for step in shard_job["steps"] if "ci_test_shards.py run" in step.get("run", "")
    )
    assert f'--shard "${{{{ matrix.shard }}}}/{count}"' in run
    assert "--timings tools/ci_test_timings.json" in run
    commands = "\n".join(step.get("run", "") for step in union["steps"])
    assert f"for lane in {' '.join(lanes)}; do" in commands
    assert commands.count(f"--expect-shards {count}") == 2
    assert "python tools/ci_test_shards.py combine --lane 3.12" in commands
    codecov = next(step for step in union["steps"] if step.get("uses", "").startswith("codecov/"))
    assert "if" not in codecov, "Codecov receives the union only after every check passed"


def test_inventory_and_preflight_share_the_shards_environment() -> None:
    """Every job collects or tests in the same set-up, through the native extension build."""
    jobs, _ = _python_test_jobs()

    def setup(job: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a job's steps up to and including the native extension build."""
        steps: list[dict[str, Any]] = job["steps"]
        end = next(
            i
            for i, step in enumerate(steps)
            if step.get("name", "").startswith("Build and install")
        )
        return steps[: end + 1]

    reference = setup(jobs["python-tests"])
    assert reference[1]["name"] == "Checkout pinned DREAM Python API fixture"
    assert all(setup(job) == reference for job in jobs.values())
    preflight = [step.get("name", "") for step in jobs["python-release-gate"]["steps"]]
    assert preflight.index("Python preflight release gate") < preflight.index(
        "Reject silent tracked evidence drift"
    )


def test_ci_materializes_the_pinned_dream_api_for_custody_tests() -> None:
    """Portable custody tests use the exact upstream API, not host-local data."""
    workflow = _workflow_text()
    checkout_start = workflow.index("- name: Checkout pinned DREAM Python API fixture")
    checkout_end = workflow.index(
        "- name: Set up Python ${{ matrix.python-version }}", checkout_start
    )
    checkout = workflow[checkout_start:checkout_end]

    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in checkout
    assert "repository: chalmersplasmatheory/DREAM" in checkout
    assert "ref: ecdd5e146537c77602c9d7cc76b36100200e4b9a" in checkout
    assert "path: data/external/full_fidelity_public_sources/repos/dream" in checkout
    assert "persist-credentials: false" in checkout
    assert checkout_start < workflow.index("Python preflight release gate")
    assert checkout_start < workflow.index("Run this shard of the release test suite")


def test_required_dependency_policy_covers_every_committed_lock() -> None:
    """Required CI checks every Python closure and all three Cargo locks."""
    import yaml

    policy = yaml.load(
        (ROOT / ".github/workflows/ci-dependency-policy.yml").read_text(), Loader=yaml.BaseLoader
    )
    jobs = policy["jobs"]
    python_job = jobs["python-lock-audit"]
    expected = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "requirements").glob("*.txt"))
    assert python_job["strategy"]["matrix"]["lock"] == expected
    assert python_job["strategy"]["fail-fast"] == "false"
    assert (
        "fail-fast: 'false'"
        not in (ROOT / ".github/workflows/ci-dependency-policy.yml").read_text()
    )
    command = python_job["steps"][-1]["run"]
    assert command.endswith(
        'python -m pip_audit --strict --disable-pip --no-deps --requirement "$RUNNER_TEMP/registry-requirements.txt"\n'
    )
    assert python_job["steps"][-1]["env"]["LOCK_FILE"] == "${{ matrix.lock }}"
    assert '"https://api.osv.dev/v1/query"' in command
    assert 'query = {"commit": commit}' in command
    assert python_job["timeout-minutes"] == "10"
    assert "continue-on-error" not in python_job
    rust_job = jobs["rust-audit"]
    assert rust_job["strategy"]["matrix"]["lock"] == [
        "Cargo.lock",
        "fuzz/Cargo.lock",
        "../validation/reference_data/rustbca.Cargo.lock",
    ]
    assert rust_job["strategy"]["fail-fast"] == "false"
    assert rust_job["steps"][-1]["run"] == 'cargo audit --deny warnings --file "${{ matrix.lock }}"'
    assert "continue-on-error" not in rust_job


@pytest.mark.parametrize("name", ["cfspopcon", "process"])
@pytest.mark.parametrize(
    "outcome",
    [
        "clean",
        "paginated",
        "advisory",
        "http-error",
        "invalid-json",
        "api-error",
        "invalid-list",
        "cycle",
        "wrong-pin",
        "unknown-source",
    ],
)
def test_source_pinned_closure_queries_and_refusals(
    tmp_path: Path, name: str, outcome: str
) -> None:
    """The actual CI preparation queries exact commits and refuses incomplete audits."""
    import yaml

    policy = yaml.safe_load((ROOT / ".github/workflows/ci-dependency-policy.yml").read_text())
    command = policy["jobs"]["python-lock-audit"]["steps"][-1]["run"]
    script = command.split("python - <<'PYTHON'\n", 1)[1].split("\nPYTHON\n", 1)[0]
    source = next(
        line
        for line in (ROOT / f"requirements/{name}.txt").read_text().splitlines()
        if " @ " in line
    )
    if outcome == "wrong-pin":
        source = source[:-1] + ("0" if source[-1] != "0" else "1")
    elif outcome == "unknown-source":
        source = "unknown" + source[source.index(" @ ") :]
    manifest_path = tmp_path / "validation/reference_data" / f"{name}_source.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes((ROOT / f"validation/reference_data/{name}_source.json").read_bytes())
    commit = json.loads(manifest_path.read_text())["commit"]
    lock = tmp_path / "closure.txt"
    registry = "# complete registry closure\ncertifi==2026.7.22\npackaging==26.3\n"
    lock.write_text(source + "\n" + registry)
    queries: list[object] = []

    class Handler(BaseHTTPRequestHandler):
        """Serve deterministic advisory responses over the actual HTTP boundary."""

        def do_POST(self) -> None:
            """Record the request and return the selected database response."""
            queries.append(
                json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            )
            self.send_response(503 if outcome == "http-error" else 200)
            self.end_headers()
            response = b"{}"
            if outcome == "advisory":
                response = b'{"vulns":[{"id":"test-advisory"}]}'
            elif outcome == "invalid-json":
                response = b"unavailable"
            elif outcome == "api-error":
                response = b'{"error":"unavailable"}'
            elif outcome == "invalid-list":
                response = b'{"vulns":null}'
            elif outcome == "cycle" or (outcome == "paginated" and len(queries) == 1):
                response = b'{"next_page_token":"second-page"}'
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the test HTTP service silent."""

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ
            | {
                "LOCK_FILE": str(lock),
                "RUNNER_TEMP": str(tmp_path),
                "OSV_API_URL": f"http://127.0.0.1:{server.server_port}/v1/query",
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive()
    if outcome in {"wrong-pin", "unknown-source"}:
        assert queries == []
    else:
        assert queries[0] == {"commit": commit}
    output = tmp_path / "registry-requirements.txt"
    if outcome in {"clean", "paginated"}:
        assert result.returncode == 0, result.stderr
        assert output.read_text() == registry
        if outcome == "paginated":
            assert queries == [{"commit": commit}, {"commit": commit, "page_token": "second-page"}]
    else:
        assert result.returncode != 0
        assert not output.exists()
