#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
# Full-fidelity benchmark bundle runner for cloud instances.
#
# This runner is fail-recording, not fail-silent: every benchmark command gets a
# log, exit code, and manifest entry. It does not delete intermediate artifacts.
# Existing validation/reports contents are copied into the timestamped artifact
# root after each stage so tracked report updates are preserved for retrieval.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

for arg in "$@"; do
  case "$arg" in
    *=*) export "$arg" ;;
    *)
      echo "Unsupported argument '$arg'. Use KEY=value arguments only." >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
RUN_ID="${RUN_ID:-full_fidelity_$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-benchmark_runs/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ARTIFACT_ROOT}/logs}"
REPORT_SNAPSHOT_DIR="${REPORT_SNAPSHOT_DIR:-${ARTIFACT_ROOT}/validation_reports}"
SUMMARY_JSON="${ARTIFACT_ROOT}/summary.json"
ARCHIVE_PATH="${ARTIFACT_ROOT}.tar.gz"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-diagnostic}"

mkdir -p "$LOG_DIR" "$REPORT_SNAPSHOT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e ".[benchmark,gpu,rust]"

snapshot_reports() {
  local stage="$1"
  local dest="$REPORT_SNAPSHOT_DIR/$stage"
  mkdir -p "$dest"
  if [[ -d validation/reports ]]; then
    cp -a validation/reports/. "$dest/" || true
  fi
  if [[ -d artifacts ]]; then
    mkdir -p "$ARTIFACT_ROOT/artifacts/$stage"
    cp -a artifacts/. "$ARTIFACT_ROOT/artifacts/$stage/" || true
  fi
}

append_summary() {
  local name="$1"
  local command_text="$2"
  local status="$3"
  local elapsed="$4"
  local log_path="$5"
  "$VENV_DIR/bin/python" - <<PY
import json
import pathlib
import time
path = pathlib.Path("${SUMMARY_JSON}")
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
else:
    data = {
        "run_id": "${RUN_ID}",
        "profile": "${BENCHMARK_PROFILE}",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": __import__("os").popen("git rev-parse HEAD 2>/dev/null").read().strip(),
        "benchmarks": [],
    }
data["benchmarks"].append({
    "name": "${name}",
    "command": "${command_text}",
    "exit_code": int("${status}"),
    "elapsed_seconds": float("${elapsed}"),
    "log": "${log_path}",
})
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

run_step() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/${name}.log"
  local start end status elapsed
  start="$(date +%s)"
  echo "=== ${name} ===" | tee "$log_path"
  printf 'command:' | tee -a "$log_path"
  printf ' %q' "$@" | tee -a "$log_path"
  printf '\n' | tee -a "$log_path"
  set +e
  "$@" 2>&1 | tee -a "$log_path"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  elapsed="$((end - start))"
  append_summary "$name" "$*" "$status" "$elapsed" "$log_path"
  snapshot_reports "$name"
  return 0
}

archive_artifacts() {
  tar -czf "$ARCHIVE_PATH" "$ARTIFACT_ROOT" 2>/dev/null || true
  echo "artifact_root=$ARTIFACT_ROOT"
  echo "artifact_archive=$ARCHIVE_PATH"
}
trap archive_artifacts EXIT

{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  lscpu | sed -n '1,30p'
  nvidia-smi || true
  "$VENV_DIR/bin/python" --version
  "$VENV_DIR/bin/python" - <<'PY'
try:
    import jax
    print("jax", jax.__version__)
    print("jax_devices", jax.devices())
except Exception as exc:
    print("jax_unavailable", type(exc).__name__, exc)
PY
} | tee "$LOG_DIR/hardware_metadata.log"

case "$BENCHMARK_PROFILE" in
  diagnostic)
    run_step gk_em_fidelity "$VENV_DIR/bin/python" validation/benchmark_gk_electromagnetic_fidelity.py
    run_step production_decomposition "$VENV_DIR/bin/python" validation/benchmark_production_decomposition_contract.py
    run_step runaway_contract "$VENV_DIR/bin/python" validation/benchmark_runaway_dream_contract.py
    run_step impurity_contract "$VENV_DIR/bin/python" validation/benchmark_impurity_transport_contract.py
    run_step full_fidelity_acceptance "$VENV_DIR/bin/python" validation/benchmark_full_fidelity_acceptance.py
    ;;
  extended)
    run_step full_fidelity_end_to_end "$VENV_DIR/bin/python" validation/full_fidelity_end_to_end_campaign.py
    run_step free_boundary_tracking "$VENV_DIR/bin/python" validation/free_boundary_tracking_acceptance.py
    run_step disturbance_rejection "$VENV_DIR/bin/python" validation/benchmark_disturbance_rejection.py --strict-hinf
    run_step code_to_code "$VENV_DIR/bin/python" validation/code_to_code_benchmark.py
    run_step collect_results "$VENV_DIR/bin/python" validation/collect_results.py --quick
    ;;
  all)
    BENCHMARK_PROFILE=diagnostic "$0" RUN_ID="${RUN_ID}_diagnostic"
    BENCHMARK_PROFILE=extended "$0" RUN_ID="${RUN_ID}_extended"
    ;;
  *)
    echo "Unknown BENCHMARK_PROFILE=$BENCHMARK_PROFILE" >&2
    exit 2
    ;;
esac

echo "Full-fidelity benchmark bundle finished. Inspect ${SUMMARY_JSON}."
