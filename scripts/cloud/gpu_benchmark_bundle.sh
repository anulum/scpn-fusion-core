#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
# Cloud GPU benchmark bundle runner.
#
# This script is intentionally provider-neutral. Provision the GPU instance by
# the provider UI/API, clone or sync this repository, then run this script from
# the repository root. Credentials must be supplied by the provider environment
# or private vault files; do not commit credentials into this repository.

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
STRESS_EPISODES="${STRESS_EPISODES:-1000}"
STRESS_SHOT_DURATION="${STRESS_SHOT_DURATION:-30}"
STRESS_CONTROLLERS="${STRESS_CONTROLLERS:-PID,H-infinity,LQR}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-benchmark_runs/${RUN_ID}}"
REPORT_DIR="${REPORT_DIR:-${ARTIFACT_ROOT}/reports}"
LOG_DIR="${LOG_DIR:-${ARTIFACT_ROOT}/logs}"
MANIFEST_PATH="${ARTIFACT_ROOT}/manifest.json"
ARCHIVE_PATH="${ARTIFACT_ROOT}.tar.gz"

mkdir -p "$REPORT_DIR" "$LOG_DIR"

write_manifest() {
  "$VENV_DIR/bin/python" - <<PY
import json
import os
import pathlib
import time

root = pathlib.Path("${ARTIFACT_ROOT}")
payload = {
    "run_id": "${RUN_ID}",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "artifact_root": str(root),
    "report_dir": "${REPORT_DIR}",
    "log_dir": "${LOG_DIR}",
    "stress_episodes": int("${STRESS_EPISODES}"),
    "stress_shot_duration_s": int("${STRESS_SHOT_DURATION}"),
    "stress_controllers": "${STRESS_CONTROLLERS}".split(","),
    "git_sha": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
    "git_dirty": bool(os.popen("git status --short 2>/dev/null").read().strip()),
}
path = pathlib.Path("${MANIFEST_PATH}")
path.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
PY
}

archive_artifacts() {
  tar -czf "$ARCHIVE_PATH" "$ARTIFACT_ROOT" 2>/dev/null || true
  echo "artifact_root=$ARTIFACT_ROOT"
  echo "artifact_archive=$ARCHIVE_PATH"
}

trap archive_artifacts EXIT

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e ".[benchmark,gpu,rust]"
write_manifest

{
  echo "=== hardware ==="
  date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
  uname -a
  lscpu | sed -n '1,20p'
  nvidia-smi || true
  echo
  echo "=== python ==="
  "$VENV_DIR/bin/python" --version
  "$VENV_DIR/bin/python" - <<'PY'
try:
    import jax
    print("jax", jax.__version__)
    print("jax_devices", jax.devices())
except Exception as exc:
    print("jax_unavailable", type(exc).__name__, exc)
PY
} | tee "$LOG_DIR/hardware_metadata.txt"

SCPN_ENABLE_HINF_RESEARCH=1 "$VENV_DIR/bin/python" validation/stress_test_campaign.py \
  --episodes "$STRESS_EPISODES" \
  --shot-duration "$STRESS_SHOT_DURATION" \
  --enable-hinf-research \
  --controllers "$STRESS_CONTROLLERS" \
  --output "$REPORT_DIR/stress_test_campaign.json" \
  | tee "$LOG_DIR/stress_test_campaign.log"

"$VENV_DIR/bin/python" validation/scpn_end_to_end_latency.py \
  | tee "$LOG_DIR/scpn_end_to_end_latency.log"

"$VENV_DIR/bin/python" validation/collect_results.py --quick \
  | tee "$LOG_DIR/collect_results_quick.log"

echo "Cloud GPU benchmark bundle complete."
