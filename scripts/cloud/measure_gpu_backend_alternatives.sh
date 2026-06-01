#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GPU Backend Alternatives Cloud Runner
#
# Cloud runner for fail-closed WGPU-vs-CUDA/JAX backend measurements.

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
RUN_ID="${RUN_ID:-gpu_backend_alternatives_$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-benchmark_runs/${RUN_ID}}"
REPORT_DIR="${REPORT_DIR:-${ARTIFACT_ROOT}/reports}"
LOG_DIR="${LOG_DIR:-${ARTIFACT_ROOT}/logs}"
JAX_SIZE="${JAX_SIZE:-256}"
JAX_REPEATS="${JAX_REPEATS:-5}"
WGPU_TIMEOUT_S="${WGPU_TIMEOUT_S:-300}"

mkdir -p "$REPORT_DIR" "$LOG_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e ".[benchmark,gpu,rust]"

{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  nvidia-smi || true
  vulkaninfo --summary || true
  "$VENV_DIR/bin/python" - <<'PY'
try:
    import jax
    print("jax", jax.__version__)
    print("jax_devices", jax.devices())
except Exception as exc:
    print("jax_unavailable", type(exc).__name__, exc)
PY
} > "$LOG_DIR/backend_probe.txt" 2>&1

"$VENV_DIR/bin/python" validation/benchmark_gpu_backend_alternatives.py \
  --json "$REPORT_DIR/gpu_backend_alternatives.json" \
  --markdown "$REPORT_DIR/gpu_backend_alternatives.md" \
  --jax-size "$JAX_SIZE" \
  --jax-repeats "$JAX_REPEATS" \
  --wgpu-timeout-s "$WGPU_TIMEOUT_S" \
  | tee "$LOG_DIR/gpu_backend_alternatives.log"

tar -czf "${ARTIFACT_ROOT}.tar.gz" "$ARTIFACT_ROOT"
echo "artifact_root=$ARTIFACT_ROOT"
echo "artifact_archive=${ARTIFACT_ROOT}.tar.gz"
