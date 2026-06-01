#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Native solver benchmark bundle for cloud/local GPU instances.
#
# Runs Rust Criterion benches for solver kernels and selected Python solver
# parity/validation benchmarks. Every command is logged and the entire artifact
# tree is archived on exit.

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
RUN_ID="${RUN_ID:-solver_$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-benchmark_runs/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ARTIFACT_ROOT}/logs}"
REPORT_SNAPSHOT_DIR="${REPORT_SNAPSHOT_DIR:-${ARTIFACT_ROOT}/validation_reports}"
CRITERION_SNAPSHOT_DIR="${CRITERION_SNAPSHOT_DIR:-${ARTIFACT_ROOT}/criterion}"
SUMMARY_JSON="${ARTIFACT_ROOT}/summary.json"
ARCHIVE_PATH="${ARTIFACT_ROOT}.tar.gz"
SOLVER_PROFILE="${SOLVER_PROFILE:-diagnostic}"

mkdir -p "$LOG_DIR" "$REPORT_SNAPSHOT_DIR" "$CRITERION_SNAPSHOT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e ".[benchmark,gpu,rust]"

snapshot_outputs() {
  local stage="$1"
  mkdir -p "$REPORT_SNAPSHOT_DIR/$stage" "$CRITERION_SNAPSHOT_DIR/$stage"
  if [[ -d validation/reports ]]; then
    cp -a validation/reports/. "$REPORT_SNAPSHOT_DIR/$stage/" || true
  fi
  if [[ -d artifacts ]]; then
    mkdir -p "$ARTIFACT_ROOT/artifacts/$stage"
    cp -a artifacts/. "$ARTIFACT_ROOT/artifacts/$stage/" || true
  fi
  if [[ -d scpn-fusion-rs/target/criterion ]]; then
    cp -a scpn-fusion-rs/target/criterion/. "$CRITERION_SNAPSHOT_DIR/$stage/" || true
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
        "profile": "${SOLVER_PROFILE}",
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
  snapshot_outputs "$name"
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
  rustc --version || true
  cargo --version || true
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

case "$SOLVER_PROFILE" in
  diagnostic)
    run_step rust_fusion_core_benches cargo bench --manifest-path scpn-fusion-rs/crates/fusion-core/Cargo.toml --bench picard_bench --bench vacuum_bench --bench source_bench
    run_step rust_fusion_math_benches cargo bench --manifest-path scpn-fusion-rs/crates/fusion-math/Cargo.toml --bench sor_bench --bench multigrid_bench --bench gmres_bench
    run_step rust_fusion_gpu_bench cargo bench --manifest-path scpn-fusion-rs/crates/fusion-gpu/Cargo.toml --bench gpu_sor_bench
    run_step freegs_parity "$VENV_DIR/bin/python" validation/benchmark_vs_freegs.py
    run_step sparc_geqdsk_rmse "$VENV_DIR/bin/python" validation/benchmark_sparc_geqdsk_rmse.py
    run_step free_boundary_strict_parity "$VENV_DIR/bin/python" validation/benchmark_free_boundary_strict_parity.py
    ;;
  extended)
    run_step rust_fusion_core_all cargo bench --manifest-path scpn-fusion-rs/crates/fusion-core/Cargo.toml
    run_step rust_fusion_math_all cargo bench --manifest-path scpn-fusion-rs/crates/fusion-math/Cargo.toml
    run_step rust_fusion_gpu_all cargo bench --manifest-path scpn-fusion-rs/crates/fusion-gpu/Cargo.toml
    run_step rust_fusion_physics_all cargo bench --manifest-path scpn-fusion-rs/crates/fusion-physics/Cargo.toml
    run_step rust_fusion_ml_all cargo bench --manifest-path scpn-fusion-rs/crates/fusion-ml/Cargo.toml
    run_step code_to_code "$VENV_DIR/bin/python" validation/code_to_code_benchmark.py
    run_step full_fidelity_acceptance "$VENV_DIR/bin/python" validation/benchmark_full_fidelity_acceptance.py
    run_step free_boundary_strict_parity "$VENV_DIR/bin/python" validation/benchmark_free_boundary_strict_parity.py
    ;;
  *)
    echo "Unknown SOLVER_PROFILE=$SOLVER_PROFILE" >&2
    exit 2
    ;;
esac

echo "Solver benchmark bundle finished. Inspect ${SUMMARY_JSON}."
