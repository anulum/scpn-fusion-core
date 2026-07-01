#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
# Local CUDA smoke benchmark runner for workstation GPUs.
#
# This is a short-run harness. It verifies CUDA visibility, verifies whether
# JAX is using the GPU, then runs focused controller benchmarks that are short
# enough for local iteration. The H-infinity stress campaign is CPU flight-sim
# work even when CUDA is visible.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
STRESS_EPISODES="${STRESS_EPISODES:-20}"
STRESS_SHOT_DURATION="${STRESS_SHOT_DURATION:-3}"
STRESS_CONTROLLERS="${STRESS_CONTROLLERS:-PID,H-infinity,LQR}"
REPORT_DIR="${REPORT_DIR:-validation/reports}"
LOG_DIR="${LOG_DIR:-benchmark_logs/local_cuda_smoke}"

mkdir -p "$REPORT_DIR" "$LOG_DIR"

nvidia-smi | tee "$LOG_DIR/nvidia_smi.txt"

"$VENV_DIR/bin/python" - <<'PY' | tee "$LOG_DIR/jax_devices.txt"
try:
    import jax
    print("jax", jax.__version__)
    print("devices", jax.devices())
except Exception as exc:
    print("jax_unavailable", type(exc).__name__, exc)
PY

SCPN_ENABLE_HINF_RESEARCH=1 "$VENV_DIR/bin/python" validation/stress_test_campaign.py \
  --episodes "$STRESS_EPISODES" \
  --shot-duration "$STRESS_SHOT_DURATION" \
  --enable-hinf-research \
  --controllers "$STRESS_CONTROLLERS" \
  --output "$REPORT_DIR/stress_test_campaign_local_smoke.json" \
  | tee "$LOG_DIR/stress_test_campaign_local_smoke.log"

echo "Local CUDA smoke benchmark bundle complete."
