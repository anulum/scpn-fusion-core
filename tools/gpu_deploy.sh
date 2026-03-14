#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GPU Training Deployment Script
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
#
# Full-pipeline deployment for GPU training on a fresh Ubuntu 24.04
# server with NVIDIA GPU + CUDA 12.x (e.g. UpCloud L40S 48GB).
#
# Usage:
#   bash tools/gpu_deploy.sh              # Full pipeline
#   bash tools/gpu_deploy.sh --quick      # Smoke test only
#   bash tools/gpu_deploy.sh --skip-data  # Skip download (data exists)
#
# Prerequisites:
#   - Ubuntu 24.04 with NVIDIA/CUDA template
#   - nvidia-smi working (driver installed)
#   - Python 3.11+ available
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

QUICK=false
SKIP_DATA=false
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=true ;;
        --skip-data) SKIP_DATA=true ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

step() { echo -e "\n${CYAN}=== $1 ===${NC}"; }
ok()   { echo -e "${GREEN}  OK: $1${NC}"; }
warn() { echo -e "${YELLOW}  WARN: $1${NC}"; }
fail() { echo -e "${RED}  FAIL: $1${NC}"; exit 1; }

# ── Step 0: System checks ────────────────────────────────────────
step "Step 0: System Checks"

command -v python3 >/dev/null || fail "python3 not found"
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python: $PYVER"

if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    ok "NVIDIA GPU detected"
else
    warn "nvidia-smi not found — training will run on CPU (slow)"
fi

# ── Step 1: Create venv + install ────────────────────────────────
step "Step 1: Python Environment"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    ok "Created virtualenv"
fi

source .venv/bin/activate
pip install --upgrade "pip==25.0.1" "wheel==0.45.1" "setuptools==78.1.0" -q

# Install with GPU + ML + dev extras
pip install --no-deps -e . -q && pip install --require-hashes -r requirements/ci-py312.txt -q 2>&1 | tail -3
ok "Installed scpn-fusion[gpu,ml,dev]"

# ── Step 2: Verify GPU ──────────────────────────────────────────
step "Step 2: GPU Verification"

python3 tools/check_gpu.py || warn "GPU not detected by JAX — check CUDA installation"

JAX_GPU=$(python3 -c "
import jax
gpu = any(d.platform == 'gpu' for d in jax.devices())
print('yes' if gpu else 'no')
" 2>/dev/null || echo "no")

if [ "$JAX_GPU" = "yes" ]; then
    ok "JAX GPU backend confirmed"
else
    warn "JAX running on CPU — check: pip install 'jax[cuda12]'"
fi

# ── Step 3: Download + preprocess data ───────────────────────────
step "Step 3: Data Pipeline"

if [ "$SKIP_DATA" = true ]; then
    warn "Skipping data download (--skip-data)"
elif [ -f "data/qlknn10d_processed/train.npz" ]; then
    ok "QLKNN-10D processed data already exists"
else
    echo "  Downloading QLKNN-10D from Zenodo (~12 GB)..."
    python3 tools/download_qlknn10d.py

    MAX_SAMPLES=500000
    if [ "$QUICK" = true ]; then MAX_SAMPLES=10000; fi

    echo "  Processing to NPZ (max $MAX_SAMPLES samples)..."
    python3 tools/qlknn10d_to_npz.py --max-samples "$MAX_SAMPLES"
    ok "QLKNN-10D preprocessed"
fi

# ── Step 4: Train Neural Transport (primary model) ───────────────
step "Step 4: Neural Transport Training"

if [ "$QUICK" = true ]; then
    echo "  Quick mode: 10 epochs, 100 samples"
    python3 tools/train_neural_transport_qlknn.py --quick
else
    echo "  Full training: 500 epochs, cosine annealing, early stopping"
    python3 tools/train_neural_transport_qlknn.py \
        --epochs 500 \
        --lr 3e-4 \
        --batch-size 4096 \
        --hidden-dims 256,128,64 \
        --patience 50 \
        --gated \
        --regime-balance
fi

if [ -f "weights/neural_transport_qlknn.npz" ]; then
    ok "Neural transport weights saved"
    ls -lh weights/neural_transport_qlknn.npz
else
    fail "Neural transport training did not produce weights"
fi

# ── Step 5: Generate FNO spatial data ────────────────────────────
step "Step 5: FNO Spatial Data Generation"

N_EQ=200
if [ "$QUICK" = true ]; then N_EQ=20; fi

if [ -f "data/fno_qlknn_spatial/train.npz" ] && [ "$QUICK" = false ]; then
    ok "FNO spatial data already exists"
else
    python3 tools/generate_fno_qlknn_spatial.py \
        --n-equilibria "$N_EQ" \
        --grid-size 64
    ok "FNO spatial data generated ($N_EQ equilibria)"
fi

# ── Step 6: Train FNO Spatial ────────────────────────────────────
step "Step 6: FNO Spatial Training"

FNO_EPOCHS=200
if [ "$QUICK" = true ]; then FNO_EPOCHS=20; fi

python3 tools/train_fno_qlknn_spatial.py \
    --epochs "$FNO_EPOCHS" \
    --modes 16 \
    --width 64 \
    --lr 1e-3 \
    --batch-size 32

if [ -f "weights/fno_turbulence_jax.npz" ]; then
    ok "FNO spatial weights saved"
    ls -lh weights/fno_turbulence_jax.npz
else
    warn "FNO training did not produce weights (may have failed gate)"
fi

# ── Step 7: Neural Equilibrium (if SPARC data available) ────────
step "Step 7: Neural Equilibrium Training"

SPARC_DIR="validation/reference_data/sparc"
GEQDSK_COUNT=$(find "$SPARC_DIR" -name "*.geqdsk" -o -name "*.eqdsk" 2>/dev/null | wc -l || echo 0)

if [ "$GEQDSK_COUNT" -ge 1 ]; then
    echo "  Found $GEQDSK_COUNT GEQDSK files — training neural equilibrium"
    python3 -c "
from src.scpn_fusion.core.neural_equilibrium_training import train_on_sparc
result = train_on_sparc()
print(f'Test NRMSE: {result.test_nrmse:.6f}')
print(f'Training time: {result.training_time_s:.1f}s')
"
    ok "Neural equilibrium trained"
else
    warn "No SPARC GEQDSK files found in $SPARC_DIR — skipping"
fi

# ── Step 8: Validation ───────────────────────────────────────────
step "Step 8: Validation"

echo "  Running test suite..."
python3 -m pytest tests/ -x -q --timeout=120 -k "not slow and not stress" 2>&1 | tail -5

echo "  Running transport validation..."
python3 -c "
from validation.collect_results import main as collect
collect()
print('Validation results collected.')
" 2>/dev/null || warn "Validation collect skipped (optional)"

# ── Summary ──────────────────────────────────────────────────────
step "DEPLOYMENT COMPLETE"

echo ""
echo "  Weights produced:"
for w in weights/*.npz; do
    [ -f "$w" ] && echo "    $(ls -lh "$w" | awk '{print $5, $NF}')"
done
echo ""

if [ "$QUICK" = true ]; then
    echo -e "${YELLOW}  Quick mode — re-run without --quick for full training${NC}"
fi

echo -e "${GREEN}  All GPU training pipelines executed successfully.${NC}"
