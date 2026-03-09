#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GPU FNO Retrain Session
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
#
# Run on UpCloud/JarvisLabs GPU instance:
#   bash tools/gpu_fno_retrain.sh
#
# Prerequisites: NVIDIA GPU with >=16GB VRAM, CUDA 12+
# Estimated time: ~30min data gen + ~2h training (L40S/A100)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== SCPN Fusion Core — FNO Retrain Session ==="
echo "Repo root: $REPO_ROOT"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ── Step 0: Environment ────────────────────────────────────────────
echo "--- Step 0: Environment setup ---"
pip install -e ".[dev]" 2>/dev/null || pip install -e .
pip install "jax[cuda12]" 2>/dev/null || pip install jax jaxlib

python -c "
import jax
devs = jax.devices()
gpu = any(d.platform == 'gpu' for d in devs)
print(f'JAX devices: {devs}')
print(f'GPU available: {gpu}')
if not gpu:
    print('WARNING: No GPU detected. Training will be slow.')
"
echo ""

# ── Step 1: Generate spatial data (CPU, ~30min) ───────────────────
echo "--- Step 1: Generate 5000 equilibria with B8_wide QLKNN oracle ---"
N_EQ=${N_EQUILIBRIA:-5000}
GRID=${GRID_SIZE:-64}

if [ -f "data/fno_qlknn_spatial/train.npz" ] && [ -f "data/fno_qlknn_spatial/metadata.json" ]; then
    EXISTING=$(python -c "import json; d=json.load(open('data/fno_qlknn_spatial/metadata.json')); print(d.get('n_equilibria', 0))")
    if [ "$EXISTING" -ge "$N_EQ" ]; then
        echo "  Spatial data already exists ($EXISTING equilibria). Skipping generation."
    else
        echo "  Existing data has $EXISTING equilibria, regenerating with $N_EQ..."
        python tools/generate_fno_qlknn_spatial.py \
            --weights weights/neural_transport_qlknn.npz \
            --n-equilibria "$N_EQ" \
            --grid-size "$GRID" \
            --seed 42
    fi
else
    python tools/generate_fno_qlknn_spatial.py \
        --weights weights/neural_transport_qlknn.npz \
        --n-equilibria "$N_EQ" \
        --grid-size "$GRID" \
        --seed 42
fi
echo ""

# ── Step 2: Train FNO (GPU, ~1-2h) ───────────────────────────────
echo "--- Step 2: Train JAX FNO (modes=24, width=128, 4 layers) ---"
MODES=${FNO_MODES:-24}
WIDTH=${FNO_WIDTH:-128}
LAYERS=${FNO_LAYERS:-4}
EPOCHS=${FNO_EPOCHS:-1500}
LR=${FNO_LR:-5e-4}
BATCH=${FNO_BATCH:-32}

python tools/train_fno_qlknn_spatial.py \
    --modes "$MODES" \
    --width "$WIDTH" \
    --n-layers "$LAYERS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH" \
    --seed 42

echo ""

# ── Step 3: Verify ────────────────────────────────────────────────
echo "--- Step 3: Post-training verification ---"
python -c "
import json, sys
from pathlib import Path
mp = Path('weights/fno_turbulence_jax.metrics.json')
if not mp.exists():
    print('ERROR: metrics file not found')
    sys.exit(1)
m = json.loads(mp.read_text())
vl2 = m['val_relative_l2']
print(f'val_relative_l2: {vl2:.4f}')
print(f'Target: <0.20')
print(f'Result: {\"PASS\" if vl2 < 0.20 else \"FAIL\"} (hard gate <0.40)')
if vl2 < 0.20:
    print('FNO retrain SUCCEEDED — target <0.20 achieved')
elif vl2 < 0.40:
    print('FNO retrain ACCEPTABLE — below hard gate 0.40')
else:
    print('FNO retrain FAILED — above hard gate 0.40')
    sys.exit(1)
"

echo ""
echo "=== FNO Retrain Complete ==="
echo "Weights: weights/fno_turbulence_jax.npz"
echo "Metrics: weights/fno_turbulence_jax.metrics.json"
echo ""
echo "Next: copy weights back and run full validation:"
echo "  python tools/run_python_preflight.py"
echo "  python validation/collect_results.py"
