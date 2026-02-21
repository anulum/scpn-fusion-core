# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Transport Validation (QLKNN-10D)
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Validate the neural transport surrogate against real QLKNN-10D test data.

Loads trained weights and held-out test set, computes relative L2 error
overall and per-output/per-regime, and compares against published
QLKNN-10D accuracy benchmarks.

Usage
-----
    python validation/validate_transport_qlknn.py
    python validation/validate_transport_qlknn.py --weights weights/neural_transport_qlknn.npz

Exit codes: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "neural_transport_qlknn.npz"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "qlknn10d_processed"
BENCHMARKS_PATH = REPO_ROOT / "validation" / "reference_data" / "qlknn10d_published_benchmarks.json"

OUTPUT_NAMES = ["chi_e", "chi_i", "D_e"]
REGIME_NAMES = {0: "stable", 1: "ITG", 2: "TEM"}


def _relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.sum((pred - target) ** 2) / max(np.sum(target ** 2), 1e-8)))


def _rmse_pct(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mean_target = float(np.mean(np.abs(target)))
    return rmse / max(mean_target, 1e-8) * 100


def validate(
    weights_path: Path,
    data_dir: Path,
) -> bool:
    """Run full validation and return True if PASS."""
    print("=== SCPN Fusion Core — Neural Transport Validation ===\n")

    # Load model
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from scpn_fusion.core.neural_transport import NeuralTransportModel

    model = NeuralTransportModel(weights_path)
    if not model.is_neural:
        print(f"FAIL: Could not load neural weights from {weights_path}")
        return False
    print(f"Model loaded: {weights_path}")
    print(f"  Checksum: {model.weights_checksum}")

    # Load test data
    test_path = data_dir / "test.npz"
    if not test_path.exists():
        print(f"FAIL: Test data not found at {test_path}")
        print("  Run: python tools/qlknn10d_to_npz.py")
        return False

    test_data = np.load(test_path)
    X_test = test_data["X"]
    Y_test = test_data["Y"]
    regimes = test_data.get("regimes", np.zeros(len(X_test), dtype=np.int32))

    print(f"Test data: {X_test.shape[0]:,} samples\n")

    # Run inference
    from scpn_fusion.core.neural_transport import _mlp_forward
    preds = _mlp_forward(X_test, model._weights)

    # ── Overall metrics ──────────────────────────────────────────
    overall_l2 = _relative_l2(preds, Y_test)
    overall_rmse_pct = _rmse_pct(preds, Y_test)

    print(f"Overall relative L2:  {overall_l2:.4f}")
    print(f"Overall RMSE %:       {overall_rmse_pct:.1f}%")

    # ── Per-output metrics ───────────────────────────────────────
    print("\nPer-output:")
    per_output_l2 = []
    for i, name in enumerate(OUTPUT_NAMES):
        l2 = _relative_l2(preds[:, i], Y_test[:, i])
        rmse = _rmse_pct(preds[:, i], Y_test[:, i])
        per_output_l2.append(l2)
        status = "PASS" if l2 < 0.10 else "WARN" if l2 < 0.20 else "FAIL"
        print(f"  {name:6s}: rel_L2={l2:.4f}, RMSE%={rmse:.1f}%  [{status}]")

    # ── Per-regime metrics ───────────────────────────────────────
    print("\nPer-regime:")
    for regime_id, regime_name in REGIME_NAMES.items():
        mask = regimes == regime_id
        if mask.sum() == 0:
            continue
        l2 = _relative_l2(preds[mask], Y_test[mask])
        n = mask.sum()
        print(f"  {regime_name:7s}: rel_L2={l2:.4f}  (n={n:,})")

    # ── Compare with published benchmarks ────────────────────────
    if BENCHMARKS_PATH.exists():
        benchmarks = json.loads(BENCHMARKS_PATH.read_text())
        print(f"\nPublished reference ({benchmarks.get('source', 'unknown')}):")
        for flux, metrics in benchmarks.get("metrics", {}).items():
            print(f"  {flux}: RMSE%={metrics.get('rmse_pct', '?')}%, R2={metrics.get('r2', '?')}")
        print(f"\nOur RMSE%: {overall_rmse_pct:.1f}% (published reference: ~5-8%)")

    # ── Verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if overall_l2 < 0.05:
        print(f"PASS: relative L2 = {overall_l2:.4f} < 0.05 (excellent)")
        verdict = True
    elif overall_l2 < 0.10:
        print(f"WARN: relative L2 = {overall_l2:.4f} < 0.10 (acceptable)")
        verdict = True
    else:
        print(f"FAIL: relative L2 = {overall_l2:.4f} >= 0.10")
        verdict = False
    print("=" * 50)

    # Save validation report
    report = {
        "weights": str(weights_path),
        "checksum": model.weights_checksum,
        "n_test": len(X_test),
        "overall_relative_l2": overall_l2,
        "overall_rmse_pct": overall_rmse_pct,
        "per_output_relative_l2": {
            name: per_output_l2[i] for i, name in enumerate(OUTPUT_NAMES)
        },
        "verdict": "PASS" if verdict else "FAIL",
    }
    report_path = REPO_ROOT / "validation" / "reports" / "transport_qlknn_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}")

    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    ok = validate(args.weights, args.data_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
