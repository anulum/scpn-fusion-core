# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Predictor ROC Analysis
"""
Disruption predictor performance analysis using ROC curves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from scpn_fusion.control.disruption_predictor import (
    predict_disruption_risk,
    simulate_tearing_mode,
)


def generate_scenario_batch(n_total: int = 100) -> list[dict]:
    rng = np.random.default_rng(42)
    shots = []
    modes = ["ntm", "density_limit", "vde"]
    n_disrupt_target = n_total // 2
    n_disrupt, n_safe = 0, 0

    while len(shots) < n_total:
        mode = rng.choice(modes)
        signal, label, t_disrupt = simulate_tearing_mode(steps=500, mode=mode, rng=rng)
        if label == 1 and n_disrupt < n_disrupt_target:
            shots.append({"signal": signal, "label": 1, "t_disrupt": t_disrupt, "mode": mode})
            n_disrupt += 1
        elif label == 0 and n_safe < (n_total - n_disrupt_target):
            shots.append({"signal": signal, "label": 0, "t_disrupt": -1, "mode": "safe"})
            n_safe += 1
    return shots


def evaluate_batch(shots: list[dict], threshold: float) -> dict:
    tp, fp, tn, fn = 0, 0, 0, 0
    for shot in shots:
        signal, label, t_dis_true, mode = (
            shot["signal"],
            shot["label"],
            shot["t_disrupt"],
            shot["mode"],
        )
        win_size = 128
        detected = False
        t_detect = -1
        for t in range(win_size, len(signal), 20):
            window = signal[t - win_size : t]
            val = window[-1]
            obs = {}
            # More realistic scaling to match predictor weights
            if mode == "ntm":
                obs["toroidal_n1_amp"] = val * 0.2
            elif mode == "density_limit":
                obs["toroidal_n2_amp"] = val * 0.1
            elif mode == "vde":
                obs["toroidal_radial_spread"] = val * 1.0
            else:
                obs["toroidal_n1_amp"] = 0.05

            risk = predict_disruption_risk(window, obs)
            if risk > threshold:
                detected = True
                t_detect = t
                break

        if label == 1:
            # TP only if detected BEFORE actual disruption
            if detected and t_detect < t_dis_true:
                tp += 1
            else:
                fn += 1
        else:
            if detected:
                fp += 1
            else:
                tn += 1
    return {"tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)}


def main():
    print("Generating batch...")
    shots = generate_scenario_batch(100)
    thresholds = np.linspace(0.0, 1.0, 51)
    tpr_list, fpr_list = [], []
    print("Sweeping...")
    for th in thresholds:
        res = evaluate_batch(shots, th)
        tpr_list.append(res["tpr"])
        fpr_list.append(res["fpr"])

    # Ensure (0,0) and (1,1)
    if not any(f == 0.0 for f in fpr_list):
        fpr_list.append(0.0)
        tpr_list.append(0.0)
    if not any(f == 1.0 for f in fpr_list):
        fpr_list.append(1.0)
        tpr_list.append(1.0)

    idx = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[idx]
    tpr_sorted = np.array(tpr_list)[idx]

    auc = trapezoid(tpr_sorted, fpr_sorted)

    results = {
        "auc": float(auc),
        "tpr": [float(x) for x in tpr_list],
        "fpr": [float(x) for x in fpr_list],
    }
    report_dir = Path("validation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "disruption_roc.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(report_dir / "disruption_roc.md", "w") as f:
        f.write(f"# Disruption Predictor ROC Analysis\n\n- **AUC**: {auc:.4f}\n")
        f.write(f"Result: {'PASS' if auc > 0.85 else 'FAIL'}\n")
    print(f"AUC={auc:.4f}")


if __name__ == "__main__":
    main()
