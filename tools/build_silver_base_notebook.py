#!/usr/bin/env python3
"""Build the Silver Base notebook variant with updated sections and labels."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def _ensure_replace(text: str, old: str, new: str) -> str:
    if old not in text:
        return text
    return text.replace(old, new, 1)


def main() -> None:
    src = Path("examples/neuro_symbolic_control_demo.ipynb")
    dst = Path("examples/neuro_symbolic_control_demo_silver_base.ipynb")
    if not src.exists():
        raise FileNotFoundError(src)

    nb = nbf.read(src, as_version=4)

    nb.cells[0].source = """# Neuro-Symbolic Control Silver Base Notebook

This Silver Base notebook fuses the `02` neuro-symbolic compiler flow with the `03` equilibrium flow into one end-to-end control demonstration:

1. Petri logic over density + plasma current + beta
2. Compile to SNN (stochastic path when `sc_neurocore` is available)
3. Confirm `sc_neurocore` API path is actually runnable
4. Closed-loop control against a FusionKernel twin
5. Real disturbance injection from a DIII-D shot profile
6. SNN vs PID vs MPC-lite comparison table
7. Formal contracts and proof bundle hashes
8. Computational cost + 3456x3456 scaling benchmark
9. Artifact export + deterministic replay

Copyright clarity:
- Concepts: Copyright 1996-2026
- Code: Copyright 2024-2026
- License: GNU AGPL v3
"""

    nb.cells[1].source = """[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/scpn-fusion-core/blob/main/examples/neuro_symbolic_control_demo_silver_base.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/anulum/scpn-fusion-core/main?labpath=examples%2Fneuro_symbolic_control_demo_silver_base.ipynb)

---
"""

    nb.cells[11].source = """## 5) FusionKernel Twin and Baseline Controllers

The plant twin calls `FusionKernel.solve_equilibrium()` every control step.

Controllers:
- **SNN**: compiled Petri/SNN pipeline
- **PID**: classical baseline
- **MPC-lite**: single-step linear-quadratic baseline (honest naming)
"""

    nb.cells[13].source = """## 6) Closed-Loop Results: SNN vs PID vs MPC-lite

This section executes all three controllers under the same DIII-D disturbance profile and prints a comparison table with controller-only and end-to-end latencies.
"""

    nb.cells[19].source = """## Summary

This Silver Base notebook demonstrates:

- Petri net control logic over density/current/beta
- SNN compilation with stochastic-path support when available
- Explicit `sc_neurocore` API-path run check
- Closed-loop operation with a `FusionKernel` equilibrium twin
- DIII-D disturbance injection from the bundled shot profile
- SNN vs PID/MPC-lite quantitative comparison
- Formal contract checks with proof hashes
- Computational-cost evidence (controller vs full-loop)
- 3456x3456 dense scaling benchmark
- Artifact export and deterministic replay proof

Deployment note:
- This is toy-scale closed-loop inference.
- Production deployment should add sensor preprocessing, actuator lag compensation, and neural equilibrium surrogates to avoid full GS solve on every real-time tick.
"""

    c2 = nb.cells[2].source
    c2 = _ensure_replace(
        c2,
        "import os\n",
        "import os\nimport sys\n",
    )
    c2 = c2.replace(
        "REPO_ROOT = Path.cwd()\nif not (REPO_ROOT / \"validation\").exists() and (REPO_ROOT / \"03_CODE\" / \"SCPN-Fusion-Core\").exists():\n    REPO_ROOT = REPO_ROOT / \"03_CODE\" / \"SCPN-Fusion-Core\"\n",
        "REPO_ROOT = Path.cwd().resolve()\nif not (REPO_ROOT / \"validation\").exists():\n    for candidate in [REPO_ROOT] + list(REPO_ROOT.parents):\n        if (candidate / \"validation\").exists() and (candidate / \"src\").exists():\n            REPO_ROOT = candidate\n            break\n",
    )
    nb.cells[2].source = c2

    c8 = nb.cells[8].source
    if "SC_NEUROCORE_STATUS" not in c8:
        marker = 'print(f"W_out shape: {compiled.W_out.shape}")\n'
        snippet = """

# Confirm sc_neurocore path and a small API call.
SC_NEUROCORE_STATUS = {
    "available": False,
    "module_path": None,
    "version": None,
    "api_check": None,
    "detail": None,
}

candidate_paths = [
    REPO_ROOT.parent / "sc-neurocore" / "src",
    REPO_ROOT / "sc-neurocore" / "src",
    Path.cwd().parent / "sc-neurocore" / "src",
]
for cand in candidate_paths:
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

try:
    import sc_neurocore
    from sc_neurocore import BitstreamEncoder

    encoder = BitstreamEncoder(x_min=0.0, x_max=1.0, length=128, seed=SEED)
    bits = encoder.encode(0.625)
    decoded = encoder.decode(bits)

    SC_NEUROCORE_STATUS.update(
        {
            "available": True,
            "module_path": getattr(sc_neurocore, "__file__", "<unknown>"),
            "version": getattr(sc_neurocore, "__version__", "<unknown>"),
            "api_check": {
                "bits_len": int(bits.shape[0]),
                "bits_mean": float(np.mean(bits)),
                "decoded": float(decoded),
            },
            "detail": "BitstreamEncoder encode/decode path executed.",
        }
    )
except Exception as exc:
    SC_NEUROCORE_STATUS["detail"] = f"sc_neurocore import/API failed: {exc}"

print("sc_neurocore status:")
print(json.dumps(SC_NEUROCORE_STATUS, indent=2, default=str))
"""
        c8 = c8.replace(marker, marker + snippet)
    nb.cells[8].source = c8

    c10 = nb.cells[10].source
    c10 = c10.replace("N_STEPS = 120", "N_STEPS = 240")
    c10 = _ensure_replace(
        c10,
        "print(f\"Disruption index (source): {int(shot['disruption_time_idx'])}\")\n",
        "print(f\"Disruption index (source): {int(shot['disruption_time_idx'])}\")\nprint(f\"Disruption type (source): {str(shot['disruption_type'])}\")\n",
    )
    c10 = c10.replace(
        'plt.suptitle("Injected Real Disturbance from DIII-D")',
        'plt.suptitle("Injected Disturbance Profile (DIII-D Shot Template)")',
    )
    nb.cells[10].source = c10

    c12 = nb.cells[12].source
    c12 = c12.replace("HeroDemo-DIIID-like", "SilverBase-DIIID-like")
    c12 = c12.replace("MPC_B", "MPC_LITE_B")
    c12 = c12.replace("MPC_LAMBDA", "MPC_LITE_LAMBDA")
    c12 = c12.replace(
        "def mpc_action(state, target):",
        "def mpc_lite_action(state, target):\n    # Honest naming: this is a single-step linear-quadratic baseline.",
    )
    c12 = c12.replace('elif controller_name == "MPC":', 'elif controller_name == "MPC-lite":')
    c12 = c12.replace("u = mpc_action(st, TARGET)", "u = mpc_lite_action(st, TARGET)")
    c12 = _ensure_replace(
        c12,
        "lat_ms = np.zeros(N_STEPS, dtype=np.float64)\n",
        "lat_ms = np.zeros(N_STEPS, dtype=np.float64)\n    e2e_lat_ms = np.zeros(N_STEPS, dtype=np.float64)\n    gs_solve_ms = np.zeros(N_STEPS, dtype=np.float64)\n",
    )
    c12 = _ensure_replace(
        c12,
        "for k in range(N_STEPS):\n",
        "for k in range(N_STEPS):\n            step_t0 = time.perf_counter()\n",
    )
    c12 = _ensure_replace(
        c12,
        "solve = self.kernel.solve_equilibrium(preserve_initial_state=True)\n",
        "t_solve0 = time.perf_counter()\n        solve = self.kernel.solve_equilibrium(preserve_initial_state=True)\n        gs_solve_ms = (time.perf_counter() - t_solve0) * 1e3\n",
    )
    c12 = _ensure_replace(
        c12,
        '"gs_residual": float(solve["gs_residual"]),\n',
        '"gs_residual": float(solve["gs_residual"]),\n            "gs_solve_ms": float(gs_solve_ms),\n',
    )
    c12 = _ensure_replace(
        c12,
        "energy_trace[k] = out[\"energy_err\"]\n",
        "energy_trace[k] = out[\"energy_err\"]\n            gs_solve_ms[k] = out[\"gs_solve_ms\"]\n",
    )
    c12 = _ensure_replace(
        c12,
        "                )\n\n        rmse = np.sqrt(np.mean((states - TARGET[np.newaxis, :]) ** 2, axis=0))\n",
        "                )\n\n            e2e_lat_ms[k] = (time.perf_counter() - step_t0) * 1e3\n\n        rmse = np.sqrt(np.mean((states - TARGET[np.newaxis, :]) ** 2, axis=0))\n",
    )
    c12 = _ensure_replace(
        c12,
        '"latency_p95_ms": float(np.percentile(lat_ms, 95)),\n            "latency_mean_ms": float(np.mean(lat_ms)),\n',
        '"latency_p95_ms": float(np.percentile(lat_ms, 95)),\n            "latency_mean_ms": float(np.mean(lat_ms)),\n            "latency_e2e_p95_ms": float(np.percentile(e2e_lat_ms, 95)),\n            "latency_e2e_mean_ms": float(np.mean(e2e_lat_ms)),\n            "gs_solve_p95_ms": float(np.percentile(gs_solve_ms, 95)),\n            "gs_solve_mean_ms": float(np.mean(gs_solve_ms)),\n',
    )
    c12 = _ensure_replace(
        c12,
        '"lat_ms": lat_ms,\n',
        '"lat_ms": lat_ms,\n            "e2e_lat_ms": e2e_lat_ms,\n            "gs_solve_ms": gs_solve_ms,\n',
    )
    nb.cells[12].source = c12

    c14 = nb.cells[14].source
    c14 = c14.replace('"MPC": run_closed_loop("MPC", seed_offset=0),', '"MPC-lite": run_closed_loop("MPC-lite", seed_offset=0),')
    c14 = c14.replace('["SNN", "PID", "MPC"]', '["SNN", "PID", "MPC-lite"]')
    c14 = c14.replace('"MPC": "tab:green"', '"MPC-lite": "tab:green"')
    if "SILVER_BASE_METRICS_JSON_START" not in c14:
        c14 += """

metrics_export = {rec["controller"]: rec for rec in records}
print("SILVER_BASE_METRICS_JSON_START")
print(json.dumps(metrics_export, indent=2, sort_keys=True))
print("SILVER_BASE_METRICS_JSON_END")
"""
    nb.cells[14].source = c14

    c16 = nb.cells[16].source
    c16 = c16.replace('for name in ["SNN", "PID", "MPC"]:', 'for name in ["SNN", "PID", "MPC-lite"]:')
    nb.cells[16].source = c16

    c18 = nb.cells[18].source
    c18 = c18.replace('name="hero_neuro_symbolic_control"', 'name="silver_base_neuro_symbolic_control"')
    nb.cells[18].source = c18

    cost_md = nbf.v4.new_markdown_cell(
        """## 8) Computational Cost and 3456x3456 Scaling

This notebook now reports two latency lanes:

- Controller-only latency: SNN/PID/MPC-lite inference path only
- End-to-end latency: controller + full `FusionKernel.solve_equilibrium()` + contract checks

Operational note:
- Real-time deployment should run a neural equilibrium surrogate in-loop.
- Full GS solve per control tick is computationally expensive and best treated as offline or low-rate supervisory validation.
"""
    )

    cost_code = nbf.v4.new_code_cell(
        """cost_rows = []
for name in ["SNN", "PID", "MPC-lite"]:
    m = results[name]["metrics"]
    gs_frac = 100.0 * m["gs_solve_mean_ms"] / max(m["latency_e2e_mean_ms"], 1e-9)
    cost_rows.append(
        {
            "controller": name,
            "controller_p95_ms": m["latency_p95_ms"],
            "e2e_p95_ms": m["latency_e2e_p95_ms"],
            "gs_solve_p95_ms": m["gs_solve_p95_ms"],
            "controller_mean_ms": m["latency_mean_ms"],
            "e2e_mean_ms": m["latency_e2e_mean_ms"],
            "gs_solve_mean_ms": m["gs_solve_mean_ms"],
            "gs_share_pct": gs_frac,
        }
    )

if HAS_PANDAS:
    df_cost = pd.DataFrame(cost_rows).set_index("controller")
    display(df_cost)
else:
    print(cost_rows)

print("Deployment note: end-to-end loop is dominated by GS equilibrium solve; use neural surrogate for real-time control tick rates.")

# Dense scaling benchmark at 3456x3456 (matrix-vector inference proxy)
SCALE_N = 3456
SCALE_RUNS = 6
rng_scale = np.random.default_rng(SEED + 123)
A = rng_scale.standard_normal((SCALE_N, SCALE_N), dtype=np.float32)
x = rng_scale.standard_normal(SCALE_N, dtype=np.float32)

_ = A @ x  # warmup
scale_ms = []
for _ in range(SCALE_RUNS):
    t0 = time.perf_counter()
    y = A @ x
    scale_ms.append((time.perf_counter() - t0) * 1e3)

scale_metrics = {
    "shape": [SCALE_N, SCALE_N],
    "runs": SCALE_RUNS,
    "mean_ms": float(np.mean(scale_ms)),
    "p95_ms": float(np.percentile(scale_ms, 95)),
    "min_ms": float(np.min(scale_ms)),
    "max_ms": float(np.max(scale_ms)),
    "y_l2": float(np.linalg.norm(y)),
}

print("SILVER_BASE_SCALE_JSON_START")
print(json.dumps(scale_metrics, indent=2, sort_keys=True))
print("SILVER_BASE_SCALE_JSON_END")

if HAS_MPL:
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, SCALE_RUNS + 1), scale_ms, marker="o")
    plt.title("3456x3456 dense matvec latency per run")
    plt.xlabel("run")
    plt.ylabel("latency [ms]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""
    )

    nb.cells.insert(17, cost_md)
    nb.cells.insert(18, cost_code)

    nb.cells[19].source = """## 9) Artifact Export and Deterministic Replay

We export a deployment artifact, reload it, and verify deterministic replay consistency of the SNN closed-loop run.
"""

    nbf.write(nb, dst)
    print(dst)


if __name__ == "__main__":
    main()
