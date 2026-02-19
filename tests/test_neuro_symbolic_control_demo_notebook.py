from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "examples" / "neuro_symbolic_control_demo_v2.ipynb"
PUBLISHED_NOTEBOOK_PATH = REPO_ROOT / "examples" / "neuro_symbolic_control_demo.ipynb"
LEGACY_SILVER_PATH = REPO_ROOT / "examples" / "neuro_symbolic_control_demo_silver_base.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _stream_lines(nb: dict) -> list[str]:
    lines: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            lines.extend(str(text).splitlines())
    return lines


def _payload_between_markers(lines: list[str], marker: str) -> dict:
    start = f"{marker}_START"
    end = f"{marker}_END"
    start_idx = lines.index(start)
    end_idx = lines.index(end)
    payload = "\n".join(lines[start_idx + 1 : end_idx])
    return json.loads(payload)


def _replay_payload(lines: list[str]) -> tuple[dict, str]:
    start_idx = lines.index("Deterministic replay payload:")
    idx = start_idx + 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    assert idx < len(lines)
    assert lines[idx].strip() == "{"

    block: list[str] = []
    depth = 0
    while idx < len(lines):
        line = lines[idx]
        depth += line.count("{")
        depth -= line.count("}")
        block.append(line)
        if depth == 0:
            break
        idx += 1
    payload = json.loads("\n".join(block))

    proof_hash = ""
    for line in lines[idx + 1 : idx + 12]:
        if line.startswith("Replay proof SHA256:"):
            proof_hash = line.split(":", 1)[1].strip()
            break
    assert proof_hash
    return payload, proof_hash


def test_golden_notebook_exists_and_replaces_legacy_silver_file() -> None:
    assert NOTEBOOK_PATH.exists()
    assert PUBLISHED_NOTEBOOK_PATH.exists()
    assert not LEGACY_SILVER_PATH.exists()


def test_golden_notebook_has_required_contract_text() -> None:
    nb = _load_notebook()
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "# Neuro-Symbolic Control Demo (Golden Base v2)" in text
    assert "Version: v2 (2026-02-19)" in text
    assert "Concepts: Copyright 1996-2026" in text
    assert "Code: Copyright 2024-2026" in text
    assert "sc_neurocore" in text
    assert "stochastic path" in text.lower()
    assert "validation/validate_real_shots.py" in text
    assert "Open-loop" in text
    assert "MPC-lite" in text
    assert "SILVER_BASE" not in text
    assert "silver_base" not in text

    # Ensure notebook has executable code cells.
    executed = [
        c
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code" and c.get("execution_count") is not None
    ]
    assert len(executed) >= 8


def test_golden_notebook_embeds_plots_and_metrics_payloads() -> None:
    nb = _load_notebook()
    png_count = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data", {}) if isinstance(output, dict) else {}
            if "image/png" in data:
                png_count += 1
    assert png_count >= 4

    lines = _stream_lines(nb)
    metrics = _payload_between_markers(lines, "GOLDEN_BASE_METRICS_JSON")
    scale = _payload_between_markers(lines, "GOLDEN_BASE_SCALE_JSON")

    for controller in ("SNN", "PID", "MPC-lite"):
        assert controller in metrics
        rec = metrics[controller]
        assert rec["rmse_total"] >= 0.0
        assert rec["violations_total"] >= 0
        assert rec["critical_total"] >= 0

    # Guard against obvious SNN collapse (previously catastrophic at ~0.48 RMSE).
    assert metrics["SNN"]["rmse_total"] < 0.1
    assert metrics["SNN"]["critical_total"] == 0

    assert scale["shape"] == [3456, 3456]
    assert scale["runs"] >= 3
    assert scale["mean_ms"] > 0.0
    assert scale["p95_ms"] > 0.0


def test_golden_notebook_replay_payload_is_deterministic() -> None:
    nb = _load_notebook()
    lines = _stream_lines(nb)
    payload, proof_hash = _replay_payload(lines)
    assert payload["hash_equal"] is True
    assert payload["state_equal"] is True
    assert payload["action_equal"] is True
    assert payload["run_hash_a"] == payload["run_hash_b"]
    assert len(str(payload["artifact_sha256"])) >= 32
    assert len(proof_hash) >= 32
