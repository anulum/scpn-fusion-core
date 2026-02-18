from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_transport_power_balance.py"
SPEC = importlib.util.spec_from_file_location("benchmark_transport_power_balance", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import contract guard
    raise RuntimeError(f"Failed to load module at {MODULE_PATH}")
transport_power_balance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport_power_balance)


def test_run_benchmark_passes_thresholds_smoke() -> None:
    report = transport_power_balance.run_benchmark(
        config_path=str(ROOT / "iter_config.json"),
        powers_mw=[20.0, 50.0],
    )
    g = report["transport_power_balance_benchmark"]
    assert g["n_cases"] == 4
    assert g["passes_thresholds"] is True
    assert g["max_relative_error"] <= g["threshold_max_relative_error"]


def test_render_markdown_contains_required_sections() -> None:
    report = transport_power_balance.run_benchmark(
        config_path=str(ROOT / "iter_config.json"),
        powers_mw=[30.0],
    )
    text = transport_power_balance.render_markdown(report)
    assert "# Transport Power-Balance Benchmark" in text
    assert "Max relative error" in text
    assert "| Mode | P_aux [MW] |" in text


def test_main_writes_reports(tmp_path: Path, monkeypatch) -> None:
    out_json = tmp_path / "transport_power_balance_benchmark.json"
    out_md = tmp_path / "transport_power_balance_benchmark.md"
    monkeypatch.setattr(
        transport_power_balance.sys,
        "argv",
        [
            "benchmark_transport_power_balance.py",
            "--config",
            str(ROOT / "iter_config.json"),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
    )
    rc = transport_power_balance.main()
    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "transport_power_balance_benchmark" in payload
