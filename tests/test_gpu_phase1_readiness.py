# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
"""Tests for the GPU Phase 1 readiness gate."""

from __future__ import annotations

import json
from pathlib import Path

from validation.benchmark_gpu_phase1_readiness import (
    evaluate_gpu_phase1_readiness,
    render_markdown,
)


def _write_gpu_surfaces(root: Path) -> None:
    files = [
        "scpn-fusion-rs/crates/fusion-gpu/Cargo.toml",
        "scpn-fusion-rs/crates/fusion-gpu/src/lib.rs",
        "scpn-fusion-rs/crates/fusion-gpu/src/gs_solver.wgsl",
        "scpn-fusion-rs/crates/fusion-gpu/benches/gpu_sor_bench.rs",
        "scpn-fusion-rs/crates/fusion-math/src/sor.rs",
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fixture\n", encoding="utf-8")


def test_gpu_phase1_blocks_without_tracked_benchmark_artifact() -> None:
    report = evaluate_gpu_phase1_readiness(benchmark_report_paths=[Path("missing-gpu.json")])

    assert report["phase1_static_implementation_ready"] is True
    assert report["accepted_phase1_readiness"] is False
    assert "tracked_gpu_physical_wgpu_sor_benchmark_artifact_missing" in report["blockers"]


def test_gpu_phase1_accepts_complete_static_and_benchmark_contract(tmp_path: Path) -> None:
    _write_gpu_surfaces(tmp_path)
    artifact = tmp_path / "gpu.json"
    artifact.write_text(
        json.dumps(
            {
                "gpu_available": True,
                "physical_gpu_adapter": True,
                "solver": "wgpu_sor",
                "output_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_gpu_phase1_readiness(root=tmp_path, benchmark_report_paths=[artifact])

    assert report["accepted_phase1_readiness"] is True
    assert report["blockers"] == []


def test_gpu_phase1_markdown_exposes_blockers() -> None:
    report = evaluate_gpu_phase1_readiness(benchmark_report_paths=[Path("missing-gpu.json")])
    markdown = render_markdown(report)

    assert "tracked_gpu_physical_wgpu_sor_benchmark_artifact_missing" in markdown
    assert "Production scaling ready: `False`" in markdown
