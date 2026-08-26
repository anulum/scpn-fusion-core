# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for official-GACODE TGLF runtime evidence generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.tglf_interface import TGLFInputDeck, TGLFOutput
from validation import benchmark_tglf_gacode_runtime as benchmark


def test_run_benchmark_gates_current_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchmark, "_resolve_tglf_command", lambda command: "/resolved/tglf")

    def _run_tglf(
        deck: TGLFInputDeck,
        *,
        tglf_command: str,
        timeout_s: float,
        work_dir: Path,
        max_retries: int,
    ) -> TGLFOutput:
        del deck, tglf_command, timeout_s, max_retries
        work_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "input.tglf",
            "out.tglf.gbflux",
            "out.tglf.eigenvalue_spectrum",
            "out.tglf.ky_spectrum",
        ):
            (work_dir / name).write_text(f"fixture {name}\n", encoding="utf-8")
        (work_dir / "out.tglf.version").write_text("b4933975 [2026-08-20]\n", encoding="utf-8")
        return TGLFOutput(
            q_i=5.0,
            q_e=3.0,
            particle_e=-0.2,
            particle_i=-0.2,
            gamma_max=0.4,
        )

    monkeypatch.setattr(benchmark, "run_tglf_binary", _run_tglf)
    monkeypatch.setattr(
        benchmark,
        "_parse_gacode_tglf_spectrum",
        lambda _path: (np.array([0.1, 0.4]), np.array([-0.2, 0.3]), np.array([0.1, 1.5])),
    )
    regression_stdout = "\n".join(f"tglf{i:02d}: PASS" for i in range(1, 10))
    monkeypatch.setattr(
        "validation.benchmark_tglf_gacode_runtime.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, regression_stdout, ""),
    )

    report = benchmark.run_benchmark(command="tglf-test", work_dir=tmp_path, timeout_s=5.0)

    assert report["status"] == "PASS"
    assert report["official_regression"]["pass_count"] == 9
    assert report["case"]["spectrum_points"] == 2
    assert report["case"]["dominant_ky"] == 1.5
    markdown = benchmark.render_markdown(report)
    assert "not a cross-solver accuracy claim" in markdown
    assert "orientation only" in markdown


def test_cyclone_like_deck_keeps_public_r_over_l_contract() -> None:
    deck = benchmark._cyclone_like_deck()
    assert deck.R_LTi == 6.9
    assert deck.R_LTe == 6.9
    assert deck.R_major == 2.78
    assert deck.a_minor == 1.0
