# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Synthetic DIII-D Disruption Generator Tests
"""Contract tests for the synthetic DIII-D disruption profile generator.

These run the real profile generators end to end (no mocking) so the physics
contract is exercised: ``generate_all`` builds all ten seeded shots,
``verify_all`` re-loads and checks the 5-disruption/5-safe split, ``main``
drives both the full and verify-only paths, and the smoothing helper's short-
window guard is covered directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.generate_disruption_profiles import (
    N_STEPS,
    SHOT_MANIFEST,
    _exp_decay,
    _smooth,
    _time_to_idx,
    generate_all,
    main,
    verify_all,
)

_ARRAY_KEYS = (
    "time_s",
    "Ip_MA",
    "BT_T",
    "beta_N",
    "q95",
    "ne_1e19",
    "n1_amp",
    "n2_amp",
    "locked_mode_amp",
    "dBdt_gauss_per_s",
    "vertical_position_m",
)


class TestHelpers:
    """Signal-shaping helpers."""

    def test_smooth_short_window_is_identity(self) -> None:
        """A window below 2 returns the array unchanged."""
        arr = np.array([1.0, 5.0, 2.0], dtype=np.float64)
        assert _smooth(arr, window=1) is arr

    def test_smooth_averages(self) -> None:
        """A normal window smooths towards the local mean."""
        arr = np.zeros(20, dtype=np.float64)
        arr[10] = 7.0
        out = _smooth(arr, window=5)
        assert out.shape == arr.shape
        assert out[10] < 7.0 and out[9] > 0.0

    def test_time_to_idx_bounds(self) -> None:
        """Time-to-index maps the endpoints to the grid bounds."""
        assert _time_to_idx(0.0) == 0
        assert _time_to_idx(3.0) == N_STEPS - 1

    def test_exp_decay_monotone_after_start(self) -> None:
        """Exponential decay is non-increasing after the start index."""
        arr = np.full(N_STEPS, 2.0, dtype=np.float64)
        out = _exp_decay(arr, start_idx=100, tau=0.05)
        assert out[100] == pytest.approx(2.0)
        assert out[-1] < out[100]


class TestGenerateAndVerify:
    """Full generate/verify round-trip over all ten shots."""

    def test_generate_all_writes_ten_shots(self, tmp_path: Path) -> None:
        """generate_all writes one NPZ per manifest entry (verbose path)."""
        paths = generate_all(tmp_path, verbose=True)
        assert len(paths) == len(SHOT_MANIFEST) == 10
        for name, _shot, _gen in SHOT_MANIFEST:
            assert (tmp_path / f"{name}.npz").exists()

    def test_generate_all_quiet(self, tmp_path: Path) -> None:
        """generate_all also runs with verbose disabled."""
        paths = generate_all(tmp_path, verbose=False)
        assert len(paths) == 10

    def test_generated_arrays_have_contract_shape(self, tmp_path: Path) -> None:
        """Every array field is a finite float64 series of N_STEPS."""
        generate_all(tmp_path, verbose=False)
        with np.load(tmp_path / f"{SHOT_MANIFEST[0][0]}.npz") as data:
            for key in _ARRAY_KEYS:
                arr = data[key]
                assert arr.shape == (N_STEPS,)
                assert arr.dtype == np.float64
                assert np.all(np.isfinite(arr))

    def test_verify_all_passes_on_generated(self, tmp_path: Path) -> None:
        """verify_all accepts a freshly generated shot set (5 + 5)."""
        generate_all(tmp_path, verbose=False)
        # Precondition: the full manifest was written before verification.
        assert len(list(tmp_path.glob("*.npz"))) == len(SHOT_MANIFEST)
        # verify_all asserts keys/shapes/dtypes/finiteness internally and raises
        # on any defect; completing without exception is the pass contract.
        verify_all(tmp_path)


class TestMain:
    """CLI entry point."""

    def test_full_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default path generates then verifies."""
        monkeypatch.setattr(
            "sys.argv",
            ["generate_disruption_profiles", "--output-dir", str(tmp_path)],
        )
        main()
        assert len(list(tmp_path.glob("*.npz"))) == 10

    def test_verify_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--verify-only checks an existing set without regenerating."""
        generate_all(tmp_path, verbose=False)
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_disruption_profiles",
                "--output-dir",
                str(tmp_path),
                "--verify-only",
            ],
        )
        main()
