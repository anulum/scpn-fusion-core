# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RealtimeMonitor h5py import guard test
"""Coverage for save_hdf5 h5py ImportError guard (lines 177-178)."""

from __future__ import annotations

import sys

import pytest

from scpn_fusion.phase.realtime_monitor import RealtimeMonitor


class TestSaveHDF5ImportGuard:
    def test_h5py_missing_raises(self, tmp_path, monkeypatch):
        """save_hdf5 without h5py raises ImportError (lines 177-178)."""
        monitor = RealtimeMonitor.from_paper27(L=3, N_per=10, seed=42)
        for _ in range(3):
            monitor.tick()

        monkeypatch.setitem(sys.modules, "h5py", None)
        with pytest.raises(ImportError, match="h5py"):
            monitor.save_hdf5(tmp_path / "test.h5")
