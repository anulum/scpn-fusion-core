# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Sandpile Fusion Reactor Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest

from scpn_fusion.core.sandpile_fusion_reactor import TokamakSandpile


def test_relax_tracks_edge_loss_events() -> None:
    reactor = TokamakSandpile(size=6)
    reactor.Z[-1] = 5

    avalanche = reactor.relax(suppression_strength=0.0)
    assert avalanche > 0
    assert reactor.last_edge_loss_events > 0
    assert reactor.edge_loss_events >= reactor.last_edge_loss_events


def test_relax_resets_last_edge_loss_counter_per_step() -> None:
    reactor = TokamakSandpile(size=6)
    reactor.Z[:] = 0
    reactor.relax(suppression_strength=0.0)
    assert reactor.last_edge_loss_events == 0


def test_relax_rejects_invalid_suppression_strength() -> None:
    reactor = TokamakSandpile(size=8)
    with pytest.raises(ValueError, match="suppression_strength"):
        reactor.relax(suppression_strength=1.5)
