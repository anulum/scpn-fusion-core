from __future__ import annotations

import numpy as np

from scpn_fusion.core import lazarus_bridge


class _DummyKernel:
    def __init__(self) -> None:
        self.Psi = np.ones((4, 4), dtype=float)
        self.RR = np.linspace(5.5, 6.5, 4)[None, :].repeat(4, axis=0)
        self.ZZ = np.linspace(-0.5, 0.5, 4)[:, None].repeat(4, axis=1)

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], None]:
        return (6.2, 0.0), None


def test_calculate_bio_resonance_returns_positive_value() -> None:
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    bridge.kernel = _DummyKernel()
    val = bridge.calculate_bio_resonance()
    assert val > 0.0


def test_generate_protocol_embeds_resonance_value() -> None:
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    script = bridge.generate_protocol(1.25)
    assert "Resonance Score: 1.2500" in script
    assert "tert_vol = 6.25" in script


def test_visualize_bridge_saves_plot(monkeypatch) -> None:
    bridge = lazarus_bridge.LazarusBridge.__new__(lazarus_bridge.LazarusBridge)
    bridge.kernel = _DummyKernel()
    saved: list[str] = []
    class _DummyAxes:
        def contour(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

        def plot(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

        def set_title(self, *_args, **_kwargs) -> None:
            return None

        def legend(self, *_args, **_kwargs) -> None:
            return None

    class _DummyFig:
        pass

    def _fake_savefig(path: str) -> None:
        saved.append(path)

    monkeypatch.setattr(lazarus_bridge.plt, "subplots", lambda **kwargs: (_DummyFig(), _DummyAxes()))
    monkeypatch.setattr(lazarus_bridge.plt, "savefig", _fake_savefig)
    bridge.visualize_bridge(1.0)
    assert saved
    assert "Lazarus_Bridge_Result.png" in saved[-1]
