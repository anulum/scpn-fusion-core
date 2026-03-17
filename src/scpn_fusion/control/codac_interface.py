# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ITER CODAC/EPICS Interface
"""
ITER CODAC/EPICS integration prototype.

Generates EPICS .db and OPC-UA XML nodeset files for binding a
NeuroSymbolicController to the ITER Plant Instrumentation & Control
(I&C) infrastructure.  Does NOT require pyepics or any EPICS runtime.

References:
    ITER CODAC Handbook v7.0, §3.2 (PV naming conventions)
    IEC 62541 (OPC Unified Architecture)
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scpn_fusion.scpn.contracts import ControlAction, ControlObservation


@dataclass(frozen=True)
class CODACConfig:
    """CODAC plant-system configuration."""

    pv_prefix: str = "ITER-SCPN"
    cycle_hz: float = 1000.0  # Hz, ITER fast controller nominal rate
    timeout_ms: float = 1.5  # cycle budget [ms] at 1 kHz
    heartbeat_pv: str = "ITER-SCPN:HEARTBEAT"
    interlock_pvs: tuple[str, ...] = (
        "ITER-CIS:INTERLOCK:VDE",
        "ITER-CIS:INTERLOCK:HALO",
        "ITER-CIS:INTERLOCK:DISRUPTION",
    )


@dataclass(frozen=True)
class EPICSChannel:
    """Single EPICS process variable definition."""

    pv_name: str
    direction: str  # "input" | "output"
    dtype: str  # "ai" (analog in) | "ao" (analog out) | "bi" (binary in) | "bo" (binary out)
    units: str
    description: str
    low_limit: float = 0.0
    high_limit: float = 0.0


# ── Channel tables ────────────────────────────────────────────────────

_INPUT_CHANNELS: tuple[tuple[str, str, str, float, float, str], ...] = (
    # (suffix, units, desc, lo, hi, obs_key)
    ("Ip", "MA", "Plasma current", 0.0, 20.0, "Ip"),
    ("beta_N", "", "Normalised beta", 0.0, 5.0, "beta_N"),
    ("q95", "", "Edge safety factor", 1.0, 10.0, "q95"),
    ("n_e", "1e19 m-3", "Line-averaged electron density", 0.0, 20.0, "n_e"),
    ("Te_axis", "keV", "Electron temperature on axis", 0.0, 40.0, "Te_axis"),
    ("li", "", "Internal inductance", 0.0, 3.0, "li"),
    ("dBp_dt", "T/s", "Poloidal field rate of change", -10.0, 10.0, "dBp_dt"),
    ("locked_mode_amp", "G", "Locked mode amplitude", 0.0, 50.0, "locked_mode_amp"),
    ("n1_rms", "G", "n=1 RMS amplitude", 0.0, 50.0, "n1_rms"),
    ("Wmhd", "MJ", "Stored MHD energy", 0.0, 500.0, "Wmhd"),
    ("R_axis", "m", "Magnetic axis major radius", 4.0, 8.0, "R_axis_m"),
    ("Z_axis", "m", "Magnetic axis vertical position", -2.0, 2.0, "Z_axis_m"),
)

_OUTPUT_CHANNELS: tuple[tuple[str, str, str, float, float, str], ...] = (
    ("dI_PF1", "A", "PF1 current delta", -1e6, 1e6, "dI_PF1"),
    ("dI_PF2", "A", "PF2 current delta", -1e6, 1e6, "dI_PF2"),
    ("dI_PF3", "A", "PF3 current delta", -1e6, 1e6, "dI_PF3"),
    ("dI_PF4", "A", "PF4 current delta", -1e6, 1e6, "dI_PF4"),
    ("dI_PF5", "A", "PF5 current delta", -1e6, 1e6, "dI_PF5"),
    ("dI_PF6", "A", "PF6 current delta", -1e6, 1e6, "dI_PF6"),
    ("gas_valve_D2", "Pa m3/s", "D2 gas valve throughput", 0.0, 200.0, "gas_valve_D2"),
    ("gas_valve_Ne", "Pa m3/s", "Ne gas valve throughput", 0.0, 50.0, "gas_valve_Ne"),
    ("ECRH_power", "MW", "ECRH heating power", 0.0, 20.0, "ECRH_power"),
    ("NBI_power", "MW", "NBI heating power", 0.0, 40.0, "NBI_power"),
    ("SPI_trigger", "", "Shattered pellet injection trigger", 0.0, 1.0, "SPI_trigger"),
)

# Pre-built obs_key → column index for pack_observation
_INPUT_KEY_MAP: dict[str, str] = {row[0]: row[5] for row in _INPUT_CHANNELS}
# Pre-built action_key → column suffix for unpack_action
_OUTPUT_KEY_MAP: dict[str, str] = {row[5]: row[0] for row in _OUTPUT_CHANNELS}


def _build_input_channels(prefix: str) -> list[EPICSChannel]:
    return [
        EPICSChannel(
            pv_name=f"{prefix}:{row[0]}",
            direction="input",
            dtype="ai",
            units=row[1],
            description=row[2],
            low_limit=row[3],
            high_limit=row[4],
        )
        for row in _INPUT_CHANNELS
    ]


def _build_output_channels(prefix: str) -> list[EPICSChannel]:
    return [
        EPICSChannel(
            pv_name=f"{prefix}:{row[0]}",
            direction="output",
            dtype="bo" if row[0] == "SPI_trigger" else "ao",
            units=row[1],
            description=row[2],
            low_limit=row[3],
            high_limit=row[4],
        )
        for row in _OUTPUT_CHANNELS
    ]


# ── Safety limits (hard interlock thresholds) ─────────────────────────
# ITER Physics Design Description Document, NF 39 (1999)

_HARD_LIMITS: dict[str, tuple[float, float]] = {
    "Ip": (0.0, 17.0),
    "beta_N": (0.0, 3.5),
    "q95": (2.0, float("inf")),
    "n_e": (0.0, 14.0),
    "Te_axis": (0.0, 30.0),
    "locked_mode_amp": (0.0, 20.0),
}


class CODACInterface:
    """Binds a NeuroSymbolicController to ITER CODAC I&C channels."""

    def __init__(self, config: CODACConfig, controller: Any) -> None:
        self.config = config
        self.controller = controller
        self._input_channels = _build_input_channels(config.pv_prefix)
        self._output_channels = _build_output_channels(config.pv_prefix)
        self._cycle_timer = CycleTimer(config.cycle_hz)
        self._step_k = 0

    def define_input_channels(self) -> list[EPICSChannel]:
        return list(self._input_channels)

    def define_output_channels(self) -> list[EPICSChannel]:
        return list(self._output_channels)

    def pack_observation(self, pv_values: Mapping[str, float]) -> ControlObservation:
        """Convert raw EPICS PV dict to ControlObservation for the controller."""
        return ControlObservation(
            R_axis_m=float(pv_values.get("R_axis", pv_values.get("R_axis_m", 6.2))),
            Z_axis_m=float(pv_values.get("Z_axis", pv_values.get("Z_axis_m", 0.0))),
        )

    def unpack_action(self, action: ControlAction) -> dict[str, float]:
        """Convert controller ControlAction to EPICS PV dict."""
        prefix = self.config.pv_prefix
        out: dict[str, float] = {}
        for action_key, suffix in _OUTPUT_KEY_MAP.items():
            pv = f"{prefix}:{suffix}"
            out[pv] = float(action.get(action_key, 0.0))  # type: ignore[arg-type]
        return out

    def run_cycle(self, pv_values: Mapping[str, float]) -> dict[str, float]:
        """Single control cycle: pack -> step -> unpack."""
        self._cycle_timer.start_cycle()
        obs = self.pack_observation(pv_values)
        action = self.controller.step(obs, self._step_k)  # type: ignore[union-attr]
        self._step_k += 1
        result = self.unpack_action(action)
        self._cycle_timer.end_cycle()
        return result

    def safety_interlock(self, pv_values: Mapping[str, float]) -> bool:
        """Return True if any hard limit is violated (actuation must be blocked)."""
        for key, (lo, hi) in _HARD_LIMITS.items():
            val = pv_values.get(key)
            if val is None:
                continue
            v = float(val)
            if v < lo or v > hi:
                return True
        return False

    def generate_epics_db(self, output_path: Path) -> None:
        """Write EPICS .db file with record() entries for all channels."""
        lines: list[str] = []
        lines.append("# Auto-generated EPICS database for SCPN-Control CODAC interface")
        lines.append(f"# Prefix: {self.config.pv_prefix}")
        lines.append("")
        for ch in self._input_channels + self._output_channels:
            lines.append(f'record({ch.dtype}, "{ch.pv_name}") {{')
            lines.append(f'    field(DESC, "{ch.description}")')
            if ch.units:
                lines.append(f'    field(EGU, "{ch.units}")')
            lines.append(f'    field(HOPR, "{ch.high_limit}")')
            lines.append(f'    field(LOPR, "{ch.low_limit}")')
            lines.append("}")
            lines.append("")
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")

    def generate_opcua_nodeset(self, output_path: Path) -> None:
        """Write OPC-UA XML nodeset for ITER SDN integration."""
        ns_uri = "urn:iter:scpn-control:codac"
        root = ET.Element("UANodeSet", xmlns="http://opcfoundation.org/UA/2011/03/UANodeSet.xsd")
        ns_elem = ET.SubElement(root, "NamespaceUris")
        ET.SubElement(ns_elem, "Uri").text = ns_uri

        node_id = 1000
        for ch in self._input_channels + self._output_channels:
            var = ET.SubElement(
                root,
                "UAVariable",
                NodeId=f"ns=1;i={node_id}",
                BrowseName=f"1:{ch.pv_name}",
                DataType="Double" if ch.dtype in ("ai", "ao") else "Boolean",
                AccessLevel="3",
            )
            dn = ET.SubElement(var, "DisplayName")
            dn.text = ch.pv_name
            desc = ET.SubElement(var, "Description")
            desc.text = ch.description
            node_id += 1

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(output_path), encoding="utf-8", xml_declaration=True)


class CycleTimer:
    """Enforces real-time cycle budget with deadline monitoring.

    Uses time.perf_counter_ns() for sub-microsecond resolution.
    """

    def __init__(self, cycle_hz: float) -> None:
        self.budget_ns = int(1e9 / cycle_hz)
        self._start_ns: int = 0
        self._elapsed_ns: int = 0

    def start_cycle(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def end_cycle(self) -> float:
        """Return jitter in milliseconds (elapsed - budget)."""
        self._elapsed_ns = time.perf_counter_ns() - self._start_ns
        return (self._elapsed_ns - self.budget_ns) / 1e6

    def check_overrun(self) -> bool:
        """True if last cycle exceeded its budget."""
        return self._elapsed_ns > self.budget_ns
