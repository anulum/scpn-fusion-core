# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HIL Sensor/Actuator Interface
"""ADC/DAC sensor and actuator interface for the HIL test harness.

Provides the abstract data-acquisition layer used by the control loop and
benchmark campaigns:

- :class:`ADCConfig` / :class:`DACConfig` — converter configuration.
- :class:`SensorInterface` — ADC quantisation, Gaussian measurement noise, and
  slew-rate-limited DAC output, plus magnetic-probe / coil-current adapters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ADCConfig:
    """ADC (Analog-to-Digital Converter) configuration."""

    resolution_bits: int = 12
    voltage_range: tuple[float, float] = (-1.5, 1.5)
    noise_rms_lsb: float = 0.5

    @property
    def n_levels(self) -> int:
        """Return the maximum integer ADC code for this resolution."""
        return (1 << self.resolution_bits) - 1

    @property
    def lsb_voltage(self) -> float:
        """Return volts represented by one least-significant bit."""
        vmin, vmax = self.voltage_range
        return (vmax - vmin) / self.n_levels


@dataclass(frozen=True)
class DACConfig:
    """DAC (Digital-to-Analog Converter) configuration."""

    resolution_bits: int = 16
    voltage_range: tuple[float, float] = (-10.0, 10.0)
    slew_rate_v_per_us: float = 50.0


class SensorInterface:
    """Abstract sensor/actuator interface with ADC quantization and noise.

    Simulates realistic data acquisition:
    - ADC quantization (configurable bit depth)
    - Gaussian measurement noise
    - DAC output with slew-rate limiting
    """

    def __init__(
        self,
        adc: ADCConfig | None = None,
        dac: DACConfig | None = None,
        *,
        rng_seed: int = 42,
    ) -> None:
        self.adc = adc or ADCConfig()
        self.dac = dac or DACConfig()
        self._rng = np.random.default_rng(rng_seed)
        self._last_dac_voltage = 0.0
        self._last_dac_time_us = 0.0
        self.dac_faults = 0

    def read_adc(self, true_voltage: float) -> float:
        """Quantize and add noise to simulate ADC reading."""
        vmin, vmax = self.adc.voltage_range
        v = float(np.clip(true_voltage, vmin, vmax))

        # Add noise
        noise = self._rng.normal(0.0, self.adc.noise_rms_lsb * self.adc.lsb_voltage)
        v += noise

        # Quantize
        code = round((v - vmin) / (vmax - vmin) * self.adc.n_levels)
        code = int(np.clip(code, 0, self.adc.n_levels))
        quantized = vmin + code * self.adc.lsb_voltage
        return quantized

    def write_dac(self, target_voltage: float, dt_us: float = 1.0) -> float:
        """Apply range-clamped, slew-rate-limited DAC output.

        The converter is the actuator boundary: the returned value is the
        voltage physically applied, and callers MUST drive the plant with this
        value rather than the raw command. A non-finite command (NaN/inf) is a
        fault the hardware cannot emit, so the last valid output is held
        (fail-safe hold) and the fault is counted rather than latched into the
        converter state — one bad sample can never poison the DAC.
        """
        if not np.isfinite(target_voltage):
            self.dac_faults += 1
            return self._last_dac_voltage

        vmin, vmax = self.dac.voltage_range
        target = float(np.clip(target_voltage, vmin, vmax))

        max_change = self.dac.slew_rate_v_per_us * dt_us
        delta = target - self._last_dac_voltage
        if abs(delta) > max_change:
            delta = np.sign(delta) * max_change

        output = self._last_dac_voltage + delta
        self._last_dac_voltage = output
        return output

    def read_magnetic_probe(self, B_true_tesla: float) -> float:
        """Read magnetic field probe via ADC (±1.5V maps to ±10T)."""
        voltage = B_true_tesla * (1.5 / 10.0)
        return self.read_adc(voltage) * (10.0 / 1.5)

    def write_coil_current(self, target_ka: float, dt_us: float = 1.0) -> float:
        """Command coil current via DAC (±10V maps to ±50kA)."""
        voltage = target_ka * (10.0 / 50.0)
        output_v = self.write_dac(voltage, dt_us)
        return output_v * (50.0 / 10.0)
