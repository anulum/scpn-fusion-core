# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the HIL ADC/DAC sensor/actuator interface."""

import numpy as np
import pytest

from scpn_fusion.control.hil_sensors import ADCConfig, DACConfig, SensorInterface


class TestSensorInterface:
    def test_adc_quantization(self) -> None:
        sensor = SensorInterface(adc=ADCConfig(resolution_bits=12))
        reading = sensor.read_adc(0.5)
        assert isinstance(reading, float)
        # 12-bit in ±1.5V → LSB ~0.73 mV
        assert abs(reading - 0.5) < 0.01  # within ~10 LSBs (noise)

    def test_adc_clamps_range(self) -> None:
        sensor = SensorInterface()
        # Beyond range should clamp
        reading = sensor.read_adc(10.0)  # way above ±1.5V
        assert reading <= 1.5 + 0.01

    def test_dac_slew_rate(self) -> None:
        sensor = SensorInterface(dac=DACConfig(slew_rate_v_per_us=50.0))
        # Large step should be slew-limited
        out = sensor.write_dac(10.0, dt_us=0.1)
        assert out < 10.0  # can't reach 10V in 0.1 us at 50V/us

    def test_magnetic_probe(self) -> None:
        sensor = SensorInterface(rng_seed=42)
        B = sensor.read_magnetic_probe(5.3)
        assert abs(B - 5.3) < 0.5  # reasonable noise level

    def test_coil_current(self) -> None:
        sensor = SensorInterface()
        I = sensor.write_coil_current(25.0, dt_us=1000.0)
        assert abs(I - 25.0) < 5.0  # slew-limited approach

    def test_adc_deterministic_with_seed(self) -> None:
        s1 = SensorInterface(rng_seed=42)
        s2 = SensorInterface(rng_seed=42)
        r1 = s1.read_adc(0.5)
        r2 = s2.read_adc(0.5)
        assert r1 == r2

    def test_dac_holds_last_output_on_nan(self) -> None:
        """A NaN command must be held, not latched — one bad sample can't poison the DAC."""
        sensor = SensorInterface()
        good = sensor.write_dac(3.0)
        assert good == pytest.approx(3.0)
        held = sensor.write_dac(float("nan"))
        assert held == pytest.approx(good)  # last valid output held
        assert sensor.dac_faults == 1
        # The converter state is NOT poisoned: a following valid command works.
        recovered = sensor.write_dac(3.0)
        assert np.isfinite(recovered)
        assert recovered == pytest.approx(3.0)

    def test_dac_holds_last_output_on_inf(self) -> None:
        sensor = SensorInterface()
        sensor.write_dac(2.0)
        held = sensor.write_dac(float("inf"))
        assert np.isfinite(held)
        assert held == pytest.approx(2.0)
        assert sensor.dac_faults == 1

    def test_dac_clamps_large_command_to_range(self) -> None:
        """A huge finite command saturates at the ±10 V DAC range, never beyond."""
        sensor = SensorInterface()
        out = sensor.write_dac(1.0e6, dt_us=1000.0)  # slew allows reaching the clamp
        assert out == pytest.approx(10.0)
        assert sensor.dac_faults == 0
        neg = sensor.write_dac(-1.0e6, dt_us=1000.0)
        assert neg == pytest.approx(-10.0)


class TestConverterConfig:
    def test_adc_levels_and_lsb(self) -> None:
        cfg = ADCConfig(resolution_bits=12, voltage_range=(-1.5, 1.5))
        assert cfg.n_levels == (1 << 12) - 1
        assert cfg.lsb_voltage == (1.5 - (-1.5)) / cfg.n_levels

    def test_dac_defaults(self) -> None:
        cfg = DACConfig()
        assert cfg.resolution_bits == 16
        assert cfg.voltage_range == (-10.0, 10.0)
        assert cfg.slew_rate_v_per_us == 50.0
