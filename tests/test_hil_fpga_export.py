# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the FPGA SNN register-level export."""

from scpn_fusion.control.hil_fpga_export import (
    FPGARegisterMap,
    FPGASNNExport,
    SNNNeuronConfig,
)


class TestFPGASNNExport:
    def test_register_map_generation(self) -> None:
        exporter = FPGASNNExport(n_neurons=50, n_channels=2)
        reg_map = exporter.generate_register_map()
        assert isinstance(reg_map, FPGARegisterMap)
        assert reg_map.n_neurons == 100  # 50 * 2 channels
        assert len(reg_map.neurons) == 100
        assert len(reg_map.input_ports) == 2
        assert len(reg_map.output_ports) == 2

    def test_verilog_header(self) -> None:
        exporter = FPGASNNExport(n_neurons=10, n_channels=2, clock_mhz=100.0)
        reg_map = exporter.generate_register_map()
        verilog = exporter.export_verilog_header(reg_map)
        assert "module snn_controller" in verilog
        assert "N_NEURONS" in verilog
        assert "CLK_HZ" in verilog
        assert "v_mem" in verilog
        assert "always @(posedge clk)" in verilog
        assert "placeholder" not in verilog
        assert "endmodule" in verilog

    def test_neuron_configs(self) -> None:
        exporter = FPGASNNExport(n_neurons=5, n_channels=1)
        reg_map = exporter.generate_register_map(v_threshold=0.4, tau_mem_us=20000.0)
        for neuron in reg_map.neurons:
            assert isinstance(neuron, SNNNeuronConfig)
            assert neuron.v_threshold == 0.4
            assert neuron.tau_mem_us == 20000.0

    def test_clock_frequency(self) -> None:
        exporter = FPGASNNExport(clock_mhz=200.0)
        reg_map = exporter.generate_register_map()
        assert reg_map.clock_hz == 200_000_000


class TestFPGAExportEdgeCases:
    def test_generate_register_map_zero_neurons_per_channel(self) -> None:
        # Covers the inner "n_neurons == 0" loop-skip arc: channels iterate but
        # no neuron slots are emitted.
        reg_map = FPGASNNExport(n_neurons=0, n_channels=2).generate_register_map()
        assert reg_map.neurons == []
        assert len(reg_map.input_ports) == 2

    def test_generate_register_map_zero_channels(self) -> None:
        # Covers the outer "n_channels == 0" loop-skip arc.
        reg_map = FPGASNNExport(n_neurons=4, n_channels=0).generate_register_map()
        assert reg_map.neurons == []
        assert reg_map.input_ports == []
        assert reg_map.output_ports == []

    def test_verilog_header_empty_register_map(self) -> None:
        # An empty map exercises every loop-skip arc and the
        # "no neurons → default SNNNeuronConfig" else branch of the header.
        empty = FPGARegisterMap(
            clock_hz=100_000_000,
            n_neurons=0,
            dt_us=1000.0,
            neurons=[],
            input_ports=[],
            output_ports=[],
        )
        verilog = FPGASNNExport(n_neurons=0, n_channels=0).export_verilog_header(empty)
        assert "module snn_controller" in verilog
        assert "V_THRESHOLD" in verilog
        assert "endmodule" in verilog

    def test_verilog_header_active_high_reset(self) -> None:
        # reset_active_low=False selects the active-high reset port/condition.
        exporter = FPGASNNExport(n_neurons=3, n_channels=1)
        reg_map = exporter.generate_register_map()
        reg_map.reset_active_low = False
        verilog = exporter.export_verilog_header(reg_map)
        assert "input  wire rst," in verilog
        assert "if (rst) begin" in verilog
        assert "rst_n" not in verilog
