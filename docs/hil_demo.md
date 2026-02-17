# Hardware-in-the-Loop (HIL) Demo Documentation

This document describes the FPGA register map and expected latency for
deploying the SNN controller on an Alveo U250 accelerator card.

## 1. Target Platform

- **FPGA**: Xilinx Alveo U250 (UltraScale+ XCU250)
- **Interface**: PCIe Gen3 x16 or AXI-Lite over JTAG (development)
- **Clock**: 250 MHz fabric clock
- **Estimated inference latency**: < 10 us per control step

## 2. Register Map

The SNN controller exposes a memory-mapped register interface compatible
with the Rust `fusion-control/src/snn.rs` implementation.

| Offset  | Width | Name                  | R/W | Description                                  |
|---------|-------|-----------------------|-----|----------------------------------------------|
| 0x0000  | 32    | CTRL_STATUS           | R   | Bit 0: ready, Bit 1: running, Bit 2: error  |
| 0x0004  | 32    | CTRL_COMMAND          | W   | Bit 0: start, Bit 1: reset, Bit 2: inject   |
| 0x0008  | 32    | CTRL_CONFIG           | R/W | Bit 0: enable_TMR, Bit 1: enable_bitflip    |
| 0x000C  | 32    | TIMESTEP_COUNT        | R   | Number of completed SNN timesteps            |
| 0x0010  | 32    | ERROR_FLAGS           | R   | Bit 0: TMR_mismatch, Bit 1: watchdog_timeout|
| 0x0020  | 32x8  | NEURON_STATE[0..7]    | R   | LIF neuron membrane potentials (Q16.16 fixed)|
| 0x0040  | 32x8  | NEURON_SPIKE[0..7]    | R   | Spike status (1 bit per neuron)              |
| 0x0060  | 32x4  | INPUT_CURRENT[0..3]   | W   | Input sensor currents (Q16.16 fixed-point)   |
| 0x0070  | 32x4  | OUTPUT_COMMAND[0..3]  | R   | Actuator commands (Q16.16 fixed-point)       |
| 0x0080  | 32    | SYNAPSE_WEIGHT_ADDR   | W   | Address for weight read/write                |
| 0x0084  | 32    | SYNAPSE_WEIGHT_DATA   | R/W | Weight value at addressed location           |
| 0x0088  | 32    | WEIGHT_COUNT          | R   | Total number of synaptic weights             |
| 0x0100  | 32    | TMR_VOTER_STATUS      | R   | Triple Modular Redundancy voter output       |
| 0x0104  | 32    | BITFLIP_INJECT_ADDR   | W   | Address for bit-flip fault injection         |
| 0x0108  | 32    | BITFLIP_INJECT_BIT    | W   | Bit index for fault injection                |
| 0x0200  | 32    | LATENCY_CYCLES        | R   | Last inference latency in clock cycles       |
| 0x0204  | 32    | LATENCY_WORST         | R   | Worst-case latency observed                  |

### Fixed-Point Format

All physical quantities use Q16.16 fixed-point (16 integer bits, 16
fractional bits). Conversion:

```python
def float_to_q16_16(x: float) -> int:
    return int(round(x * 65536.0)) & 0xFFFFFFFF

def q16_16_to_float(x: int) -> float:
    if x & 0x80000000:
        x -= 0x100000000
    return x / 65536.0
```

## 3. Pin Mapping

Maps to the SNN neuron pool in `fusion-control/src/snn.rs`:

| FPGA Pin Group  | SNN Signal           | Physical Meaning                |
|-----------------|---------------------|---------------------------------|
| INPUT[0]        | vertical_error      | Z_plasma - Z_target [m]        |
| INPUT[1]        | radial_error        | R_plasma - R_target [m]        |
| INPUT[2]        | beta_n_error        | beta_N - beta_N_target          |
| INPUT[3]        | q95_error           | q95 - q95_target                |
| OUTPUT[0]       | pf_coil_upper_A     | Upper PF coil current command   |
| OUTPUT[1]       | pf_coil_lower_A     | Lower PF coil current command   |
| OUTPUT[2]       | ohmic_coil_A        | Ohmic heating coil command      |
| OUTPUT[3]       | gas_valve_sccm      | Gas puffing rate command        |

## 4. Expected Latency Budget

| Stage                          | Cycles  | Time (@ 250 MHz) |
|--------------------------------|---------|-------------------|
| Input register capture         | 2       | 8 ns              |
| Input scaling (Q16.16 -> LIF)  | 4       | 16 ns             |
| LIF neuron update (8 neurons)  | 16      | 64 ns             |
| Synapse weight multiply-accumulate | 64  | 256 ns            |
| Output decode                  | 4       | 16 ns             |
| TMR voter                      | 3       | 12 ns             |
| Output register write          | 2       | 8 ns              |
| **Total**                      | **95**  | **380 ns**        |

Worst-case with bit-flip recovery: ~1.5 us (TMR re-vote + pipeline flush).

## 5. Bit-Flip Tolerance

The SNN controller uses Triple Modular Redundancy (TMR) for all neuron
state registers. The voter circuit detects and corrects single-bit upsets:

1. Three identical neuron pipelines run in lockstep
2. A majority voter selects the correct output
3. If a mismatch is detected, `TMR_VOTER_STATUS` bit is set
4. The faulted pipeline is re-initialized from the voted output
5. Recovery takes < 4 clock cycles (16 ns)

This matches the software TMR implementation tested in
`disruption_predictor.py:apply_bit_flip_fault()`.

## 6. Demo Script

The demo script `src/scpn_fusion/control/hil_harness.py` provides a
`HILDemoRunner` class that:

1. Simulates the FPGA register interface in software
2. Maps Python SNN controller state to/from Q16.16 registers
3. Injects bit-flip faults and verifies TMR recovery
4. Measures simulated latency in clock cycles

```python
from scpn_fusion.control.hil_harness import HILDemoRunner

runner = HILDemoRunner()
runner.load_weights_from_controller(snn_controller)
runner.run_episode(n_steps=1000, inject_faults=True)
print(runner.report())
```

## 7. Deployment Notes

- **Vivado version**: 2024.1 or later recommended
- **HLS synthesis**: Not used; RTL hand-coded for deterministic latency
- **Power estimate**: ~15W for SNN inference at 250 MHz
- **Board support**: Alveo U250 shell v2.0 or compatible
- **Not included**: Actual bitstream generation (requires Vivado license + hardware)
