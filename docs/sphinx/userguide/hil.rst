==============================
HIL and FPGA Register Mapping
==============================

SCPN-Fusion-Core includes a synthetic hardware-in-the-loop (HIL) validation lane
and a software-simulated FPGA register interface for the SNN controller runtime.

Primary implementation surfaces:

- ``scpn_fusion.control.hil_harness`` (runtime + register-map simulation)
- ``validation/task7_hil_testing.py`` (strict validation gate)
- ``docs/hil_demo.md`` (full register map and latency budget)

HIL Scope
---------

The HIL lane validates:

- deterministic SNN closed-loop replay
- sub-millisecond control-loop latency gate (P95)
- bit-flip tolerance with TMR recovery checks
- register-level IO mapping for FPGA integration planning

Register Map (Core Offsets)
---------------------------

The software FPGA harness exposes a fixed register file. Core offsets:

.. list-table::
   :header-rows: 1
   :widths: 18 26 56

   * - Offset
     - Register
     - Purpose
   * - ``0x0020``
     - ``V_MEM[0..7]``
     - Membrane-state snapshot (Q16.16)
   * - ``0x0060``
     - ``INPUT[0..3]``
     - Controller inputs (Q16.16)
   * - ``0x0070``
     - ``OUTPUT[0..3]``
     - Controller outputs (Q16.16)
   * - ``0x0200``
     - ``LATENCY_CYCLES``
     - Last inference latency in FPGA cycles

See ``docs/hil_demo.md`` for the complete table and pin mapping.

Latency and Determinism
-----------------------

Release validation tracks both latency and deterministic replay:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Metric
     - Target
     - Validation Path
   * - P95 closed-loop latency
     - ``<= 1.0 ms``
     - ``validation/task7_hil_testing.py --strict``
   * - Replay determinism
     - exact match
     - repeated seeded runs in Task 7 + tests
   * - Register map integrity
     - pass
     - ``tests/test_hil_harness.py``

Reproduction
------------

Run the HIL regression lane locally:

.. code-block:: bash

   python -m pytest tests/test_hil_harness.py tests/test_task7_hil_testing.py -q
   python validation/task7_hil_testing.py --strict

