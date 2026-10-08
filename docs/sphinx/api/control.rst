Control API
===========

The :mod:`scpn_fusion.control` subpackage contains reactor control algorithms,
digital twin infrastructure, disruption prediction, and mitigation systems.

Tokamak Flight Simulator
--------------------------

.. automodule:: scpn_fusion.control.tokamak_flight_sim
   :members:
   :undoc-members:
   :show-inheritance:

Tokamak Digital Twin
-----------------------

.. automodule:: scpn_fusion.control.tokamak_digital_twin
   :members:
   :undoc-members:
   :show-inheritance:

Digital Twin Ingest
---------------------

.. automodule:: scpn_fusion.control.digital_twin_ingest
   :members:
   :undoc-members:
   :show-inheritance:

Traceable Runtime (JAX/TorchScript)
-------------------------------------

The JAX control model requires binary64 computation. The application selects
X64 before creating arrays or tracing, for example through ``JAX_ENABLE_X64=1``
at process startup. Imports preserve that setting. Disabled X64 causes the
typed ``scpn_fusion.core.jax_precision.JaxPrecisionRefusal`` before a JAX
rollout; it does not trigger a different backend or widen a float32 result on
the host. The explicit NumPy and TorchScript routes retain their own policies.

.. automodule:: scpn_fusion.control.jax_traceable_runtime
   :members:
   :undoc-members:
   :show-inheritance:

Model-Predictive Control (Optimal)
------------------------------------

.. automodule:: scpn_fusion.control.fusion_optimal_control
   :members:
   :undoc-members:
   :show-inheritance:

Neural-Surrogate MPC
-----------------------

.. automodule:: scpn_fusion.control.neural_surrogate_mpc
   :members:
   :undoc-members:
   :show-inheritance:

Disruption Predictor
-----------------------

.. automodule:: scpn_fusion.control.disruption_predictor
   :members:
   :undoc-members:
   :show-inheritance:

Shattered Pellet Injection
----------------------------

.. automodule:: scpn_fusion.control.spi_mitigation
   :members:
   :undoc-members:
   :show-inheritance:

Free-Boundary Tracking
----------------------

.. autoclass:: scpn_fusion.control.free_boundary_tracking.FreeBoundaryTrackingController
   :members: identify_response_matrix, compute_correction, evaluate_objectives, evaluate_supervisor, run_tracking_shot
   :inherited-members:

Reduced Tritium-Breeding Proxy
------------------------------

.. autofunction:: scpn_fusion.control.disruption_contracts.mcnp_lite_tbr

The following example exercises the public mitigation contracts with finite
inputs. The breeding result is a reduced proxy, not a validated neutronics
prediction.

.. code-block:: python

   from scpn_fusion.control.disruption_contracts import mcnp_lite_tbr
   from scpn_fusion.control.spi_mitigation import ShatteredPelletInjection

   tbr, factor, bounds = mcnp_lite_tbr(
       base_tbr=0.96,
       li6_enrichment=0.92,
       be_multiplier_fraction=0.70,
       reflector_albedo=0.60,
       return_uncertainty=True,
   )
   assert bounds["tbr_p95_low"] <= tbr <= bounds["tbr_p95_high"]

   spi = ShatteredPelletInjection(Plasma_Energy_MJ=300.0, Plasma_Current_MA=15.0)
   times_ms, energies_mj, currents_ma = spi.trigger_mitigation(
       neon_quantity_mol=0.1, duration_s=0.001, verbose=False
   )
   assert len(times_ms) == len(energies_mj) == len(currents_ma)

Integrated Control Room
-------------------------

.. automodule:: scpn_fusion.control.fusion_control_room
   :members:
   :undoc-members:
   :show-inheritance:

Neuro-Cybernetic Controller
------------------------------

.. automodule:: scpn_fusion.control.neuro_cybernetic_controller
   :members:
   :undoc-members:
   :show-inheritance:

SOC Fusion Learning
---------------------

.. automodule:: scpn_fusion.control.advanced_soc_fusion_learning
   :members:
   :undoc-members:
   :show-inheritance:

Analytic Solver
-----------------

.. automodule:: scpn_fusion.control.analytic_solver
   :members:
   :undoc-members:
   :show-inheritance:

Director Interface
--------------------

.. automodule:: scpn_fusion.control.director_interface
   :members:
   :undoc-members:
   :show-inheritance:

Fueling Mode Controller
-------------------------

.. automodule:: scpn_fusion.control.fueling_mode
   :members:
   :undoc-members:
   :show-inheritance:

TORAX Hybrid Loop
--------------------

.. automodule:: scpn_fusion.control.torax_hybrid_loop
   :members:
   :undoc-members:
   :show-inheritance:

Real-Time EFIT
----------------

.. automodule:: scpn_fusion.control.realtime_efit
   :members:
   :undoc-members:
   :show-inheritance:

Plasma Shape Controller
-------------------------

.. automodule:: scpn_fusion.control.shape_controller
   :members:
   :undoc-members:
   :show-inheritance:

Vertical Stabiliser (Sliding Mode)
-------------------------------------

.. automodule:: scpn_fusion.control.sliding_mode_vertical
   :members:
   :undoc-members:
   :show-inheritance:

Fault-Tolerant Control
------------------------

.. automodule:: scpn_fusion.control.fault_tolerant_control
   :members:
   :undoc-members:
   :show-inheritance:

Safe RL Controller
--------------------

.. automodule:: scpn_fusion.control.safe_rl_controller
   :members:
   :undoc-members:
   :show-inheritance:

Scenario Scheduler
--------------------

.. automodule:: scpn_fusion.control.scenario_scheduler
   :members:
   :undoc-members:
   :show-inheritance:

Gain-Scheduled Controller
---------------------------

.. automodule:: scpn_fusion.control.gain_scheduled_controller
   :members:
   :undoc-members:
   :show-inheritance:
