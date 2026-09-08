====================================================
:mod:`scpn_fusion.engineering` -- Engineering
====================================================

The engineering subpackage provides balance-of-plant thermal cycle
models and CAD raytrace surface-loading estimation.

Balance of Plant
------------------

.. automodule:: scpn_fusion.engineering.balance_of_plant
   :members:
   :undoc-members:
   :show-inheritance:

CAD Raytrace Surface Loading
-------------------------------

.. automodule:: scpn_fusion.engineering.cad_raytrace
   :members:
   :undoc-members:
   :show-inheritance:

Coolant Channels
----------------

.. automodule:: scpn_fusion.engineering.thermal_hydraulics
   :members:
   :show-inheritance:

Plant cooling geometry must be supplied explicitly for a reactor design.
The default is a single pipe. Parallel paths share the total thermal load
equally; reported pressure drop is per path and electrical pumping power
is summed across paths. Headers and flow maldistribution are not modelled.
