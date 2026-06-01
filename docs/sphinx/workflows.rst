Workflows
=========

3D Quickstart
-------------

Generate a 3D LCFS mesh OBJ from validated ITER config::

    python examples/run_3d_flux_quickstart.py --toroidal 24 --poloidal 24

Generate OBJ + PNG preview::

    python examples/run_3d_flux_quickstart.py --toroidal 24 --poloidal 24 --preview-png artifacts/SCPN_Plasma_3D_quickstart.png

Output file:

- ``artifacts/SCPN_Plasma_3D_quickstart.obj``
- ``artifacts/SCPN_Plasma_3D_quickstart.png`` (optional preview)

Validation RMSE Dashboard
-------------------------

Generate RMSE summary reports for confinement, beta_N surrogate, and SPARC axis error::

    python validation/rmse_dashboard.py

Default outputs:

- ``validation/reports/rmse_dashboard.json``
- ``validation/reports/rmse_dashboard.md``

Full-Fidelity Campaign Snapshot
-------------------------------

Generate the fail-closed production-parity snapshot::

    python validation/full_fidelity_end_to_end_campaign.py

Default outputs:

- ``validation/reports/full_fidelity_end_to_end_campaign.json``
- ``validation/reports/full_fidelity_end_to_end_campaign.md``

The snapshot must remain blocked when same-case external solver artefacts,
thresholds, or distributed runtime measurements are missing.  Do not replace
those rows with synthetic or reduced-order evidence.

Production Decomposition Contract
---------------------------------

Refresh the production-decomposition contract and local large-grid CPU
evidence::

    python validation/benchmark_production_decomposition_contract.py

Default outputs:

- ``validation/reports/production_decomposition_contract.json``
- ``validation/reports/production_decomposition_contract.md``

Distributed MPI or multi-GPU measurements can be supplied through the
``SCPN_PRODUCTION_DECOMPOSITION_DISTRIBUTED_RUNS_JSON`` environment variable.
Incomplete rows remain blocked.

Profiling
---------

Profile equilibrium solve hot paths::

    python profiling/profile_kernel.py --top 50

Profile 3D mesh generation hot paths::

    python profiling/profile_geometry_3d.py --toroidal 48 --poloidal 48 --top 50

Outputs are written under ``artifacts/profiling/`` by default.
