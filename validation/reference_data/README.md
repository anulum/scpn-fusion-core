# Validation Reference Data

Experimental and design reference data for cross-validating SCPN Fusion Core
simulation outputs against real tokamak parameters.

## Datasets

### SPARC (Commonwealth Fusion Systems)

Equilibrium files from the [SPARCPublic](https://github.com/cfs-energy/SPARCPublic)
repository. SPARC is a compact, high-field (B = 12.2 T) tokamak under construction
by CFS, designed to achieve Q > 2 in D-T plasmas.

| File | Description | Grid | I_p (MA) | B_T (T) |
|------|-------------|------|----------|---------|
| `lmode_vv.geqdsk` | L-mode lower single-null, vertical-vertical divertor | 129x129 | 8.5 | 12.2 |
| `lmode_vh.geqdsk` | L-mode lower single-null, vertical-horizontal divertor | 129x129 | 8.5 | 12.2 |
| `lmode_hv.geqdsk` | L-mode lower single-null, horizontal-vertical divertor | 129x129 | 8.5 | 12.2 |
| `sparc_1300.eqdsk` | EQ library entry 1300 (low current, 0.2 MA) | 61x129 | 0.2 | 12.2 |
| `sparc_1305.eqdsk` | EQ library entry 1305 | 61x129 | — | 12.2 |
| `sparc_1310.eqdsk` | EQ library entry 1310 | 61x129 | — | 12.2 |
| `sparc_1315.eqdsk` | EQ library entry 1315 | 61x129 | — | 12.2 |
| `sparc_1349.eqdsk` | EQ library entry 1349 (full current, 8.0 MA) | 61x129 | 8.0 | 12.2 |
| `device_description.json` | IMAS-format device description (coils, wall) | — | — | — |
| `prd_popcon.csv` | Primary Reference Discharge POPCON data | — | — | — |

**License:** See https://github.com/cfs-energy/SPARCPublic for terms.

### ITPA H-Mode Confinement Database

Curated subset of the International Tokamak Physics Activity (ITPA) global
H-mode confinement database, covering 18 tokamaks worldwide.

| File | Description |
|------|-------------|
| `hmode_confinement.csv` | Machine parameters and measured τ_E for 20 entries |
| `ipb98y2_coefficients.json` | IPB98(y,2) scaling law coefficients and uncertainties |

**Source:** Verdoolaege et al., Nuclear Fusion 61 (2021) 076006

### ITER Configurations (existing)

Four ITER-scale validation configurations with different coil current optimisations:

| File | Description |
|------|-------------|
| `../iter_validated_config.json` | Human-designed baseline (I_p = 15 MA) |
| `../iter_analytic_config.json` | Analytically optimised coil currents |
| `../iter_force_balanced.json` | Newton-Raphson force-balanced equilibrium |
| `../iter_genetic_config.json` | Genetic algorithm optimised |

## Usage

```python
from scpn_fusion.core.eqdsk import read_geqdsk

eq = read_geqdsk("validation/reference_data/sparc/lmode_vv.geqdsk")
print(f"R_axis = {eq.rmaxis:.3f} m, B_T = {eq.bcentr:.2f} T")
print(f"ψ(R,Z) shape: {eq.psirz.shape}")
```
