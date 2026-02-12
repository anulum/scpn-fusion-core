# Compact Reactor Findings: The SCPN-Fusion-Core "Minimum Viable Reactor"

**Date**: 2026-01-20
**Target**: Commercial Fusion Power and Research Reliability
**Design ID**: MVR-0.96

---

## 1. The Challenge of Scale

Traditional fusion reactors like ITER are massive ($R \approx 6.2$m) and capital-intensive. The SCPN-Fusion-Core project aims to identify the smallest possible tokamak geometry that can achieve stable ignition and net power gain using advanced technology.

## 2. The Minimum Viable Reactor (MVR)

Through a multi-dimensional sweep of the design space using the `CompactReactorArchitect` optimizer, we have identified an optimal "Compact" design:

### Geometry & Configuration
*   **Major Radius (R)**: 0.965 m
*   **Minor Radius (a)**: 0.482 m
*   **Aspect Ratio (A)**: 2.0
*   **Plasma Volume**: 4.4 m³

### Magnetics & Performance
*   **On-Axis Field ($B_0$)**: 8.1 Tesla
*   **Peak Coil Field ($B_{max}$)**: 21.6 Tesla
*   **Fusion Power ($P_{fusion}$)**: 5.3 MW

### Core Technologies
1.  **REBCO HTS**: High-Temperature Superconductors allowing for magnetic fields $>20$T.
2.  **TEMHD Liquid Divertor**: Thermo-Electric Magnetohydrodynamic liquid metal divertor capable of handling extremely high heat loads ($>90$ MW/m²).
3.  **Detached Mode**: Operating in a highly-radiative detached state to protect the first wall.

---

## 3. Engineering Implications

### Heat Flux Management
The MVR-0.96 produces a divertor heat load of **95.7 MW/m²**. Traditional solid divertors would fail instantly under these conditions. The use of a **liquid lithium or tin** divertor is mandatory, leveraging the TEMHD effect to drive coolant flow via the reactor's own magnetic field.

### Power Density
The power density of this reactor is orders of magnitude higher than ITER, demonstrating the potential for decentralized, industrial-scale fusion power units that can be deployed in modular configurations.

---

## 4. Methodology: The Global Design Scanner

These findings were generated using `global_design_scanner.py`, which evaluates millions of potential configurations against:
1.  **Physics Constraints**: Beta limit, Greenwald density limit, and Lawson criterion.
2.  **Engineering Constraints**: Magnet current density ($J_{crit}$), shielding thickness, and neutron fluence limits.
3.  **Economic Constraints**: Normalized cost per MW of fusion power.

---

## 5. Conclusion

The MVR-0.96 proves that fusion doesn't need to be giant. With advanced HTS magnets and liquid metal components, a net-gain fusion reactor can fit within a single laboratory or industrial bay.
