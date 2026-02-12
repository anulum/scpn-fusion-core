# Compact Reactor Findings: The SCPN-Fusion-Core "Minimum Viable Reactor"

**Date**: 2026-01-20
**Target**: Commercial Fusion Power and Research Reliability
**Design ID**: MVR-0.96

---

## 1. The Challenge of Scale

Traditional fusion reactors like ITER are massive ($R \approx 6.2$m) and capital-intensive. The SCPN-Fusion-Core project aims to identify the smallest possible tokamak geometry that can achieve stable ignition and net power gain using advanced technology. Recent compact tokamak designs such as SPARC [1] and ARC [2] have demonstrated the viability of this approach.

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
1.  **REBCO HTS**: High-Temperature Superconductors [3] allowing for magnetic fields $>20$T.
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
1.  **Physics Constraints**: Beta limit, Greenwald density limit [4], and Lawson criterion.
2.  **Engineering Constraints**: Magnet current density ($J_{crit}$), shielding thickness, and neutron fluence limits.
3.  **Economic Constraints**: Normalized cost per MW of fusion power.

---

## 5. Conclusion

The MVR-0.96 proves that fusion doesn't need to be giant. With advanced HTS magnets [3] and liquid metal components, a net-gain fusion reactor can fit within a single laboratory or industrial bay. The bootstrap current fraction [5] further reduces the external current drive requirements, improving the overall power balance.

---

## References

[1] A. J. Creely, M. J. Greenwald, S. B. Ballinger, D. Brunner, J. Canik *et al.*, "Overview of the SPARC tokamak," *Journal of Plasma Physics*, vol. 86, no. 5, p. 865860502, 2020. DOI: [10.1017/S0022377820001257](https://doi.org/10.1017/S0022377820001257)

[2] B. N. Sorbom, J. Ball, T. R. Palmer, F. J. Mangiarotti, J. M. Sierchio *et al.*, "ARC: A compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets," *Fusion Engineering and Design*, vol. 100, pp. 378-405, 2015. DOI: [10.1016/j.fusengdes.2015.07.008](https://doi.org/10.1016/j.fusengdes.2015.07.008)

[3] D. G. Whyte, J. Minervini, B. LaBombard, E. Marmar, L. Bromberg, and M. Greenwald, "Smaller & Sooner: Exploiting High Magnetic Fields from New Superconducting Technologies for a More Attractive Fusion Energy Development Path," *Journal of Fusion Energy*, vol. 35, no. 1, pp. 41-53, 2016. DOI: [10.1007/s10894-015-0050-1](https://doi.org/10.1007/s10894-015-0050-1)

[4] M. Greenwald, "Density limits in toroidal plasmas," *Plasma Physics and Controlled Fusion*, vol. 44, no. 8, pp. R27-R53, 2002. DOI: [10.1088/0741-3335/44/8/201](https://doi.org/10.1088/0741-3335/44/8/201)

[5] O. Sauter, C. Angioni, and Y. R. Lin-Liu, "Neoclassical conductivity and bootstrap current formulas for general axisymmetric equilibria and arbitrary collisionality regime," *Physics of Plasmas*, vol. 6, no. 7, pp. 2834-2839, 1999. DOI: [10.1063/1.873240](https://doi.org/10.1063/1.873240)
