# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
from scpn_fusion.core.eped_pedestal import EpedPedestalModel

def test_eped():
    print("--- EPED Pedestal Model Test ---")

    # ITER parameters
    iter_model = EpedPedestalModel(
        R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0, kappa=1.7
    )
    res_iter = iter_model.predict(n_ped_1e19=8.0)
    print(f"ITER: T_ped = {res_iter.T_ped_keV:.2f} keV, Delta = {res_iter.Delta_ped:.3f}")

    # SPARC parameters
    sparc_model = EpedPedestalModel(
        R0=1.85, a=0.57, B0=12.2, Ip_MA=8.7, kappa=1.75
    )
    res_sparc = sparc_model.predict(n_ped_1e19=30.0) # High density SPARC
    print(f"SPARC: T_ped = {res_sparc.T_ped_keV:.2f} keV, Delta = {res_sparc.Delta_ped:.3f}")

if __name__ == "__main__":
    test_eped()
