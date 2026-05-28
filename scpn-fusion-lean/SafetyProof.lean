/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Lean Safety Proofs
-/
import SCPNFusionSolvers

namespace SCPNFusionSolvers

/-!
First-fusion safety contract:

`solveGradShafranov` must reject invalid physical case descriptions before any
numerical work is performed. This theorem gives a machine-checkable proof that
validation errors are propagated exactly as-is, so unsafe state cannot enter the
solver core.
-/
theorem solveGradShafranov_rejects_validation_error
    (c : GradShafranovCase) (err : String)
    (h : validateCase c = Except.error err) :
    solveGradShafranov c = Except.error err := by
  simp [solveGradShafranov, h]

end SCPNFusionSolvers
